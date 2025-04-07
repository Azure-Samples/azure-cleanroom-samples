param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("cleanroomhello-job", "cleanroomhello-api", "analytics", "inference", "training")]
    [string]$demo,

    [Parameter(Mandatory = $true)]
    [string]$contractId,

    [string]$persona = "$env:PERSONA",
    [string]$resourceGroup = "$env:RESOURCE_GROUP",

    [string]$samplesRoot = "/home/samples",
    [string]$publicDir = "$samplesRoot/demo-resources/public",
    [string]$privateDir = "$samplesRoot/demo-resources/private",
    [string]$secretDir = "$samplesRoot/demo-resources/secret",

    [string]$oidcContainerName = "cgs-oidc",

    [string]$secretstoreConfig = "$privateDir/secretstores.config",
    [string]$datastoreConfig = "$privateDir/datastores.config",
    [string]$environmentConfig = "$privateDir/$resourceGroup.generated.json",
    [string]$contractConfig = "$privateDir/$resourceGroup-$demo.generated.json",
    [string]$cgsClient = "azure-cleanroom-samples-governance-client-$persona"
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../common/common.psm1
Import-Module $PSScriptRoot/../azure-helpers/azure-helpers.psm1 -Force -DisableNameChecking

Test-AzureAccessToken

Write-Log OperationStarted `
    "Granting access to resources required for '$demo' demo to deployments implementing" `
    "contract '$contractId'..." 

$contractConfigResult = (Get-Content $contractConfig | ConvertFrom-Json)
$environmentConfigResult = Get-Content $environmentConfig | ConvertFrom-Json

#
# Create a KEK with SKR policy, wrap DEKs with the KEK and put in kv.
#
az cleanroom config wrap-deks `
    --contract-id $contractId `
    --cleanroom-config $contractConfigResult.contractFragment `
    --datastore-config $datastoreConfig `
    --secretstore-config $secretStoreConfig `
    --governance-client $cgsClient

#
# Setup managed identity access to storage/KV in collaborator tenant.
#
$managedIdentity = $contractConfigResult.mi

# Cleanroom needs both read/write permissions on storage account, hence assigning Storage Blob Data Contributor.
$role = "Storage Blob Data Contributor"
Write-Log Verbose `
    "Assigning permission for '$role' to '$($managedIdentity.name)' on" `
    "storage account '$($environmentConfigResult.datasa.name)'"
az role assignment create `
    --role "Storage Blob Data Contributor" `
    --scope $environmentConfigResult.datasa.id `
    --assignee-object-id $managedIdentity.principalId `
    --assignee-principal-type ServicePrincipal
CheckLastExitCode

# KEK vault access.
$kekVault = $environmentConfigResult.kek.kv
if ($kekVault.type -eq "Microsoft.KeyVault/managedHSMs") {
    $role = "Managed HSM Crypto User"

    $roleAssignment = (az keyvault role assignment list `
            --assignee-object-id $managedIdentity.principalId `
            --hsm-name $kekVault.name `
            --role $role) | ConvertFrom-Json
    if ($roleAssignment.Length -eq 1) {
        Write-Log Warning `
            "Skipping assignment as '$role' permission already exists for" `
            "'$($managedIdentity.name)' on mHSM '$($kekVault.name)'."
    }
    else {
        Write-Log Verbose `
            "Assigning permissions for '$role' to '$($managedIdentity.name)' on" `
            "mHSM '$($kekVault.name)'"
        az keyvault role assignment create `
            --role $role `
            --scope "/" `
            --assignee-object-id $managedIdentity.principalId `
            --hsm-name $kekVault.name `
            --assignee-principal-type ServicePrincipal
        CheckLastExitCode
    }
}
elseif ($kekVault.type -eq "Microsoft.KeyVault/vaults") {
    $role = "Key Vault Crypto Officer"

    $roleAssignment = (az role assignment list `
            --assignee $managedIdentity.principalId `
            --scope $kekVault.id `
            --role $role) | ConvertFrom-Json
    if ($roleAssignment.Length -eq 1) {
        Write-Log Warning `
            "Skipping assignment as '$role' permission already exists for" `
            "'$($managedIdentity.name)' on key vault '$($kekVault.name)'."
    }
    else {
        Write-Log Verbose `
            "Assigning permissions for '$role' to '$($managedIdentity.name)' on" `
            "key vault '$($kekVault.name)'"
        az role assignment create `
            --role $role `
            --scope $kekVault.id `
            --assignee-object-id $managedIdentity.principalId `
            --assignee-principal-type ServicePrincipal
        CheckLastExitCode
    }
}

# DEK vault access.
$dekVault = $environmentConfigResult.dek.kv
$role = "Key Vault Secrets User"
Write-Log Verbose `
    "Assigning permission for '$role' to '$($managedIdentity.name)' on" `
    "storage account '$($dekVault.name)'"
az role assignment create `
    --role $role `
    --scope $dekVault.id `
    --assignee-object-id $managedIdentity.principalId `
    --assignee-principal-type ServicePrincipal
CheckLastExitCode

#
# Setup OIDC issuer for tenant.
#
$tenantId = az account show --query "tenantId" --output tsv
$tenantData = (az cleanroom governance oidc-issuer show `
        --governance-client $cgsClient `
        --query "tenantData" | ConvertFrom-Json)
if ($null -ne $tenantData -and $tenantData.tenantId -eq $tenantId) {
    $issuerUrl = $tenantData.issuerUrl
    Write-Log Warning `
        "OIDC issuer already set for tenant '$tenantId' to '$issuerUrl'. Skipping!"
}
else {
    $oidcsa = $environmentConfigResult.oidcsa.name
    Write-Log Verbose `
        "Setting up OIDC issuer for tenant '$tenantId' using storage account '$oidcsa'..."

    Write-Log Verbose `
        "Creating public access blob container '$oidcContainerName' in '$oidcsa'..."
    az storage container create `
        --name $oidcContainerName `
        --account-name $oidcsa `
        --public-access blob `
        --auth-mode login
    CheckLastExitCode
    Write-Log OperationCompleted `
        "Created public access blob container '$oidcContainerName' in '$oidcsa'."

    Write-Log Verbose `
        "Uploading openid-configuration to container '$oidcContainerName' in '$oidcsa'..." `
        "$($PSStyle.Reset)"
    @"
{
"issuer": "https://$oidcsa.blob.core.windows.net/$oidcContainerName",
"jwks_uri": "https://$oidcsa.blob.core.windows.net/$oidcContainerName/openid/v1/jwks",
"response_types_supported": [
"id_token"
],
"subject_types_supported": [
"public"
],
"id_token_signing_alg_values_supported": [
"RS256"
]
}
"@ | Out-File $privateDir/openid-configuration.json
    az storage blob upload `
        --container-name $oidcContainerName `
        --file $privateDir/openid-configuration.json `
        --name .well-known/openid-configuration `
        --account-name $oidcsa `
        --overwrite `
        --auth-mode login
    CheckLastExitCode

    Write-Log Verbose `
        "Uploading jwks to container '$oidcContainerName' in '$oidcsa'..."
    $ccfEndpoint = (Get-Content "$publicDir/ccfEndpoint.json" | ConvertFrom-Json)
    $url = "$($ccfEndpoint.url)/app/oidc/keys"
    curl -sL -k $url | jq | Out-File $privateDir/jwks.json
    az storage blob upload `
        --container-name $oidcContainerName `
        --file $privateDir/jwks.json `
        --name openid/v1/jwks `
        --account-name $oidcsa `
        --overwrite `
        --auth-mode login
    CheckLastExitCode

    Write-Log Verbose `
        "Setting OIDC issuer for tenant '$tenantId'..."
    az cleanroom governance oidc-issuer set-issuer-url `
        --governance-client $cgsClient `
        --url "https://$oidcsa.blob.core.windows.net/$oidcContainerName"
    $tenantData = (az cleanroom governance oidc-issuer show `
            --governance-client $cgsClient `
            --query "tenantData" | ConvertFrom-Json)
    $issuerUrl = $tenantData.issuerUrl

    Write-Log OperationCompleted `
        "Set OIDC issuer for tenant '$tenantId' to '$issuerUrl'."
}

#
# Setup federated credential on managed identity.
#
Write-Log OperationStarted `
    "Setting up federation on managed identity '$($managedIdentity.name)' for" `
    "issuer '$issuerUrl' and subject '$contractId'..."
az identity federated-credential create `
    --name "$contractId-federation" `
    --identity-name $managedIdentity.name `
    --resource-group $resourceGroup `
    --issuer $issuerUrl `
    --subject $contractId

# See Note at https://learn.microsoft.com/en-us/azure/aks/workload-identity-deploy-cluster#create-the-federated-identity-credential
$sleepTime = 30
Write-Log Verbose `
    "Waiting for $sleepTime seconds for federated identity credential to propagate..."
Start-Sleep -Seconds $sleepTime

Write-Log OperationCompleted `
    "Granted access to resources required for '$demo' demo to deployments implementing" `
    "contract '$contractId' through federation on managed identity '$($managedIdentity.name)'." 