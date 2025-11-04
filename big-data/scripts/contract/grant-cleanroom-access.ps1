param(
    [string]$demo = "$env:DEMO",

    [string]$persona = "$env:PERSONA",
    [string]$resourceGroup = "$env:RESOURCE_GROUP",

    [string]$samplesRoot = "/home/samples",
    [string]$publicDir = "$samplesRoot/demo-resources/public",
    [string]$privateDir = "$samplesRoot/demo-resources/private",

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

if ($persona -eq "woodgrove" -and $demo -eq "analytics-s3-sse") {
    Write-Log Verbose `
        "No grant access step required for '$persona' for '$demo' demo..."
    return
}

Test-AzureAccessToken


$contractId = Get-Content $publicDir/analytics.contract-id

Write-Log OperationStarted `
    "Granting access to resources required for '$demo' demo to deployments implementing" `
    "contract '$contractId'..." 

$contractConfigResult = (Get-Content $contractConfig | ConvertFrom-Json)
$environmentConfigResult = Get-Content $environmentConfig | ConvertFrom-Json

#
# Setup managed identity access to storage/KV in collaborator tenant.
#
$managedIdentity = $contractConfigResult.mi

# Cleanroom needs both read/write permissions on storage account, hence assigning Storage Blob Data Contributor.
$role = "Storage Blob Data Contributor"
$roleAssignment = (az role assignment list `
        --assignee-object-id $managedIdentity.principalId `
        --scope $environmentConfigResult.datasa.id `
        --role $role `
        --fill-principal-name false `
        --fill-role-definition-name false) | ConvertFrom-Json
if ($roleAssignment.Length -eq 1) {
    Write-Log Warning `
        "Skipping assignment as '$role' permission already exists for" `
        "'$($managedIdentity.name)' on storage account '$($environmentConfigResult.datasa.name)'."
}
else {
    Write-Log Verbose `
        "Assigning permission for '$role' to '$($managedIdentity.name)' on" `
        "storage account '$($environmentConfigResult.datasa.name)'"
    az role assignment create `
        --role $role `
        --scope $environmentConfigResult.datasa.id `
        --assignee-object-id $managedIdentity.principalId `
        --assignee-principal-type ServicePrincipal
}

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
    }
}
elseif ($kekVault.type -eq "Microsoft.KeyVault/vaults") {
    $role = "Key Vault Crypto Officer"

    $roleAssignment = (az role assignment list `
            --assignee-object-id $managedIdentity.principalId `
            --scope $kekVault.id `
            --role $role `
            --fill-principal-name false `
            --fill-role-definition-name false) | ConvertFrom-Json
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
    }
}

# DEK vault access.
$dekVault = $environmentConfigResult.dek.kv
$role = "Key Vault Secrets User"
$roleAssignment = (az role assignment list `
        --assignee-object-id $managedIdentity.principalId `
        --scope $dekVault.id `
        --role $role `
        --fill-principal-name false `
        --fill-role-definition-name false) | ConvertFrom-Json
if ($roleAssignment.Length -eq 1) {
    Write-Log Warning `
        "Skipping assignment as '$role' permission already exists for" `
        "'$($managedIdentity.name)' on key vault '$($dekVault.name)'."
}
else {
    Write-Log Verbose `
        "Assigning permission for '$role' to '$($managedIdentity.name)' on" `
        "storage account '$($dekVault.name)'"
    az role assignment create `
        --role $role `
        --scope $dekVault.id `
        --assignee-object-id $managedIdentity.principalId `
        --assignee-principal-type ServicePrincipal
}

#
# Setup federated credential on managed identity.
#
$issuerUrl = Get-Content $publicDir/issuer.url
$userId = az cleanroom governance client show --name $cgsClient --query userTokenClaims.oid -o tsv
$subject = $contractId + "-" + $userId
Write-Log OperationStarted `
    "Setting up federation on managed identity '$($managedIdentity.name)' for" `
    "issuer '$issuerUrl' and subject '$subject'..."
az identity federated-credential create `
    --name "$subject-federation" `
    --identity-name $managedIdentity.name `
    --resource-group $resourceGroup `
    --issuer $issuerUrl `
    --subject $subject

# See Note at https://learn.microsoft.com/en-us/azure/aks/workload-identity-deploy-cluster#create-the-federated-identity-credential
$sleepTime = 30
Write-Log Verbose `
    "Waiting for $sleepTime seconds for federated identity credential to propagate..."
Start-Sleep -Seconds $sleepTime

Write-Log OperationCompleted `
    "Granted access to resources required for '$demo' demo to deployments implementing" `
    "contract '$contractId' through federation on managed identity '$($managedIdentity.name)'." 