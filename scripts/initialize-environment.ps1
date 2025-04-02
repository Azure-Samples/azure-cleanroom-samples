param(
    [ValidateSet("mhsm", "akvpremium")]
    [string]$kvType = "akvpremium",

    [string]$persona = "$env:PERSONA",
    [string]$resourceGroup = "$env:RESOURCE_GROUP",
    [string]$resourceGroupLocation = "$env:RESOURCE_GROUP_LOCATION",

    [string]$samplesRoot = "/home/samples",
    [string]$privateDir = "$samplesRoot/demo-resources/private",
    [string]$secretDir = "$samplesRoot/demo-resources/secret",

    [string]$maaEndpoint = "https://sharedneu.neu.attest.azure.net",

    [string]$overridesFilePath = "",
    [string]$resourceGroupTags = "",

    [string]$environmentConfig = "$privateDir/$resourceGroup.generated.json",
    [string]$secretstoreConfig = "$privateDir/secretstores.config",
    [string]$localSecretStore = "$secretDir/$persona-local-store"
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/common/common.psm1

$isCollaborator = ($persona -eq "fabrikam") -or ($persona -eq "contosso")
$isDeveloper = ($persona -eq "litware")
$isOperator = ($persona -eq "operator")

if (!$isCollaborator -and !$isDeveloper -and !$isOperator)
{
    Write-Log Warning `
        "No initialization required for persona '$persona'."
    return
}

Import-Module $PSScriptRoot/azure-helpers/azure-helpers.psm1 -Force -DisableNameChecking

Test-AzureAccessToken

Write-Log OperationStarted `
    "Creating resource group '$resourceGroup' in '$resourceGroupLocation'..."
az group create --location $resourceGroupLocation --name $resourceGroup --tags $resourceGroupTags

$result = @{
    kek             = @{}
    dek             = @{}
    datasa          = @{}
    oidcsa          = @{}
    ccfsa           = @{}
    acr             = @{}
    maa_endpoint    = ""
}

$uniqueString = Get-UniqueString($resourceGroup)
if ($overridesFilePath -ne "")
{
    $overrides = Get-Content $overridesFilePath | Out-String | ConvertFrom-StringData
}
else
{
    $overrides = @{}
}

$objectId = GetLoggedInEntityObjectId

#
# Create secure key stores for:
#   a) Collaborators to store data set encyrption keys.
#   b) Application developer to store telemetry encyrption keys.
#
if ($isCollaborator -or $isDeveloper)
{
    $kvName = $($overrides['$KEYVAULT_NAME'] ?? "${uniqueString}kv")
    $mhsmName = $($overrides['$MHSM_NAME'] ?? "${uniqueString}mhsm")
    if ($kvType -eq "mhsm")
    {
        Write-Log OperationStarted `
            "Creating HSM '$mhsmName' in resource group '$resourceGroup'..."
        $keyStore = Create-Hsm `
            -resourceGroup $resourceGroup `
            -hsmName $mhsmName `
            -adminObjectId $objectId `
            -privateDir $privateDir
    
        $result.kek.kv = $keyStore
        # Creating the Key Vault upfront so as not to run into naming issues
        # while storing the wrapped DEK
        Write-Log OperationStarted `
            "Creating Key Vault '$kvName' to store the wrapped DEK..."
        $result.dek.kv = Create-KeyVault `
            -resourceGroup $resourceGroup `
            -keyVaultName $kvName `
            -adminObjectId $objectId
    }
    else
    {
        Write-Log OperationStarted `
            "Creating Key Vault '$kvName' in resource group '$resourceGroup'..."
        $result.kek.kv = Create-KeyVault `
            -resourceGroup $resourceGroup `
            -keyVaultName $kvName `
            -sku premium `
            -adminObjectId $objectId
        $result.dek.kv = $result.kek.kv
    }

    $result.maa_endpoint = $maaEndpoint

    Write-Log OperationStarted `
        "Generating secret store configuration for '$persona'..."
    
    Write-Log Verbose `
        "Adding local secret store '$localSecretStore'..."
    az cleanroom secretstore add `
        --name $persona-local-store `
        --config $secretstoreConfig `
        --backingstore-type Local_File `
        --backingstore-path $localSecretStore
    
    Write-Log Verbose `
        "Adding DEK store '$($result.dek.kv.id)'..."
    az cleanroom secretstore add `
        --name $persona-dek-store `
        --config $secretstoreConfig `
        --backingstore-type Azure_KeyVault `
        --backingstore-id $result.dek.kv.id 
    
    Write-Log Verbose `
        "Adding KEK store '$($result.kek.kv.id)'..."
    az cleanroom secretstore add `
        --name $persona-kek-store `
        --config $secretstoreConfig `
        --backingstore-type Azure_KeyVault_Managed_HSM `
        --backingstore-id $result.kek.kv.id `
        --attestation-endpoint $result.maa_endpoint
    
    Write-Log OperationCompleted `
        "Secret store configuration written to '$secretstoreConfig'."
}
else
{
    Write-Log Warning `
        "Skipped creation of secure key stores for persona '$persona'."
}

#
# Create data storage account for:
#   a) Collaborators to store data sets.
#   b) Application developer to store telemetry.
#
if ($isCollaborator -or $isDeveloper)
{
    $dataStorageAccount = $($overrides['$DATA_STORAGE_ACCOUNT_NAME'] ?? "datasa${uniqueString}")
    $result.datasa = Create-Storage-Resources `
        -resourceGroup $resourceGroup `
        -storageAccountName @($dataStorageAccount) `
        -objectId $objectId
}
else
{
    Write-Log Warning `
        "Skipped creation of data storage account for persona '$persona'."
}

#
# Create OIDC storage account for:
#   a) Collaborators to authenticate data sets access.
#   b) Application developer to authenticate telemetry access.
#
if ($isCollaborator -or $isDeveloper)
{
    $oidcStorageAccount = $($overrides['$OIDC_STORAGE_ACCOUNT_NAME'] ?? "oidcsa${uniqueString}")
    $result.oidcsa = Create-Storage-Resources `
        -resourceGroup $resourceGroup `
        -storageAccountName @($oidcStorageAccount) `
        -objectId $objectId
    az storage account update `
        --name $oidcStorageAccount `
        --resource-group $resourceGroup `
        --allow-blob-public-access true
    Write-Log OperationCompleted `
        "Enabled public blob access for '$oidcStorageAccount'."
}
else
{
    Write-Log Warning `
        "Skipped creation of OIDC storage account for persona '$persona'."
}

#
# Create CCF storage account for Operator to create CCF network.
#
if ($isOperator)
{
    $ccfStorageAccount = $($overrides['$CCF_STORAGE_ACCOUNT_NAME'] ?? "ccfsa${uniqueString}")
    $result.ccfsa = Create-Storage-Resources `
        -resourceGroup $resourceGroup `
        -storageAccountName @($ccfStorageAccount) `
        -objectId $objectId
    az storage account update `
        --name $ccfStorageAccount `
        --resource-group $resourceGroup `
        --allow-shared-key-access true
    Write-Log OperationCompleted `
        "Enabled shared key access for '$ccfStorageAccount'."
}
else
{
    Write-Log Warning `
        "Skipped creation of CCF storage account for persona '$persona'."
}

#
# Create ACR for application developer to push application images and OCI artefacts.
#
if ($isDeveloper)
{
    $acrName = "acr${uniqueString}"
    Write-Log Verbose `
        "Creating container registry '$acrName'..."
    $result.acr = (az acr create `
        --resource-group $resourceGroup `
        --name $acrName `
        --sku Standard) | ConvertFrom-Json

    Write-Log Verbose `
        "Enabling anonymous pull access for '$acrName'..."
    az acr update --name $acrName --anonymous-pull-enabled

    Write-Log OperationCompleted `
        "Created container registry '$acrName' and enabled anonymous pull access."
}
else
{
    Write-Log Warning `
        "Skipped creation of container registry for persona '$persona'."
}

$result | ConvertTo-Json -Depth 100 | Out-File "$environmentConfig"
Write-Log OperationCompleted `
    "Initialization configuration written to '$environmentConfig'."

return $result