param(
    [string]$demo = "$env:DEMO",

    [string]$persona = "$env:PERSONA",
    [string]$resourceGroup = "$env:RESOURCE_GROUP",

    [string]$samplesRoot = "/home/samples",
    [string]$privateDir = "$samplesRoot/demo-resources/private",
    [string]$publicDir = "$samplesRoot/demo-resources/public",

    [string]$contractConfig = "$privateDir/$resourceGroup-$demo.generated.json",
    [string]$contractFragment = "$privateDir/$persona-$demo.config",
    [string]$governanceClient = "azure-cleanroom-samples-governance-client-$persona",

    [string]$managedIdentityName = ""
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../common/common.psm1
Import-Module $PSScriptRoot/../azure-helpers/azure-helpers.psm1 -Force -DisableNameChecking

Write-Log OperationStarted `
    "Initializing cleanroom specification '$contractFragment'..." 

$personaUserId = $(az cleanroom governance client show --name $governanceClient --query userTokenClaims.oid -o tsv)

az cleanroom collaboration context add `
    --collaboration-name $governanceClient `
    --collaborator-id $personaUserId `
    --governance-client $governanceClient

if ($persona -eq "woodgrove" -and $demo -eq "analytics-s3-sse") {
    Write-Log Verbose `
        "Skipping any managed identity created for '$persona' for '$demo' demo..."
}
else {
    Test-AzureAccessToken

    if ($managedIdentityName -eq "") {
        $uniqueString = Get-UniqueString($resourceGroup)
        $managedIdentityName = "${uniqueString}-mi-$demo"
    }

    Write-Log OperationStarted `
        "Creating managed identity '$managedIdentityName' in resource group '$resourceGroup'..."
    $mi = (az identity create `
            --name $managedIdentityName `
            --resource-group $resourceGroup) | ConvertFrom-Json

    az cleanroom collaboration context set `
        --collaboration-name $governanceClient

    az cleanroom collaboration identity add az-federated `
        --identity-name "$persona-identity" `
        --client-id $mi.clientId `
        --tenant-id $mi.tenantId `
        --token-issuer-url $(Get-Content $publicDir/issuer.url) `
        --backing-identity cleanroom_cgs_oidc

    Write-Log OperationCompleted `
        "Added identity '$persona-identity' backed by '$managedIdentityName'."

    $configResult = @{
        contractFragment = ""
        mi               = @{}
    }
    $configResult.contractFragment = $contractFragment
    $configResult.mi = $mi

    $configResult | ConvertTo-Json -Depth 100 | Out-File $contractConfig
    Write-Log OperationCompleted `
        "Contract configuration written to '$contractConfig'."
    return $configResult
}