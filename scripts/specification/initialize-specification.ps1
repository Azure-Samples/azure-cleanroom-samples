param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("cleanroomhello-job", "cleanroomhello-api", "analytics", "inference")]
    [string]$demo,

    [string]$persona = "$env:PERSONA",
    [string]$resourceGroup = "$env:RESOURCE_GROUP",

    [string]$samplesRoot = "/home/samples",
    [string]$privateDir = "$samplesRoot/demo-resources/private",
    [string]$publicDir = "$samplesRoot/demo-resources/public",

    [string]$contractConfig = "$privateDir/$resourceGroup-$demo.generated.json",
    [string]$contractFragment = "$publicDir/$persona-$demo.config",

    [string]$managedIdentityName = ""
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../common/common.psm1
Import-Module $PSScriptRoot/../azure-helpers/azure-helpers.psm1 -Force -DisableNameChecking

Test-AzureAccessToken

Write-Log OperationStarted `
    "Initializing cleanroom specification '$contractFragment'..." 
az cleanroom config init `
    --cleanroom-config $contractFragment

if ($managedIdentityName -eq "")
{
    $uniqueString = Get-UniqueString($resourceGroup)
    $managedIdentityName = "${uniqueString}-mi-$demo"
}

Write-Log OperationStarted `
    "Creating managed identity '$managedIdentityName' in resource group '$resourceGroup'..."
$mi = (az identity create `
    --name $managedIdentityName `
    --resource-group $resourceGroup) | ConvertFrom-Json
az cleanroom config add-identity az-federated `
    --cleanroom-config $contractFragment `
    -n "$persona-identity" `
    --client-id $mi.clientId `
    --tenant-id $mi.tenantId `
    --backing-identity cleanroom_cgs_oidc
Write-Log OperationCompleted `
    "Added identity '$persona-identity' backed by '$managedIdentityName'."

$configResult = @{
    contractFragment = ""
    mi         = @{}
}
$configResult.contractFragment = $contractFragment
$configResult.mi = $mi

$configResult | ConvertTo-Json -Depth 100 | Out-File $contractConfig
Write-Log OperationCompleted `
    "Contract configuration written to '$contractConfig'."
return $configResult