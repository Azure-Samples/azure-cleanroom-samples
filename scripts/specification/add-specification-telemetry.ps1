param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("cleanroomhello-job", "cleanroomhello-api", "analytics", "inference")]
    [string]$demo,

    [string]$persona = "$env:PERSONA",
    [string]$resourceGroup = "$env:RESOURCE_GROUP",
    [string]$kekName = $($($(New-Guid).Guid) -replace '-').ToLower(),

    [string]$samplesRoot = "/home/samples",
    [string]$privateDir = "$samplesRoot/demo-resources/private",
    [string]$secretDir = "$samplesRoot/demo-resources/secret",

    [string]$contractConfig = "$privateDir/$resourceGroup-$demo.generated.json",
    [string]$environmentConfig = "$privateDir/$resourceGroup.generated.json",
    [string]$secretstoreConfig = "$privateDir/secretstores.config",
    [string]$datastoreConfig = "$privateDir/datastores.config"
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../common/common.psm1

$contractConfigResult = Get-Content $contractConfig | ConvertFrom-Json
$environmentConfigResult = Get-Content $environmentConfig | ConvertFrom-Json

Write-Log OperationStarted `
    "Adding telemetry details for '$persona' in the '$demo' demo to" `
    "'$($contractConfigResult.contractFragment)'..."

# $result below refers to the output of the prepare-resources.ps1 that was run earlier.
az cleanroom config set-logging `
    --cleanroom-config $contractConfigResult.contractFragment `
    --datastore-config $datastoreConfig `
    --storage-account $environmentConfigResult.datasa.id `
    --identity "$persona-identity" `
    --secretstore-config $secretStoreConfig `
    --datastore-secret-store $persona-local-store `
    --dek-secret-store $persona-dek-store `
    --kek-secret-store $persona-kek-store `
    --kek-name $kekName `
    --encryption-mode CPK
Write-Log OperationCompleted `
    "Added application telemetry details."

az cleanroom config set-telemetry `
    --cleanroom-config $contractConfigResult.contractFragment `
    --datastore-config $datastoreConfig `
    --storage-account $environmentConfigResult.datasa.id `
    --identity "$persona-identity" `
    --secretstore-config $secretStoreConfig `
    --datastore-secret-store $persona-local-store `
    --dek-secret-store $persona-dek-store `
    --kek-secret-store $persona-kek-store `
    --kek-name $kekName `
    --encryption-mode CPK
Write-Log OperationCompleted `
    "Added infrastructure telemetry details."

