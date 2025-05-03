param(
    [Parameter(Mandatory = $true)]
    [string]$contractId,

    [string]$persona = "$env:PERSONA",

    [string]$samplesRoot = "/home/samples",
    [string]$privateDir = "$samplesRoot/demo-resources/private",
    [string]$publicDir = "$samplesRoot/demo-resources/public",
    [string]$demosRoot = "$samplesRoot/demos",

    [string]$cleanRoomName = "cleanroom-$contractId",
    [string]$cleanroomEndpoint = (Get-Content "$publicDir/$cleanRoomName.endpoint"),

    [string]$datastoreDir = "$privateDir/datastores",
    [string]$datastoreConfig = "$privateDir/datastores.config",

    [string]$demo = "$(Split-Path $PSScriptRoot -Leaf)",
    [string]$datasinkPath = "$demosRoot/$demo/datasink/$persona",

    [string]$cgsClient = "azure-cleanroom-samples-governance-client-$persona",
    [switch]$interactive
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../../scripts/common/common.psm1

if ($cleanroomEndpoint -eq '')
{
    Write-Log Warning `
        "No endpoint details available for cleanroom '$cleanRoomName' at" `
        "'$publicDir/$cleanRoomName.endpoint'."
    return
}

if (-not(Test-Path -Path $datasinkPath))
{
    Write-Log Warning `
        "No output available for persona '$persona' in demo '$demo'."
    return
}

Test-AzureAccessToken

Write-Log OperationStarted `
    "Showing output from cleanroom '$cleanRoomName' (${cleanroomEndpoint}) for '$persona' in" `
    "the '$demo' demo and contract '$contractId'..."

$dirs = Get-ChildItem -Path $datasinkPath -Directory -Name
foreach ($dir in $dirs)
{
    $datasinkName = "$persona-$dir".ToLower()
    Write-Log Verbose `
        "Enumerated datasink '$datasinkName' in '$datasinkPath'."

    $datastoreName = "$demo-$persona-$dir".ToLower()
    $datastoreFolder = "$datastoreDir/$datastoreName"
    Write-Log OperationStarted `
        "Downloading data for datasink '$datasinkName' ('$datastoreName') to" `
        "'$datastoreFolder'..."
    az cleanroom datastore download `
        --name $datastoreName `
        --config $datastoreConfig `
        --dst $datastoreDir
    Write-Log OperationCompleted `
        "Downloaded data for datasink '$datasinkName' ('$datastoreName') to" `
        "'$datastoreFolder'."

    # TODO: Understand why this is being copied into a nested folder.
    $datastoreFolder = "$datastoreFolder/**"
    Write-Log Information `
        "Output for datasink '$datasinkName' ('$datastoreName'):"
    Write-Log Verbose `
        "-----BEGIN OUTPUT-----" `
        "$($PSStyle.Reset)"
    Write-Log Verbose `
        "$([environment]::NewLine)-----END OUTPUT-----"
}

Write-Log OperationCompleted `
    "Completed showing output from cleanroom '$cleanRoomName' (${cleanroomEndpoint})" `
    "for '$persona' in the '$demo' demo and contract '$contractId'."
