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

Write-Log OperationStarted `
    "Showing output from cleanroom '$cleanRoomName' (${cleanroomEndpoint}) for '$persona' in" `
    "the '$demo' demo and contract '$contractId'..."

Write-Log Verbose `
    "$([environment]::NewLine)Invoking allowed cleanroom endpoint..."
curl -s -k --fail-with-body https://${cleanroomEndpoint}:8080

& {
    # Disable $PSNativeCommandUseErrorActionPreference for this scriptblock
    $PSNativeCommandUseErrorActionPreference = $false
    Write-Log Verbose `
        "$([environment]::NewLine)Invoking disallowed cleanroom endpoint..."
    curl -s -k --fail-with-body https://${cleanroomEndpoint}:8080/blah
    Write-Log Verbose `
        "$([environment]::NewLine)"
}

Write-Log OperationCompleted `
    "Completed showing output from cleanroom '$cleanRoomName' (${cleanroomEndpoint})" `
    "for '$persona' in the '$demo' demo and contract '$contractId'."
