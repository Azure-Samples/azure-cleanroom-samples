<#
.SYNOPSIS
    Enables or disables execution consent on an existing document without republishing it.
#>
param(
    [Parameter(Mandatory)][string]$Persona,
    [Parameter(Mandatory)][string]$CollaborationId,
    [Parameter(Mandatory)][string]$DocumentId,
    [Parameter(Mandatory)][ValidateSet("enable", "disable")][string]$ConsentAction,
    [string]$Frontend,
    [string]$TokenFile,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/Invoke-Frontend.ps1"
$fe = Get-FrontendContext -Persona $Persona -Frontend $Frontend -TokenFile $TokenFile -DryRun:$DryRun
$id = [Uri]::EscapeDataString($CollaborationId)
$document = [Uri]::EscapeDataString($DocumentId)
Invoke-Frontend -Context $fe -Path "$id/consent/$document" -Method PUT `
    -Body @{ consentAction = $ConsentAction.ToLowerInvariant() }
if (-not $DryRun) { Write-Host "Execution consent set to '$ConsentAction' for '$DocumentId'." }
