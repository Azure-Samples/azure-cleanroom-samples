<#
.SYNOPSIS
    Cancels a non-terminal query using its full cl-spark-<uuid> job ID.
.DESCRIPTION
    Cancel persistent failures or unwanted runs, not a single transient mount warning.
    Returns the cancellation response; never submits another run.
#>
param(
    [Parameter(Mandatory)][string]$Persona,
    [Parameter(Mandatory)][string]$CollaborationId,
    [Parameter(Mandatory)][string]$QueryName,
    [Parameter(Mandatory)][Alias("RunId")][ValidatePattern("^cl-spark-.+")][string]$JobId,
    [string]$Frontend,
    [string]$TokenFile,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/Invoke-Frontend.ps1"
$fe = Get-FrontendContext -Persona $Persona -Frontend $Frontend -TokenFile $TokenFile -DryRun:$DryRun
$id = [Uri]::EscapeDataString($CollaborationId)
$query = [Uri]::EscapeDataString($QueryName)
$job = [Uri]::EscapeDataString($JobId)
Invoke-Frontend -Context $fe -Path "$id/analytics/queries/$query/runs/$job/cancel" -Method POST
