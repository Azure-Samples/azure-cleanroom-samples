<#
.SYNOPSIS
    Lists audit events for the analytics collaboration.

.DESCRIPTION
    Run by: Any collaborator (Northwind or Woodgrove).
    Retrieves and displays all audit events for the collaboration, showing
    actions taken by collaborators (dataset publishes, query votes, runs, etc.).

.PARAMETER collaborationId
    The collaboration frontend UUID.

.PARAMETER frontendEndpoint
    Frontend service URL.

.PARAMETER persona
    Persona (northwind or woodgrove) for naming/logging.
#>
param(
    [Parameter(Mandatory)]
    [string]$collaborationId,

    [Parameter(Mandatory)]
    [string]$frontendEndpoint,

    [string]$persona,

    [string]$TokenFile,

    [ValidateSet("rest", "cli")]
    [string]$ApiMode = "rest"
)

# Configure Private CleanRoom cloud and verify local user auth
. "$PSScriptRoot/common/setup-local-auth.ps1"

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# Load common frontend helpers (supports REST and CLI modes)
. "$PSScriptRoot/common/frontend-helpers.ps1"
$feCtx = New-FrontendContext -frontendEndpoint $frontendEndpoint -ApiMode $ApiMode

Write-Host "=== Audit Events ===" -ForegroundColor Cyan
$auditEvents = Get-FrontendAuditEvents -Context $feCtx `
    -CollaborationId $collaborationId `
    -TokenFile $TokenFile

if ($auditEvents) {
    # Response may be wrapped in { value: [...], nextLink: ... }
    $entries = if ($auditEvents.value) { $auditEvents.value }
               elseif ($auditEvents -is [array]) { $auditEvents }
               else { @($auditEvents) }
    $count = $entries.Count
    Write-Host "Found $count event(s)." -ForegroundColor Green

    foreach ($entry in $entries) {
        # Actual API fields: scope, id (category), timestamp, timestampIso, data.source, data.message
        $source = $entry.data.source
        if (-not $source) { $source = $entry.id }
        if (-not $source) { $source = "unknown" }
        $message = $entry.data.message
        if (-not $message) { $message = "" }
        $timestamp = $entry.timestampIso
        if (-not $timestamp) {
            # Convert epoch millis to ISO if timestampIso is empty
            if ($entry.timestamp) {
                try {
                    $epochMs = [long]$entry.timestamp
                    $dt = [DateTimeOffset]::FromUnixTimeMilliseconds($epochMs)
                    $timestamp = $dt.ToString("yyyy-MM-dd HH:mm:ss UTC")
                } catch {
                    $timestamp = $entry.timestamp
                }
            } else {
                $timestamp = ""
            }
        }

        Write-Host "  [$timestamp] ($source) $message" -ForegroundColor White
    }

    Write-Host "`nFull response:" -ForegroundColor Cyan
    $auditEvents | ConvertTo-Json -Depth 10
} else {
    Write-Host "No audit events found." -ForegroundColor Yellow
}
