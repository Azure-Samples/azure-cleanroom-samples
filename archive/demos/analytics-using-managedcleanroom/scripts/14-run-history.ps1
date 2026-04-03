<#
.SYNOPSIS
    Lists run history for a query in the analytics collaboration.

.DESCRIPTION
    Run by: Any collaborator (Northwind or Woodgrove).
    Retrieves all past runs for a given query, showing job IDs, states,
    and timestamps.

    Prerequisites:
    - 11-run-query.ps1 must have been run at least once.

.PARAMETER collaborationId
    The collaboration frontend UUID.

.PARAMETER queryName
    Name of the query to view history for (used as document-id).

.PARAMETER frontendEndpoint
    Frontend service URL.

.PARAMETER persona
    Persona (northwind or woodgrove) for naming/logging.
#>
param(
    [Parameter(Mandatory)]
    [string]$collaborationId,

    [Parameter(Mandatory)]
    [string]$queryName,

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

Write-Host "=== Run History for query '$queryName' ===" -ForegroundColor Cyan

# Use try/catch because the API returns NotFound (404) when no runs have completed yet
$runHistory = $null
try {
    $runHistory = Get-FrontendQueryRunHistory -Context $feCtx `
        -CollaborationId $collaborationId `
        -DocumentId $queryName `
        -TokenFile $TokenFile
}
catch {
    $errMsg = $_.Exception.Message
    if ($errMsg -match "NotFound|No run history|404") {
        Write-Host "No run history found for query '$queryName' (runs may still be in progress)." -ForegroundColor Yellow
        exit 0
    }
    throw
}

if ($runHistory) {
    # Response may have { runs: [...] }, { value: [...] }, be an array, or a single object
    $entries = if ($runHistory.runs) { $runHistory.runs }
               elseif ($runHistory.value) { $runHistory.value }
               elseif ($runHistory -is [array]) { $runHistory }
               else { @($runHistory) }
    $count = $entries.Count
    Write-Host "Found $count run(s)." -ForegroundColor Green

    foreach ($entry in $entries) {
        # Resolve job ID: runId > jobId > id
        $entryJobId = $entry.runId
        if (-not $entryJobId) { $entryJobId = $entry.jobId }
        if (-not $entryJobId) { $entryJobId = $entry.id }

        # Resolve state: applicationState.state > state > isSuccessful boolean
        $entryState = $entry.status.applicationState.state
        if (-not $entryState) { $entryState = $entry.state }
        if (-not $entryState -and $null -ne $entry.isSuccessful) {
            $entryState = if ($entry.isSuccessful) { "COMPLETED" } else { "FAILED" }
        }
        if (-not $entryState) { $entryState = "unknown" }

        $duration = $entry.durationSeconds
        $rowsRead = $entry.stats.rowsRead
        $rowsWritten = $entry.stats.rowsWritten
        $statsStr = ""
        if ($duration) { $statsStr += "  Duration: ${duration}s" }
        if ($null -ne $rowsRead) { $statsStr += "  Rows read: $rowsRead" }
        if ($null -ne $rowsWritten) { $statsStr += "  Rows written: $rowsWritten" }

        Write-Host "  Job: $entryJobId  State: $entryState$statsStr" -ForegroundColor $(
            if ($entryState -eq "COMPLETED") { "Green" }
            elseif ($entryState -in @("FAILED","SUBMISSION_FAILED")) { "Red" }
            else { "Yellow" }
        )
    }

    # Show summary if present
    if ($runHistory.summary) {
        $s = $runHistory.summary
        Write-Host "`nSummary: $($s.totalRuns) total, $($s.successfulRuns) succeeded, $($s.failedRuns) failed, avg $($s.avgDurationSeconds)s" -ForegroundColor Cyan
    }

    Write-Host "`nFull response:" -ForegroundColor Cyan
    $runHistory | ConvertTo-Json -Depth 10
} else {
    Write-Host "No run history found for query '$queryName'." -ForegroundColor Yellow
}
