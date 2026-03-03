<#
.SYNOPSIS
    Votes to accept a proposed query in the analytics collaboration.

.DESCRIPTION
    Run by: Each collaborator (Northwind and Woodgrove).
    Votes to accept the query via the managed cleanroom frontend CLI.
    Per the CLI help, --body is optional (accepts reason/metadata only),
    so no proposalId extraction is needed.

    Prerequisites:
    - 09-publish-query.ps1 must have been run.

.PARAMETER collaborationId
    The collaboration ARM resource ID.

.PARAMETER queryName
    Name of the query to vote on (used as the document-id).
#>
param(
    [Parameter(Mandatory)]
    [string]$collaborationId,

    [Parameter(Mandatory)]
    [string]$queryName
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# --- Helper: call az, skip on known "already done" errors, throw on real failures. ---
function Invoke-AzIdempotent {
    param(
        [string[]]$Arguments,
        [string]$ActionName = "Command"
    )
    try {
        $prev = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
        $output = az @Arguments 2>&1
        $exit = $LASTEXITCODE
        $PSNativeCommandUseErrorActionPreference = $prev

        if ($exit -ne 0) {
            $stderr = ($output | Where-Object { $_ -is [System.Management.Automation.ErrorRecord] }) -join "`n"
            $skipPattern = "already voted|already accepted|already approved|Conflict|duplicate"
            if ($stderr -match $skipPattern) {
                Write-Host "$ActionName — already done (skipped). Server said: $stderr" -ForegroundColor Yellow
                return $null
            }
            throw "$ActionName failed (exit $exit): $stderr"
        }
        $stdout = ($output | Where-Object { $_ -isnot [System.Management.Automation.ErrorRecord] }) -join "`n"
        return $stdout
    }
    catch {
        $PSNativeCommandUseErrorActionPreference = $prev
        throw
    }
}

# Step 1: Vote to accept the query (idempotent — skip if already voted).
Write-Host "=== Voting to accept query '$queryName' ===" -ForegroundColor Cyan

Invoke-AzIdempotent @("managedcleanroom", "frontend", "analytics", "query", "vote", "accept",
    "--collaboration-id", $collaborationId,
    "--document-id", $queryName) -ActionName "Vote accept for '$queryName'"

Write-Host "Vote submitted." -ForegroundColor Green

# Step 2: Verify query state.
Write-Host "`n=== Verifying query state ===" -ForegroundColor Cyan
az managedcleanroom frontend analytics query show `
    --collaboration-id $collaborationId `
    --document-id $queryName

Write-Host "`nVote complete for query '$queryName'." -ForegroundColor Green
