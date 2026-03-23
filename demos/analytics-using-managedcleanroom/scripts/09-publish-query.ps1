<#
.SYNOPSIS
    Publishes a Spark SQL query using parameterized CLI commands.

.DESCRIPTION
    Publishes a query by reading SQL segment files and using inline parameters
    instead of constructing JSON manually. Supports two modes:
    
    1. JSON segment files (segment*.json) - includes executionSequence in each file
    2. SQL text files (segment*.txt) - requires --execution-sequence parameter
    
    This script uses SQL text files (mode 2) for compatibility with existing demos.

.PARAMETER collaborationId
    The collaboration ARM resource ID.

.PARAMETER queryName
    Name to assign to the query.

.PARAMETER queryDir
    Path to the directory containing segment*.txt or segment*.json files.

.PARAMETER publisherInputDataset
    Name of the publisher's (Northwind) input dataset document ID.

.PARAMETER consumerInputDataset
    Name of the consumer's (Woodgrove) input dataset document ID.

.PARAMETER outputDataset
    Name of the output dataset document ID.

.PARAMETER outDir
    Output directory for generated metadata (default: ./generated).
#>
param(
    [Parameter(Mandatory)]
    [string]$collaborationId,

    [Parameter(Mandatory)]
    [string]$queryName,

    [Parameter(Mandatory)]
    [string]$queryDir,

    [Parameter(Mandatory)]
    [string]$publisherInputDataset,

    [Parameter(Mandatory)]
    [string]$consumerInputDataset,

    [Parameter(Mandatory)]
    [string]$outputDataset,

    [string]$outDir = "./generated"
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# --- Helper: call az and return $null on failure instead of throwing. ---
function Invoke-AzSafe {
    param([string[]]$Arguments)
    try {
        $prev = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $true
        $result = az @Arguments 2>$null
        $PSNativeCommandUseErrorActionPreference = $prev
        return $result
    }
    catch {
        $PSNativeCommandUseErrorActionPreference = $prev
        return $null
    }
}

# --- Helper: call az and return $null on failure instead of throwing. ---
function Invoke-AzSafe {
    param([string[]]$Arguments)
    try {
        $prev = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $true
        $result = az @Arguments 2>$null
        $PSNativeCommandUseErrorActionPreference = $prev
        return $result
    }
    catch {
        $PSNativeCommandUseErrorActionPreference = $prev
        return $null
    }
}

if (-not (Test-Path $queryDir)) {
    Write-Host "ERROR: Query directory '$queryDir' not found." -ForegroundColor Red
    exit 1
}

$segment1File = Join-Path $queryDir "segment1.txt"
$segment2File = Join-Path $queryDir "segment2.txt"
$segment3File = Join-Path $queryDir "segment3.txt"

if (-not (Test-Path $segment1File) -or -not (Test-Path $segment2File) -or -not (Test-Path $segment3File)) {
    Write-Host "ERROR: Missing segment files in '$queryDir'." -ForegroundColor Red
    exit 1
}

Write-Host "=== Publishing query '$queryName' ===" -ForegroundColor Cyan

$segment1Sql = (Get-Content $segment1File -Raw).Trim()
$segment2Sql = (Get-Content $segment2File -Raw).Trim()
$segment3Sql = (Get-Content $segment3File -Raw).Trim()

$existingQuery = Invoke-AzSafe @("managedcleanroom", "frontend", "analytics", "query", "show",
    "--collaboration-id", $collaborationId, "--document-id", $queryName)
if ($existingQuery) {
    Write-Host "Query '$queryName' already published (skipped)." -ForegroundColor Yellow
} else {
    az managedcleanroom frontend analytics query publish `
        --collaboration-id $collaborationId `
        --document-id $queryName `
        --query-segment $segment1Sql `
        --query-segment $segment2Sql `
        --query-segment $segment3Sql `
        --execution-sequence "1,1,2" `
        --input-datasets "${publisherInputDataset}:publisher_data,${consumerInputDataset}:consumer_data" `
        --output-dataset "${outputDataset}:output"
    Write-Host "Query '$queryName' published." -ForegroundColor Green
}

Write-Host "`nVerifying query state..." -ForegroundColor Cyan
az managedcleanroom frontend analytics query show `
    --collaboration-id $collaborationId `
    --document-id $queryName

Write-Host "`nAll collaborators must now vote to approve this query before execution." -ForegroundColor Yellow

# -- Enable execution consent on query -------------------------------------------
Write-Host "`n=== Enabling execution consent on query ===" -ForegroundColor Cyan
az managedcleanroom frontend consent set `
    --collaboration-id $collaborationId `
    --document-id $queryName `
    --consent-action enable
Write-Host "Execution consent enabled for query '$queryName'." -ForegroundColor Green
