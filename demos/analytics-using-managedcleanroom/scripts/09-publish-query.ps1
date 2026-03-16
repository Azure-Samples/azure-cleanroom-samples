<#
.SYNOPSIS
    Publishes a Spark SQL query WITHOUT az cleanroom CLI.

.DESCRIPTION
    Replaces: az cleanroom collaboration spark-sql query segment add
              az cleanroom collaboration spark-sql publish --prepare-only
    Uses: Hand-crafted query specification JSON

    Reads query segment files directly, builds the Query JSON structure,
    and publishes via the managed cleanroom frontend CLI.

    The query specification contains:
    - queryData: segments array with executionSequence, SQL data, preConditions, postFilters
    - inputDatasets: array of {datasetDocumentId, view} for each input
    - outputDataset: {datasetDocumentId, view} for the output

.PARAMETER collaborationId
    The collaboration ARM resource ID.

.PARAMETER queryName
    Name to assign to the query.

.PARAMETER queryDir
    Path to the directory containing segment*.txt files.

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

if (-not (Test-Path $queryDir)) {
    Write-Host "ERROR: Query directory '$queryDir' not found." -ForegroundColor Red
    exit 1
}

# Ensure temp directory.
$tempDir = Join-Path $outDir "temp"
if (-not (Test-Path $tempDir)) {
    New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
}

# Step 1: Read query segment files.
# Convention: segment1.txt and segment2.txt run in parallel (executionSequence=1),
# segment3.txt runs after them (executionSequence=2).
Write-Host "=== Step 1: Reading query segments from '$queryDir' ===" -ForegroundColor Cyan

$segments = @()

$segment1File = Join-Path $queryDir "segment1.txt"
$segment2File = Join-Path $queryDir "segment2.txt"
$segment3File = Join-Path $queryDir "segment3.txt"

if (Test-Path $segment1File) {
    $segments += @{
        executionSequence = 1
        data = (Get-Content $segment1File -Raw).Trim()
    }
    Write-Host "  Loaded segment1.txt (sequence 1)" -ForegroundColor Green
}

if (Test-Path $segment2File) {
    $segments += @{
        executionSequence = 1
        data = (Get-Content $segment2File -Raw).Trim()
    }
    Write-Host "  Loaded segment2.txt (sequence 1)" -ForegroundColor Green
}

if (Test-Path $segment3File) {
    $segments += @{
        executionSequence = 2
        data = (Get-Content $segment3File -Raw).Trim()
    }
    Write-Host "  Loaded segment3.txt (sequence 2)" -ForegroundColor Green
}

if ($segments.Count -eq 0) {
    Write-Host "ERROR: No segment files found in '$queryDir'." -ForegroundColor Red
    exit 1
}

Write-Host "  Total segments: $($segments.Count)" -ForegroundColor Yellow

# Step 2: Build the query specification JSON.
# This matches the output of `az cleanroom collaboration spark-sql publish --prepare-only`.
Write-Host "`n=== Step 2: Building query specification ===" -ForegroundColor Cyan

# The queryData field is a JSON object with a "segments" array.
$queryData = @{
    segments = $segments
}

$querySpec = @{
    queryData    = $queryData
    inputDatasets = @(
        @{
            datasetDocumentId = $publisherInputDataset
            view              = "publisher_data"
        }
        @{
            datasetDocumentId = $consumerInputDataset
            view              = "consumer_data"
        }
    )
    outputDataset = @{
        datasetDocumentId = $outputDataset
        view              = "output"
    }
}

$queryBodyFile = Join-Path $tempDir "query-$queryName-body.json"
$querySpec | ConvertTo-Json -Depth 10 | Out-File -FilePath $queryBodyFile -Encoding utf8
Write-Host "Query specification saved to: $queryBodyFile" -ForegroundColor Yellow

# Step 3: Publish via managed cleanroom frontend CLI (idempotent — skip if already published).
Write-Host "`n=== Step 3: Publishing query '$queryName' ===" -ForegroundColor Cyan

$existingQuery = Invoke-AzSafe @("managedcleanroom", "frontend", "analytics", "query", "show",
    "--collaboration-id", $collaborationId, "--document-id", $queryName)
if ($existingQuery) {
    Write-Host "Query '$queryName' already published (skipped)." -ForegroundColor Yellow
} else {
    az managedcleanroom frontend analytics query publish `
        --collaboration-id $collaborationId `
        --document-id $queryName `
        --body "@$queryBodyFile"
    Write-Host "Query '$queryName' published." -ForegroundColor Green
}

# Verify.
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
