<#
.SYNOPSIS
    Builds query publish JSON body from query segment files.

.DESCRIPTION
    Reads SQL segment files (segment*.json or segment*.txt) from a query
    directory and generates a ready-to-publish query JSON body.

    No frontend calls are made.

.PARAMETER queryName
    Name of the query (used as document-id).

.PARAMETER queryDir
    Path to directory containing segment*.json or segment*.txt files.

.PARAMETER publisherInputDataset
    Document ID of the publisher's (Northwind's) input dataset.

.PARAMETER consumerInputDataset
    Document ID of the consumer's (Woodgrove's) input dataset.

.PARAMETER outputDataset
    Document ID of the output dataset.

.PARAMETER outDir
    Output directory (default: ./generated).
#>
param(
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

# Resolve outDir to absolute path
$outDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($outDir)

if (-not (Test-Path $queryDir)) {
    Write-Host "ERROR: Query directory '$queryDir' not found." -ForegroundColor Red
    exit 1
}

# -- Detect format and load segments -----------------------------------------------
$jsonSegments = Get-ChildItem -Path $queryDir -Filter "segment*.json" -File | Sort-Object Name
$txtSegments = Get-ChildItem -Path $queryDir -Filter "segment*.txt" -File | Sort-Object Name

if ($jsonSegments.Count -gt 0) {
    Write-Host "Found $($jsonSegments.Count) JSON segment file(s)." -ForegroundColor Cyan
    $queryData = @()
    foreach ($seg in $jsonSegments) {
        $segContent = Get-Content $seg.FullName -Raw | ConvertFrom-Json
        $queryData += @{
            data              = $segContent.data
            executionSequence = $segContent.executionSequence
            preConditions     = if ($segContent.preConditions) { $segContent.preConditions } else { "" }
            postFilters       = if ($segContent.postFilters) { $segContent.postFilters } else { "" }
        }
    }
} elseif ($txtSegments.Count -gt 0) {
    Write-Host "Found $($txtSegments.Count) TXT segment file(s)." -ForegroundColor Cyan
    $queryData = @()
    $seq = 1
    foreach ($seg in $txtSegments) {
        $sql = (Get-Content $seg.FullName -Raw).Trim()
        # Check for metadata comment at top: -- seq=N
        if ($sql -match '^--\s*seq\s*=\s*(\d+)') {
            $seq = [int]$Matches[1]
            $sql = ($sql -replace '^--\s*seq\s*=\s*\d+\s*\n?', '').Trim()
        }
        $queryData += @{
            executionSequence = $seq
            data              = $sql
            preConditions     = ""
            postFilters       = ""
        }
    }
} else {
    Write-Host "ERROR: No segment*.json or segment*.txt files in '$queryDir'." -ForegroundColor Red
    exit 1
}

# -- Build query body --------------------------------------------------------------
$body = [ordered]@{
    inputDatasets = "${publisherInputDataset}:publisher_data,${consumerInputDataset}:consumer_data"
    outputDataset = "${outputDataset}:output"
    queryData     = @($queryData)
}

$publishDir = Join-Path $outDir "publish"
New-Item -ItemType Directory -Path $publishDir -Force | Out-Null

$outputFile = Join-Path $publishDir "$queryName.json"
$body | ConvertTo-Json -Depth 20 | Out-File -FilePath $outputFile -Encoding utf8

Write-Host "Query body: $outputFile" -ForegroundColor Green
Write-Host "  Segments: $($queryData.Count)" -ForegroundColor Yellow
Write-Host "  Publisher input: $publisherInputDataset" -ForegroundColor Yellow
Write-Host "  Consumer input: $consumerInputDataset" -ForegroundColor Yellow
Write-Host "  Output: $outputDataset" -ForegroundColor Yellow
