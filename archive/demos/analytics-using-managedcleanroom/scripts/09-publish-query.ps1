<#
.SYNOPSIS
    Publishes a Spark SQL query to the collaboration.

.DESCRIPTION
    Run by: Woodgrove (consumer).
    Publishes a query by reading SQL segment files (segment*.txt or segment*.json)
    and calling the frontend API. Auto-detects file format from directory contents.

    Supports both REST API and az managedcleanroom CLI modes via -ApiMode parameter.

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

.PARAMETER frontendEndpoint
    Frontend service URL.

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

    [Parameter(Mandatory)]
    [string]$frontendEndpoint,

    [string]$outDir = "./generated",

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

if (-not (Test-Path $queryDir)) {
    Write-Host "ERROR: Query directory '$queryDir' not found." -ForegroundColor Red
    exit 1
}

# Auto-detect segment file format: prefer JSON, fall back to TXT
$jsonSegments = Get-ChildItem -Path $queryDir -Filter "segment*.json" -ErrorAction SilentlyContinue | Sort-Object Name
$txtSegments  = Get-ChildItem -Path $queryDir -Filter "segment*.txt" -ErrorAction SilentlyContinue | Sort-Object Name

$queryData = @()

if ($jsonSegments.Count -gt 0) {
    Write-Host "=== Publishing query '$queryName' (JSON segments) ===" -ForegroundColor Cyan
    Write-Host "Found $($jsonSegments.Count) segment files:" -ForegroundColor Yellow
    $jsonSegments | ForEach-Object { Write-Host "  - $($_.Name)" -ForegroundColor Gray }

    foreach ($segFile in $jsonSegments) {
        $seg = Get-Content $segFile.FullName -Raw | ConvertFrom-Json
        $queryData += @{
            data              = $seg.data
            executionSequence = $seg.executionSequence
            preConditions     = if ($seg.preConditions) { $seg.preConditions } else { "" }
            postFilters       = if ($seg.postFilters) { $seg.postFilters } else { "" }
        }
    }
} elseif ($txtSegments.Count -ge 3) {
    Write-Host "=== Publishing query '$queryName' (TXT segments) ===" -ForegroundColor Cyan

    $segment1Sql = (Get-Content (Join-Path $queryDir "segment1.txt") -Raw).Trim()
    $segment2Sql = (Get-Content (Join-Path $queryDir "segment2.txt") -Raw).Trim()
    $segment3Sql = (Get-Content (Join-Path $queryDir "segment3.txt") -Raw).Trim()

    $queryData = @(
        @{ data = $segment1Sql; executionSequence = 1; preConditions = ""; postFilters = "" },
        @{ data = $segment2Sql; executionSequence = 1; preConditions = ""; postFilters = "" },
        @{ data = $segment3Sql; executionSequence = 2; preConditions = ""; postFilters = "" }
    )
} else {
    Write-Host "ERROR: No segment*.json or segment*.txt files found in '$queryDir'." -ForegroundColor Red
    exit 1
}

# Check if already published
$existingQuery = Get-FrontendQuery -Context $feCtx -CollaborationId $collaborationId -DocumentId $queryName -TokenFile $TokenFile
if ($existingQuery) {
    Write-Host "Query '$queryName' already published (skipped)." -ForegroundColor Yellow
} else {
    $queryBody = @{
        inputDatasets = "${publisherInputDataset}:publisher_data,${consumerInputDataset}:consumer_data"
        outputDataset = "${outputDataset}:output"
        queryData     = $queryData
    }

    Publish-FrontendQuery -Context $feCtx `
        -CollaborationId $collaborationId `
        -DocumentId $queryName `
        -Body $queryBody `
        -TokenFile $TokenFile
    Write-Host "Query '$queryName' published." -ForegroundColor Green
}

Write-Host "`nVerifying query state..." -ForegroundColor Cyan
$queryInfo = Get-FrontendQuery -Context $feCtx -CollaborationId $collaborationId -DocumentId $queryName -TokenFile $TokenFile
if ($queryInfo) {
    $queryInfo | ConvertTo-Json -Depth 10
}

Write-Host "`nAll collaborators must now vote to approve this query before execution." -ForegroundColor Yellow
