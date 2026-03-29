<#
.SYNOPSIS
    Publishes a Spark SQL query using JSON segment files.

.DESCRIPTION
    Publishes a query by reading segment*.json files (which include executionSequence,
    data, preConditions, postFilters). Supports both REST API and az managedcleanroom CLI modes via -ApiMode parameter.

.PARAMETER collaborationId
    The collaboration ARM resource ID.

.PARAMETER queryName
    Name to assign to the query.

.PARAMETER queryDir
    Path to the directory containing segment*.json files.

.PARAMETER publisherInputDataset
    Name of the publisher's (Northwind) input dataset document ID.

.PARAMETER consumerInputDataset
    Name of the consumer's (Woodgrove) input dataset document ID.

.PARAMETER outputDataset
    Name of the output dataset document ID.

.PARAMETER frontendEndpoint
    Frontend service URL.

.PARAMETER persona
    Persona (northwind or woodgrove) for naming/logging.

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

    [string]$persona,
    [string]$outDir = "./generated",

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

# Discover segment JSON files dynamically (segment1.json, segment2.json, ...)
$segmentFiles = Get-ChildItem -Path $queryDir -Filter "segment*.json" | Sort-Object Name
if ($segmentFiles.Count -eq 0) {
    Write-Host "ERROR: No segment*.json files found in '$queryDir'." -ForegroundColor Red
    exit 1
}

Write-Host "=== Publishing query '$queryName' (JSON mode) ===" -ForegroundColor Cyan
Write-Host "Found $($segmentFiles.Count) segment files:" -ForegroundColor Yellow
$segmentFiles | ForEach-Object { Write-Host "  - $($_.Name)" -ForegroundColor Gray }

# Read segment JSON files into queryData array
$queryData = @()
foreach ($segFile in $segmentFiles) {
    $seg = Get-Content $segFile.FullName -Raw | ConvertFrom-Json
    $queryData += @{
        data              = $seg.data
        executionSequence = $seg.executionSequence
        preConditions     = if ($seg.preConditions) { $seg.preConditions } else { "" }
        postFilters       = if ($seg.postFilters) { $seg.postFilters } else { "" }
    }
}

# Check if already published
$existingQuery = Get-FrontendQuery -Context $feCtx -CollaborationId $collaborationId -DocumentId $queryName
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
        -Body $queryBody
    Write-Host "Query '$queryName' published." -ForegroundColor Green
}

Write-Host "`nVerifying query state..." -ForegroundColor Cyan
$queryInfo = Get-FrontendQuery -Context $feCtx -CollaborationId $collaborationId -DocumentId $queryName
if ($queryInfo) {
    $queryInfo | ConvertTo-Json -Depth 10
}

Write-Host "`nAll collaborators must now vote to approve this query before execution." -ForegroundColor Yellow

# -- Enable execution consent on query -------------------------------------------
Write-Host "`n=== Enabling execution consent on query ===" -ForegroundColor Cyan
Set-FrontendConsent -Context $feCtx -CollaborationId $collaborationId -DocumentId $queryName -Action "enable"
Write-Host "Execution consent enabled for query '$queryName'." -ForegroundColor Green
