<#
.SYNOPSIS
    Publishes a Spark SQL query to the collaboration.

.DESCRIPTION
    Run by: Woodgrove (consumer).
    Publishes a query by reading SQL segment files and calling the frontend
    REST API directly (replaces broken az managedcleanroom frontend CLI).

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

.PARAMETER persona
    Persona (northwind or woodgrove) for naming/logging.
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

# Check if already published
$existingQuery = Get-FrontendQuery -Context $feCtx -CollaborationId $collaborationId -DocumentId $queryName -TokenFile $TokenFile
if ($existingQuery) {
    Write-Host "Query '$queryName' already published (skipped)." -ForegroundColor Yellow
} else {
    # Build query publish body
    # Body structure verified from CLI source (_frontend_custom.py, lines 654-658):
    #   inputDatasets (comma-separated "ds:view" pairs), outputDataset ("ds:view"),
    #   queryData (array of {data, executionSequence, preConditions, postFilters})
    #
    # Execution sequences from the demo: segment1=1, segment2=1, segment3=2
    # (segments 1 and 2 run in parallel, segment 3 runs after both complete)
    $queryBody = @{
        inputDatasets = "${publisherInputDataset}:publisher_data,${consumerInputDataset}:consumer_data"
        outputDataset = "${outputDataset}:output"
        queryData     = @(
            @{
                data              = $segment1Sql
                executionSequence = 1
                preConditions     = ""
                postFilters       = ""
            },
            @{
                data              = $segment2Sql
                executionSequence = 1
                preConditions     = ""
                postFilters       = ""
            },
            @{
                data              = $segment3Sql
                executionSequence = 2
                preConditions     = ""
                postFilters       = ""
            }
        )
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
Write-Host "NOTE: Execution consent can only be enabled AFTER the query is accepted (voted on)." -ForegroundColor Yellow
Write-Host "Run 10-vote-query.ps1 next — it will vote AND enable consent." -ForegroundColor Yellow
