<#
.SYNOPSIS
    Views query run history, audit events, and optionally downloads CPK-encrypted output.

.DESCRIPTION
    Run by: Any collaborator (Northwind or Woodgrove).
    Retrieves and displays query run history and collaboration audit events
    via the frontend service. Supports both REST API and az managedcleanroom CLI modes via -ApiMode parameter.

    For CPK datasets, can also download and decrypt the output using azcopy --cpk-by-value.

    Prerequisites:
    - 11-run-query.ps1 must have been run at least once.

.PARAMETER collaborationId
    The collaboration ARM resource ID.

.PARAMETER queryName
    Name of the query to view history for (used as document-id).

.PARAMETER frontendEndpoint
    Frontend service URL.

.PARAMETER persona
    Persona (northwind or woodgrove) for naming/logging.

.PARAMETER DownloadCpkOutput
    If set, downloads CPK-encrypted output using azcopy --cpk-by-value.

.PARAMETER OutputDekFile
    Path to the output DEK binary file (required if -DownloadCpkOutput is set).

.PARAMETER OutputStorageAccount
    Storage account name for output (required if -DownloadCpkOutput is set).

.PARAMETER OutputContainer
    Output container name (default: woodgrove-output).

.PARAMETER OutputLocalDir
    Local directory to download output to (default: ./generated/output).
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
    [string]$ApiMode = "rest",

    [switch]$DownloadCpkOutput,

    [string]$OutputDekFile,

    [string]$OutputStorageAccount,

    [string]$OutputContainer = "woodgrove-output",

    [string]$OutputLocalDir = "./generated/output"
)

# Configure Private CleanRoom cloud and verify local user auth
. "$PSScriptRoot/common/setup-local-auth.ps1"

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# Load common frontend helpers (supports REST and CLI modes)
. "$PSScriptRoot/common/frontend-helpers.ps1"
$feCtx = New-FrontendContext -frontendEndpoint $frontendEndpoint -ApiMode $ApiMode

# Step 1: Get run history.
Write-Host "=== Query Run History ===" -ForegroundColor Cyan
$runHistory = $null
try {
    $runHistory = Get-FrontendQueryRunHistory -Context $feCtx `
        -CollaborationId $collaborationId `
        -DocumentId $queryName `
        -TokenFile $TokenFile
}
catch {
    $errMsg = $_.Exception.Message
    if ($errMsg -match "NotFound|No run history") {
        Write-Host "No run history found (runs may still be in progress)." -ForegroundColor Yellow
    } else {
        Write-Host "Error fetching run history: $errMsg" -ForegroundColor Red
    }
}
if ($runHistory) {
    $runHistory | ConvertTo-Json -Depth 10
}

# Step 2: Get audit events.
Write-Host "`n=== Audit Events ===" -ForegroundColor Cyan
$auditEvents = Get-FrontendAuditEvents -Context $feCtx `
    -CollaborationId $collaborationId `
    -TokenFile $TokenFile
if ($auditEvents) {
    $auditEvents | ConvertTo-Json -Depth 10
} else {
    Write-Host "No audit events found." -ForegroundColor Yellow
}

# Step 3: Download CPK-encrypted output (optional).
if ($DownloadCpkOutput) {
    Write-Host "`n=== Downloading CPK-Encrypted Output ===" -ForegroundColor Cyan

    if (-not $OutputDekFile) {
        Write-Host "ERROR: -OutputDekFile is required when -DownloadCpkOutput is set." -ForegroundColor Red
        exit 1
    }
    if (-not $OutputStorageAccount) {
        Write-Host "ERROR: -OutputStorageAccount is required when -DownloadCpkOutput is set." -ForegroundColor Red
        exit 1
    }
    if (-not (Test-Path $OutputDekFile)) {
        Write-Host "ERROR: Output DEK file '$OutputDekFile' not found." -ForegroundColor Red
        exit 1
    }

    $OutputLocalDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputLocalDir)
    New-Item -ItemType Directory -Path $OutputLocalDir -Force | Out-Null

    # Read DEK and compute CPK env vars.
    $dekBytes = [System.IO.File]::ReadAllBytes($OutputDekFile)
    $dekBase64 = [Convert]::ToBase64String($dekBytes)
    $dekSha256 = [Convert]::ToBase64String(
        [System.Security.Cryptography.SHA256]::HashData($dekBytes)
    )
    $tenantId = (az account show --query tenantId -o tsv)

    $containerUrl = "https://${OutputStorageAccount}.blob.core.windows.net/${OutputContainer}/"

    Write-Host "  Source: $containerUrl" -ForegroundColor Yellow
    Write-Host "  Destination: $OutputLocalDir" -ForegroundColor Yellow

    $env:CPK_ENCRYPTION_KEY = $dekBase64
    $env:CPK_ENCRYPTION_KEY_SHA256 = $dekSha256
    $env:AZCOPY_AUTO_LOGIN_TYPE = "AZCLI"
    $env:AZCOPY_TENANT_ID = $tenantId

    try {
        $PSNativeCommandUseErrorActionPreference = $false
        # Use --include-pattern to skip HNS directory marker blobs that cause
        # "is a directory" errors when the storage account has hierarchical namespace.
        azcopy copy "$containerUrl*" $OutputLocalDir --recursive --cpk-by-value --include-pattern "*.csv;*.crc;*_SUCCESS*" 2>&1 | ForEach-Object {
            Write-Host "  $_" -ForegroundColor DarkGray
        }
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: azcopy download failed with exit code $LASTEXITCODE" -ForegroundColor Red
        } else {
            Write-Host "Output downloaded to: $OutputLocalDir" -ForegroundColor Green

            # Display CSV output files.
            $csvFiles = Get-ChildItem -Path $OutputLocalDir -Filter "*.csv" -Recurse
            foreach ($csvFile in $csvFiles) {
                Write-Host "`n--- $($csvFile.Name) ---" -ForegroundColor Cyan
                Get-Content $csvFile.FullName | Select-Object -First 20
                $totalLines = (Get-Content $csvFile.FullName | Measure-Object).Count
                if ($totalLines -gt 20) {
                    Write-Host "... ($totalLines total lines, showing first 20)" -ForegroundColor Yellow
                }
            }
        }
        $PSNativeCommandUseErrorActionPreference = $true
    }
    finally {
        Remove-Item env:CPK_ENCRYPTION_KEY -ErrorAction SilentlyContinue
        Remove-Item env:CPK_ENCRYPTION_KEY_SHA256 -ErrorAction SilentlyContinue
        Remove-Item env:AZCOPY_AUTO_LOGIN_TYPE -ErrorAction SilentlyContinue
        Remove-Item env:AZCOPY_TENANT_ID -ErrorAction SilentlyContinue
    }
}

Write-Host "`nResults viewing complete." -ForegroundColor Green
