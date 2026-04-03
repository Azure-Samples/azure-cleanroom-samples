<#
.SYNOPSIS
    Downloads query output from Azure Storage (SSE or CPK).

.DESCRIPTION
    Auto-detects encryption mode from the datastore metadata.
    - SSE: Downloads via az storage blob download
    - CPK: Downloads via azcopy --cpk-by-value using the DEK

    If -JobId is provided, filters to blobs matching that run.
    Otherwise downloads the latest CSV blob.

.PARAMETER resourceGroup
    Azure resource group (for loading generated names).

.PARAMETER persona
    Collaborator persona (default: woodgrove — typically the output owner).

.PARAMETER datasetSuffix
    Dataset suffix (e.g., "-cpk-v1", "-v1").

.PARAMETER JobId
    Optional job ID from query run (e.g., "cl-spark-<uuid>").
    Filters output to this specific run.

.PARAMETER OutputDir
    Local directory for downloaded output (default: ./generated/output).

.PARAMETER outDir
    Generated metadata directory (default: ./generated).
#>
param(
    [Parameter(Mandatory)]
    [string]$resourceGroup,

    [string]$persona = "woodgrove",

    [Parameter(Mandatory)]
    [string]$datasetSuffix,

    [string]$JobId,

    [string]$OutputDir,

    [string]$outDir = "./generated"
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# Resolve paths
$outDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($outDir)
if (-not $OutputDir) {
    $OutputDir = Join-Path $outDir "output"
}
$OutputDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)

# Load resource names
$namesFile = Join-Path $outDir $resourceGroup "names.generated.ps1"
if (-not (Test-Path $namesFile)) {
    Write-Host "ERROR: '$namesFile' not found. Run 04-prepare-resources.ps1 first." -ForegroundColor Red
    exit 1
}
. $namesFile

# Detect encryption mode
$datastoreMetadataFile = Join-Path $outDir "datastores" "$persona-datastore-metadata.json"
if (-not (Test-Path $datastoreMetadataFile)) {
    Write-Host "ERROR: '$datastoreMetadataFile' not found." -ForegroundColor Red
    exit 1
}
$datastoreMeta = Get-Content $datastoreMetadataFile -Raw | ConvertFrom-Json
$isCpk = ($datastoreMeta.input.encryptionMode -eq "CPK")

$outputContainer = "woodgrove-output$datasetSuffix"

Write-Host "=== Downloading query output ===" -ForegroundColor Cyan
Write-Host "  Storage account: $STORAGE_ACCOUNT_NAME" -ForegroundColor Yellow
Write-Host "  Container: $outputContainer" -ForegroundColor Yellow
Write-Host "  Encryption: $(if ($isCpk) { 'CPK' } else { 'SSE' })" -ForegroundColor Yellow
if ($JobId) { Write-Host "  Job ID filter: $JobId" -ForegroundColor Yellow }

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

if ($isCpk) {
    # -- CPK: Download via azcopy --cpk-by-value ----------------------------------
    $dekFile = Join-Path $outDir "datastores" "keys" "woodgrove-output-csv$datasetSuffix-dek.bin"
    if (-not (Test-Path $dekFile)) {
        Write-Host "ERROR: DEK file '$dekFile' not found." -ForegroundColor Red
        exit 1
    }

    $dekBytes = [System.IO.File]::ReadAllBytes($dekFile)
    $env:CPK_ENCRYPTION_KEY = [Convert]::ToBase64String($dekBytes)
    $env:CPK_ENCRYPTION_KEY_SHA256 = [Convert]::ToBase64String(
        [System.Security.Cryptography.SHA256]::HashData($dekBytes))
    $env:AZCOPY_AUTO_LOGIN_TYPE = "AZCLI"
    $env:AZCOPY_TENANT_ID = (az account show --query tenantId -o tsv)

    $srcUrl = "https://$STORAGE_ACCOUNT_NAME.blob.core.windows.net/$outputContainer/Analytics/*"
    Write-Host "  Downloading via azcopy --cpk-by-value..." -ForegroundColor Cyan

    azcopy copy $srcUrl $OutputDir --cpk-by-value --recursive --include-pattern "*.csv" 2>&1 | ForEach-Object {
        Write-Host "  $_" -ForegroundColor Gray
    }

    # Cleanup env vars
    Remove-Item env:CPK_ENCRYPTION_KEY -ErrorAction SilentlyContinue
    Remove-Item env:CPK_ENCRYPTION_KEY_SHA256 -ErrorAction SilentlyContinue
    Remove-Item env:AZCOPY_AUTO_LOGIN_TYPE -ErrorAction SilentlyContinue
    Remove-Item env:AZCOPY_TENANT_ID -ErrorAction SilentlyContinue
} else {
    # -- SSE: Download via az storage blob download --------------------------------
    $blobs = az storage blob list --account-name $STORAGE_ACCOUNT_NAME `
        --container-name $outputContainer `
        --prefix "Analytics/" --auth-mode login -o json | ConvertFrom-Json

    # Filter to CSV blobs (exclude .crc checksum files)
    $csvBlobs = $blobs | Where-Object {
        $_.name -match '\.csv$' -and $_.name -notmatch '\.crc$'
    }

    # Optionally filter by job ID
    if ($JobId) {
        $runUuid = $JobId -replace '^cl-spark-', ''
        $csvBlobs = $csvBlobs | Where-Object { $_.name -match $runUuid }
    }

    if (-not $csvBlobs -or $csvBlobs.Count -eq 0) {
        Write-Host "No CSV output blobs found." -ForegroundColor Yellow
        exit 0
    }

    # Download each matching blob
    foreach ($blob in $csvBlobs) {
        $localFile = Join-Path $OutputDir (Split-Path $blob.name -Leaf)
        Write-Host "  Downloading: $($blob.name)" -ForegroundColor Cyan
        az storage blob download --account-name $STORAGE_ACCOUNT_NAME `
            --container-name $outputContainer `
            --name $blob.name `
            --file $localFile `
            --auth-mode login --output none
    }
}

# Display results
$csvFiles = Get-ChildItem $OutputDir -Recurse -Filter *.csv
if ($csvFiles.Count -gt 0) {
    Write-Host "`n=== Output ($($csvFiles.Count) CSV file(s)) ===" -ForegroundColor Green
    foreach ($f in $csvFiles) {
        Write-Host "--- $($f.Name) ---" -ForegroundColor Cyan
        Get-Content $f.FullName | Select-Object -First 20
        $totalLines = (Get-Content $f.FullName).Count
        if ($totalLines -gt 20) {
            Write-Host "  ... ($totalLines total rows)" -ForegroundColor Gray
        }
    }
} else {
    Write-Host "No CSV files downloaded." -ForegroundColor Yellow
}
