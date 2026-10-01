<#
.SYNOPSIS
    Downloads query output from Azure Storage (SSE or CPK).

.DESCRIPTION
    Auto-detects encryption mode from the datastore metadata.
    - SSE: Downloads via az storage blob download
    - CPK: Downloads via azcopy --cpk-by-value using the DEK

    If -JobId is provided, downloads all CSV partitions under that exact run's
    Analytics/<date>/<run-id>/ path. Otherwise downloads all CSV outputs.
    Preserves blob paths locally to avoid collisions between runs/partitions.

.PARAMETER resourceGroup
    Azure resource group (for loading generated names).

.PARAMETER persona
    Collaborator persona (default: woodgrove — typically the output owner).

.PARAMETER datasetSuffix
    Dataset suffix (e.g., "-cpk-v1", "-v1").

.PARAMETER JobId
    Optional job ID from query run (e.g., "cl-spark-<uuid>").
    Filters output to this specific run in both SSE and CPK modes.
    Also accepts the run UUID without the cl-spark- prefix.

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
$versionedMetadataFile = Join-Path $outDir "datastores" "$persona-datastore-metadata$datasetSuffix.json"
$datastoreMetadataFile = if (Test-Path $versionedMetadataFile) {
    $versionedMetadataFile
} else {
    Join-Path $outDir "datastores" "$persona-datastore-metadata.json"
}
if (-not (Test-Path $datastoreMetadataFile)) {
    Write-Host "ERROR: '$datastoreMetadataFile' not found." -ForegroundColor Red
    exit 1
}
$datastoreMeta = Get-Content $datastoreMetadataFile -Raw | ConvertFrom-Json
$outputMeta = $datastoreMeta.output
if (-not $outputMeta.containerName -or $outputMeta.encryptionMode -notin @("SSE", "CPK")) {
    throw "Output metadata must specify containerName and encryptionMode (SSE or CPK): $datastoreMetadataFile"
}
$isCpk = ($outputMeta.encryptionMode -eq "CPK")
$outputContainer = $outputMeta.containerName
$storageUrl = if ($outputMeta.storeUrl) {
    $outputMeta.storeUrl.TrimEnd('/')
} else {
    "https://$STORAGE_ACCOUNT_NAME.blob.core.windows.net"
}
$storageUri = $null
if (-not [Uri]::TryCreate($storageUrl, [UriKind]::Absolute, [ref]$storageUri) -or
    $storageUri.Scheme -ne "https" -or $storageUri.Query -or $storageUri.Fragment -or $storageUri.UserInfo) {
    throw "Output storeUrl must be an absolute HTTPS storage endpoint without credentials, query, or fragment."
}
$storageAccountName = $storageUri.Host.Split('.')[0]
$runId = if ($JobId) { $JobId -creplace '^cl-spark-', '' } else { $null }
if ($JobId -and ([string]::IsNullOrWhiteSpace($runId) -or $runId -match '[/\\]')) {
    throw "JobId must be a run ID or cl-spark-<run-id>, not a path."
}

Write-Host "=== Downloading query output ===" -ForegroundColor Cyan
Write-Host "  Storage account: $storageAccountName" -ForegroundColor Yellow
Write-Host "  Container: $outputContainer" -ForegroundColor Yellow
Write-Host "  Encryption: $(if ($isCpk) { 'CPK' } else { 'SSE' })" -ForegroundColor Yellow
if ($JobId) { Write-Host "  Job ID filter: $JobId" -ForegroundColor Yellow }

$blobs = az storage blob list --account-name $storageAccountName `
    --container-name $outputContainer `
    --prefix "Analytics/" --auth-mode login -o json | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw "Failed to list output blobs in '$outputContainer'." }

$csvBlobs = @($blobs | Where-Object {
    $segments = $_.name.Split('/')
    $_.name.EndsWith('.csv', [StringComparison]::OrdinalIgnoreCase) -and
        $segments.Count -ge 4 -and $segments[0] -ceq "Analytics" -and
        (-not $runId -or $segments[2] -ceq $runId)
})
if ($csvBlobs.Count -eq 0) {
    Write-Host "No CSV output blobs found$(if ($JobId) { " for job '$JobId'" })." -ForegroundColor Yellow
    return
}

$downloads = @($csvBlobs | ForEach-Object {
    $localFile = [IO.Path]::GetFullPath((Join-Path $OutputDir $_.name))
    $outputPrefix = $OutputDir.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar) +
        [IO.Path]::DirectorySeparatorChar
    if (-not $localFile.StartsWith($outputPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Output blob path escapes OutputDir: $($_.name)"
    }
    [pscustomobject]@{ BlobName = $_.name; LocalFile = $localFile }
})

if ($isCpk) {
    $dekFile = $outputMeta._local.dekFile
    if (-not $dekFile -or -not (Test-Path -LiteralPath $dekFile -PathType Leaf)) {
        throw "Output DEK file from metadata was not found: '$dekFile'."
    }
    $dekBytes = Get-Content -LiteralPath $dekFile -AsByteStream -Raw
    if ($dekBytes.Length -ne 32) { throw "Output DEK must contain exactly 32 bytes: $dekFile" }

    $environment = @{
        CPK_ENCRYPTION_KEY = [Convert]::ToBase64String($dekBytes)
        CPK_ENCRYPTION_KEY_SHA256 = [Convert]::ToBase64String(
            [System.Security.Cryptography.SHA256]::HashData($dekBytes))
        AZCOPY_AUTO_LOGIN_TYPE = "AZCLI"
        AZCOPY_TENANT_ID = (az account show --query tenantId -o tsv)
    }
    if ($LASTEXITCODE -ne 0 -or -not $environment.AZCOPY_TENANT_ID) {
        throw "Could not determine the Azure CLI tenant for AzCopy."
    }
    $previousEnvironment = @{}
    try {
        foreach ($name in $environment.Keys) {
            $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
            [Environment]::SetEnvironmentVariable($name, $environment[$name], 'Process')
        }

        foreach ($download in $downloads) {
            New-Item -ItemType Directory -Path (Split-Path $download.LocalFile -Parent) -Force | Out-Null
            $blobPath = ($download.BlobName.Split('/') | ForEach-Object { [Uri]::EscapeDataString($_) }) -join '/'
            $srcUrl = "$storageUrl/$([Uri]::EscapeDataString($outputContainer))/$blobPath"
            Write-Host "  Downloading: $($download.BlobName)" -ForegroundColor Cyan
            azcopy copy $srcUrl $download.LocalFile --cpk-by-value --overwrite=true 2>&1 | ForEach-Object {
                Write-Host "  $_" -ForegroundColor Gray
            }
            if ($LASTEXITCODE -ne 0) { throw "AzCopy failed to download '$($download.BlobName)' (exit $LASTEXITCODE)." }
        }
    } finally {
        foreach ($name in $previousEnvironment.Keys) {
            if ($null -eq $previousEnvironment[$name]) {
                Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
            } else {
                [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], 'Process')
            }
        }
    }
} else {
    foreach ($download in $downloads) {
        New-Item -ItemType Directory -Path (Split-Path $download.LocalFile -Parent) -Force | Out-Null
        Write-Host "  Downloading: $($download.BlobName)" -ForegroundColor Cyan
        az storage blob download --account-name $storageAccountName `
            --container-name $outputContainer `
            --name $download.BlobName `
            --file $download.LocalFile `
            --auth-mode login --output none
        if ($LASTEXITCODE -ne 0) { throw "Failed to download '$($download.BlobName)'." }
    }
}

# Display results
$csvFiles = @($downloads | ForEach-Object { Get-Item -LiteralPath $_.LocalFile })
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
