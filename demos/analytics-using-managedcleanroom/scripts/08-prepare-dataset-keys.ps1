<#
.SYNOPSIS
    Prepares CPK encryption keys for datasets (KEK creation, DEK wrapping).

.DESCRIPTION
    CPK-only utility script. For each dataset in the datastore metadata:
    1. Fetches the SKR release policy from the frontend (for the attestation hash).
    2. Generates a local RSA-2048 KEK.
    3. Imports the KEK to Key Vault with the SKR release policy.
    4. Wraps the DEK with the KEK (RSA-OAEP-SHA256, client-side).
    5. Stores the wrapped DEK as a Key Vault secret.

    This script uses HTTPS REST calls to fetch SKR policies; the managedcleanroom
    CLI extension is not required and its configuration is not changed.

    Prerequisites:
    - 04-prepare-resources.ps1 must have been run.
    - 05-prepare-data.ps1 must have been run with -variant cpk (generates DEK files).

.PARAMETER collaborationId
    The collaboration frontend UUID.

.PARAMETER resourceGroup
    Azure resource group.

.PARAMETER persona
    The collaborator persona (northwind or woodgrove).

.PARAMETER frontendEndpoint
    Frontend service URL (needed for SKR policy fetch).

.PARAMETER maaUrl
    MAA URL for the SKR release policy authority.

.PARAMETER outDir
    Output directory.

.PARAMETER TokenFile
    Token file for frontend authentication. If omitted, uses CLEANROOM_FRONTEND_TOKEN,
    then MANAGEDCLEANROOM_ACCESS_TOKEN, then the current Azure CLI ARM access token.
#>
param(
    [Parameter(Mandatory)]
    [string]$collaborationId,

    [Parameter(Mandatory)]
    [string]$resourceGroup,

    [Parameter(Mandatory)]
    [ValidateSet("northwind", "woodgrove", IgnoreCase = $false)]
    [string]$persona,

    [Parameter(Mandatory)]
    [string]$frontendEndpoint,

    [string]$maaUrl = "https://sharedeus.eus.attest.azure.net",

    [string]$outDir = "./generated",

    [string]$TokenFile
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# Resolve outDir to absolute path
$outDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($outDir)

function Invoke-AzSafe {
    param([string[]]$Arguments)
    $PSNativeCommandUseErrorActionPreference = $false
    $result = & az @Arguments 2>$null
    if ($LASTEXITCODE -eq 0) { return $result }
    return $null
}

# -- Load prerequisites -----------------------------------------------------------
$namesFile = Join-Path $outDir $resourceGroup "names.generated.ps1"
if (-not (Test-Path $namesFile)) {
    Write-Host "ERROR: '$namesFile' not found. Run 04-prepare-resources.ps1 first." -ForegroundColor Red
    exit 1
}
. $namesFile

$datastoreMetadataFile = Join-Path $outDir "datastores" "$persona-datastore-metadata.json"
if (-not (Test-Path $datastoreMetadataFile)) {
    Write-Host "ERROR: '$datastoreMetadataFile' not found. Run 05-prepare-data.ps1 with -variant cpk first." -ForegroundColor Red
    exit 1
}
$datastoreMeta = Get-Content $datastoreMetadataFile -Raw | ConvertFrom-Json

if ($datastoreMeta.input.encryptionMode -ne "CPK") {
    Write-Host "Encryption mode is not CPK — nothing to do." -ForegroundColor Yellow
    exit 0
}

# -- Set up frontend auth (used internally for SKR policy fetch) -------------------
$feBase = $frontendEndpoint.TrimEnd('/')
if ($feBase.EndsWith('/collaborations')) {
    $feBase = $feBase.Substring(0, $feBase.Length - '/collaborations'.Length)
}

function Get-SkrPolicy {
    param([string]$DatasetName)

    $endpointUri = $null
    if (-not [Uri]::TryCreate($feBase, [UriKind]::Absolute, [ref]$endpointUri) -or
        $endpointUri.Scheme -ne "https" -or $endpointUri.Query -or $endpointUri.Fragment -or
        $endpointUri.UserInfo) {
        throw "frontendEndpoint must be an absolute HTTPS URL without credentials, query, or fragment."
    }

    if ($TokenFile) {
        if (-not (Test-Path -LiteralPath $TokenFile -PathType Leaf)) {
            throw "Frontend token file not found: $TokenFile"
        }
        $token = ([string](Get-Content -LiteralPath $TokenFile -Raw)).Trim()
    } elseif ($env:CLEANROOM_FRONTEND_TOKEN) {
        $token = $env:CLEANROOM_FRONTEND_TOKEN.Trim()
    } elseif ($env:MANAGEDCLEANROOM_ACCESS_TOKEN) {
        $token = $env:MANAGEDCLEANROOM_ACCESS_TOKEN.Trim()
    } else {
        try {
            $token = az account get-access-token --resource "https://management.azure.com/" `
                --query accessToken -o tsv
            if ($LASTEXITCODE -ne 0) { throw "Azure CLI exited with code $LASTEXITCODE." }
        } catch {
            throw "Could not acquire a frontend token. Supply -TokenFile or CLEANROOM_FRONTEND_TOKEN. $($_.Exception.Message)"
        }
        $token = ($token -join "`n").Trim()
    }
    if ([string]::IsNullOrWhiteSpace($token)) {
        throw "Frontend token is empty. Supply a valid -TokenFile or CLEANROOM_FRONTEND_TOKEN."
    }

    $headers = @{ Authorization = "Bearer $token"; "Content-Type" = "application/json" }
    $encodedCollaborationId = [Uri]::EscapeDataString($collaborationId)
    $encodedDatasetName = [Uri]::EscapeDataString($DatasetName)
    $url = "$feBase/collaborations/$encodedCollaborationId/analytics/datasets/$encodedDatasetName/skrpolicy?api-version=2026-03-01-preview"
    try {
        $policy = Invoke-RestMethod -Uri $url -Headers $headers -Method Get -ErrorAction Stop
    } catch {
        throw "Failed to fetch SKR policy for dataset '$DatasetName': $($_.Exception.Message)"
    }
    if (-not $policy.version -or @($policy.anyOf).Count -eq 0 -or -not $policy.anyOf[0]) {
        throw "Frontend returned an invalid SKR policy for dataset '$DatasetName': expected version and nonempty anyOf."
    }

    return $policy
}

# -- Process each dataset ----------------------------------------------------------
function New-DatasetKekAndWrappedDek {
    param(
        [PSCustomObject]$DatasetMeta,
        [string]$KeyVaultName,
        [string]$OutputDir
    )

    $datasetName = $DatasetMeta.name
    $kekName = $DatasetMeta.encryption.kekName
    $dekFile = $DatasetMeta._local.dekFile
    $wrappedDekSecretName = $DatasetMeta.encryption.dekSecretName

    Write-Host "`n  --- Dataset: $datasetName ---" -ForegroundColor White
    Write-Host "  KEK name: $kekName" -ForegroundColor Yellow

    if (-not (Test-Path $dekFile)) {
        Write-Host "  ERROR: DEK file '$dekFile' not found." -ForegroundColor Red
        exit 1
    }

    $dekBytes = [System.IO.File]::ReadAllBytes($dekFile)

    # Step 1: Fetch SKR release policy
    Write-Host "  Fetching SKR release policy..." -ForegroundColor Yellow
    $skrPolicy = Get-SkrPolicy -DatasetName $datasetName
    $skrPolicy.anyOf[0].authority = $maaUrl
    $skrPolicyJson = $skrPolicy | ConvertTo-Json -Depth 10 -Compress
    Write-Host "  SKR policy fetched. ccePolicyHash: $($skrPolicy.anyOf[0].allOf[0].equals)" -ForegroundColor Green

    # Step 2: Generate RSA-2048 KEK locally
    Write-Host "  Generating KEK '$kekName' (RSA-2048)..." -ForegroundColor Yellow
    $kekOutputDir = Join-Path $OutputDir "datastores" "keys"
    New-Item -ItemType Directory -Path $kekOutputDir -Force | Out-Null

    $rsa = [System.Security.Cryptography.RSA]::Create(2048)
    $kekPemFile = Join-Path $kekOutputDir "$kekName.pem"
    $rsa.ExportPkcs8PrivateKeyPem() | Set-Content -Path $kekPemFile -NoNewline -Encoding utf8

    $skrPolicyFile = Join-Path $kekOutputDir "$kekName-skr-policy.json"
    [System.IO.File]::WriteAllText($skrPolicyFile, $skrPolicyJson)

    # Step 3: Import KEK to Key Vault with SKR policy
    Write-Host "  Importing KEK to Key Vault..." -ForegroundColor Yellow
    $existingKey = Invoke-AzSafe @("keyvault", "key", "show", "--vault-name", $KeyVaultName, "--name", $kekName)
    if ($existingKey) {
        Invoke-AzSafe @("keyvault", "key", "delete", "--vault-name", $KeyVaultName, "--name", $kekName)
        Start-Sleep -Seconds 5
        Invoke-AzSafe @("keyvault", "key", "purge", "--vault-name", $KeyVaultName, "--name", $kekName)
        Start-Sleep -Seconds 5
    }

    az keyvault key import --vault-name $KeyVaultName --name $kekName `
        --pem-file $kekPemFile --protection hsm --exportable true `
        --ops wrapKey unwrapKey --policy "@$skrPolicyFile" --output none
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to import KEK '$kekName'."
    }
    Write-Host "  KEK imported." -ForegroundColor Green

    # Step 4: Wrap DEK with KEK (RSA-OAEP-SHA256, client-side)
    $wrappedDekBytes = $rsa.Encrypt($dekBytes, [System.Security.Cryptography.RSAEncryptionPadding]::OaepSHA256)
    $wrappedDekBase64 = [Convert]::ToBase64String($wrappedDekBytes)
    $rsa.Dispose()

    # Step 5: Store wrapped DEK as KV secret
    Write-Host "  Storing wrapped DEK as secret '$wrappedDekSecretName'..." -ForegroundColor Yellow
    az keyvault secret set --vault-name $KeyVaultName --name $wrappedDekSecretName `
        --value $wrappedDekBase64 --output none
    Write-Host "  Done." -ForegroundColor Green
}

Write-Host "=== Preparing CPK keys ===" -ForegroundColor Cyan

New-DatasetKekAndWrappedDek -DatasetMeta $datastoreMeta.input `
    -KeyVaultName $KEYVAULT_NAME -OutputDir $outDir

if ($persona -eq "woodgrove" -and $datastoreMeta.output) {
    New-DatasetKekAndWrappedDek -DatasetMeta $datastoreMeta.output `
        -KeyVaultName $KEYVAULT_NAME -OutputDir $outDir
}

Write-Host "`nCPK key preparation complete for '$persona'." -ForegroundColor Green
