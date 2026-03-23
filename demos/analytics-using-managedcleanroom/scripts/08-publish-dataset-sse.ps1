<#
.SYNOPSIS
    Publishes dataset metadata to the collaboration (SSE variant).

.DESCRIPTION
    Run by: Each collaborator (Northwind and Woodgrove).

    This script constructs the DatasetSpecification JSON directly from metadata
    saved by previous steps, then publishes via `az managedcleanroom frontend`.
    No `az cleanroom` CLI is used.

    The DatasetSpecification JSON structure was derived from the cleanroom extension
    source code (CleanRoomSpecification model, AccessPoint, PrivacyProxySettings)
    in cleanroom-5.0.0.

    Northwind publishes their input dataset.
    Woodgrove publishes both their input dataset and output dataset.

    Prerequisites:
    - 04-prepare-resources.ps1 must have been run.
    - 05-prepare-data-sse.ps1 must have been run (uploads data, saves metadata).
    - 06-setup-identity.ps1 must have been run (OIDC setup, saves identity metadata).

.NOTES
    NEEDS LIVE VALIDATION — The JSON body will be accepted by the frontend (it does
    no field-level validation), but the cleanroom runtime may fail if any of these
    derived values are wrong:

    1. tokenIssuer.url = "https://cgs/oidc" — This is a symbolic CGS reference used
       by cleanroom v5.0.0. The old sample scripts used the actual OIDC issuer URL
       (the static website URL from Step 6). If the cleanroom runtime expects the
       real URL, change this to $identityMeta.tokenIssuerUrl in New-DatasetBody.

    2. encryptionSecrets omitted for SSE — SSE means Azure handles encryption
       server-side, so logically no client-side secrets are needed. If the runtime
       always expects an encryptionSecrets block, this will need adjustment.

    3. store.id = datastore name (e.g., "northwind-input-csv") — Follows the v5.0.0
       config_add_datastore() convention. If CGS expects the ARM resource ID of the
       storage account here, use $Meta.storeId instead.

    Source references used to derive the JSON structure:
    - CleanRoomSpecification model: cleanroom_common/azure_cleanroom_core/models/model.py
    - config_add_datastore: cleanroom_common/azure_cleanroom_core/utilities/datastore_helpers.py:43
    - Frontend DatasetSpecification: src/workloads/frontend/Models/CGS/DatasetSpecification.cs
    - Frontend AccessPoint: src/workloads/frontend/Models/CGS/AccessPoint.cs

.PARAMETER collaborationId
    The collaboration ARM resource ID.

.PARAMETER resourceGroup
    Azure resource group containing the storage account.

.PARAMETER persona
    The collaborator persona (northwind or woodgrove).

.PARAMETER outDir
    Output directory for generated metadata (default: ./generated).
#>
param(
    [Parameter(Mandatory)]
    [string]$collaborationId,

    [Parameter(Mandatory)]
    [string]$resourceGroup,

    [Parameter(Mandatory)]
    [ValidateSet("northwind", "woodgrove", IgnoreCase = $false)]
    [string]$persona,

    [string]$outDir = "./generated"
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# Runs an az command, returning $null instead of throwing if it fails.
function Invoke-AzSafe {
    param([string[]]$Arguments)
    $PSNativeCommandUseErrorActionPreference = $false
    $result = & az @Arguments 2>$null
    if ($LASTEXITCODE -eq 0) { return $result }
    return $null
}

# Builds the DatasetSpecification JSON body for the frontend dataset publish API.
# Structure matches the CleanRoomSpecification AccessPoint model (cleanroom v5.0.0).
function New-DatasetBody {
    param(
        [PSCustomObject]$Meta,         # Datastore metadata (name, storeType, storeUrl, containerName, schema)
        [PSCustomObject]$Identity,     # Identity metadata (identityName, clientId, tenantId)
        [string]$AccessMode,           # "read" or "write"
        [string[]]$AllowedFields       # Fields allowed by the access policy
    )

    $isRead = ($AccessMode -eq "read")
    $accessPointType = if ($isRead) { "Volume_ReadOnly" } else { "Volume_ReadWrite" }
    $proxyType = if ($isRead) {
        "SecureVolume__ReadOnly__Azure__BlobStorage"
    } else {
        "SecureVolume__ReadWrite__Azure__BlobStorage"
    }

    $body = [ordered]@{
        data = [ordered]@{
            name = $Meta.name
            datasetSchema = [ordered]@{
                format = $Meta.schema.format
                fields = @($Meta.schema.fields | ForEach-Object {
                    [ordered]@{ fieldName = $_.fieldName; fieldType = $_.fieldType }
                })
            }
            datasetAccessPolicy = [ordered]@{
                accessMode    = $AccessMode
                allowedFields = @($AllowedFields)
            }
            datasetAccessPoint = [ordered]@{
                name = $Meta.name
                type = $accessPointType
                path = ""
                store = [ordered]@{
                    name     = $Meta.containerName
                    type     = $Meta.storeType
                    id       = $Meta.name
                    provider = [ordered]@{
                        protocol      = $Meta.storeType
                        url           = $Meta.storeUrl
                        configuration = ""
                    }
                }
                identity = [ordered]@{
                    name     = $Identity.identityName
                    clientId = $Identity.clientId
                    tenantId = $Identity.tenantId
                    tokenIssuer = [ordered]@{
                        issuer = [ordered]@{
                            protocol      = "Attested_OIDC"
                            url           = "https://cgs/oidc"
                            configuration = ""
                        }
                        issuerType = "AttestationBasedTokenIssuer"
                    }
                }
                protection = [ordered]@{
                    proxyType     = $proxyType
                    proxyMode     = "Secure"
                    configuration = ""
                }
            }
        }
    }

    return $body | ConvertTo-Json -Depth 15
}

# -- Load generated resource names ----------------------------------------------
$namesFile = Join-Path $outDir $resourceGroup "names.generated.ps1"
if (-not (Test-Path $namesFile)) {
    Write-Host "ERROR: '$namesFile' not found. Run 04-prepare-resources.ps1 first." -ForegroundColor Red
    exit 1
}
. $namesFile

# -- Load metadata from previous steps -----------------------------------------
$datastoreMetadataFile = Join-Path $outDir "datastores" "$persona-datastore-metadata.json"
if (-not (Test-Path $datastoreMetadataFile)) {
    Write-Host "ERROR: '$datastoreMetadataFile' not found. Run 05-prepare-data-sse.ps1 first." -ForegroundColor Red
    exit 1
}
$datastoreMeta = Get-Content $datastoreMetadataFile -Raw | ConvertFrom-Json

$identityMetadataFile = Join-Path $outDir $resourceGroup "identity-metadata.json"
if (-not (Test-Path $identityMetadataFile)) {
    Write-Host "ERROR: '$identityMetadataFile' not found. Run 06-setup-identity.ps1 first." -ForegroundColor Red
    exit 1
}
$identityMeta = Get-Content $identityMetadataFile -Raw | ConvertFrom-Json

$inputMeta = $datastoreMeta.input
$outputDir = Join-Path $outDir $resourceGroup
$tempDir = Join-Path $outputDir "temp"
if (-not (Test-Path $tempDir)) {
    New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
}

# -- Publish input dataset ------------------------------------------------------
Write-Host "=== Publishing input dataset '$($inputMeta.name)' ===" -ForegroundColor Cyan

$inputSchemaFile = Join-Path $tempDir "$persona-input-schema.json"
$inputMeta.schema | ConvertTo-Json -Depth 10 | Out-File -FilePath $inputSchemaFile -Encoding utf8

$existingInput = Invoke-AzSafe @("managedcleanroom", "frontend", "analytics", "dataset", "show", "--collaboration-id", $collaborationId, "--document-id", $inputMeta.name)
if ($existingInput) {
    Write-Host "Input dataset '$($inputMeta.name)' already published (skipped)." -ForegroundColor Yellow
} else {
    az managedcleanroom frontend analytics dataset publish `
        --collaboration-id $collaborationId `
        --document-id $inputMeta.name `
        --storage-account-url $inputMeta.storeUrl `
        --container-name $inputMeta.containerName `
        --storage-account-type $inputMeta.storeType `
        --encryption-mode "SSE" `
        --schema-file "@$inputSchemaFile" `
        --schema-format $inputMeta.schema.format `
        --access-mode "read" `
        --allowed-fields "date,author,mentions" `
        --identity-name $identityMeta.identityName `
        --identity-client-id $identityMeta.clientId `
        --identity-tenant-id $identityMeta.tenantId `
        --identity-issuer-url "https://cgs/oidc"
    Write-Host "Input dataset published." -ForegroundColor Green
}

az managedcleanroom frontend analytics dataset show `
    --collaboration-id $collaborationId `
    --document-id $inputMeta.name

# -- Publish output dataset (Woodgrove only) -----------------------------------
if ($persona -eq "woodgrove" -and $datastoreMeta.output) {
    Write-Host "`n=== Publishing output dataset '$($datastoreMeta.output.name)' ===" -ForegroundColor Cyan

    $outputSchemaFile = Join-Path $tempDir "woodgrove-output-schema.json"
    $datastoreMeta.output.schema | ConvertTo-Json -Depth 10 | Out-File -FilePath $outputSchemaFile -Encoding utf8

    $existingOutput = Invoke-AzSafe @("managedcleanroom", "frontend", "analytics", "dataset", "show", "--collaboration-id", $collaborationId, "--document-id", $datastoreMeta.output.name)
    if ($existingOutput) {
        Write-Host "Output dataset '$($datastoreMeta.output.name)' already published (skipped)." -ForegroundColor Yellow
    } else {
        az managedcleanroom frontend analytics dataset publish `
            --collaboration-id $collaborationId `
            --document-id $datastoreMeta.output.name `
            --storage-account-url $datastoreMeta.output.storeUrl `
            --container-name $datastoreMeta.output.containerName `
            --storage-account-type $datastoreMeta.output.storeType `
            --encryption-mode "SSE" `
            --schema-file "@$outputSchemaFile" `
            --schema-format $datastoreMeta.output.schema.format `
            --access-mode "write" `
            --allowed-fields "author,Number_Of_Mentions" `
            --identity-name $identityMeta.identityName `
            --identity-client-id $identityMeta.clientId `
            --identity-tenant-id $identityMeta.tenantId `
            --identity-issuer-url "https://cgs/oidc"
        Write-Host "Output dataset published." -ForegroundColor Green
    }

    az managedcleanroom frontend analytics dataset show `
        --collaboration-id $collaborationId `
        --document-id $datastoreMeta.output.name
}

Write-Host "`nDataset publishing complete for '$persona'." -ForegroundColor Green

# -- Enable execution consent on published datasets -----------------------------
Write-Host "`n=== Enabling execution consent on datasets ===" -ForegroundColor Cyan
az managedcleanroom frontend consent set `
    --collaboration-id $collaborationId `
    --document-id $inputMeta.name `
    --consent-action enable
Write-Host "Execution consent enabled for input dataset '$($inputMeta.name)'." -ForegroundColor Green

if ($persona -eq "woodgrove" -and $datastoreMeta.output) {
    az managedcleanroom frontend consent set `
        --collaboration-id $collaborationId `
        --document-id $datastoreMeta.output.name `
        --consent-action enable
    Write-Host "Execution consent enabled for output dataset '$($datastoreMeta.output.name)'." -ForegroundColor Green
}
