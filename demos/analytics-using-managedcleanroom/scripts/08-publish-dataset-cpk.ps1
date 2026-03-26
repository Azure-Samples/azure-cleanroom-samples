<#
.SYNOPSIS
    Publishes dataset metadata to the collaboration (CPK variant).

.DESCRIPTION
    Run by: Each collaborator (Northwind and Woodgrove).

    This script constructs the DatasetSpecification JSON directly from metadata
    saved by previous steps, then publishes via `az managedcleanroom frontend`.
    No `az cleanroom` CLI is used.

    Unlike SSE, CPK (Client-Provided Keys) requires encryption secret references
    (DEK/KEK) in the dataset specification so the cleanroom can decrypt data at
    runtime. These references point to the Key Vault secrets/keys created in
    step 05-prepare-data-cpk.ps1.

    The DatasetSpecification JSON structure was derived from the cleanroom extension
    source code (CleanRoomSpecification model, AccessPoint, PrivacyProxySettings,
    EncryptionSecrets) in cleanroom-5.0.0.

    Northwind publishes their input dataset.
    Woodgrove publishes both their input dataset and output dataset.

    Prerequisites:
    - 04-prepare-resources.ps1 must have been run (creates Key Vault with Premium SKU).
    - 05-prepare-data-cpk.ps1 must have been run (encrypts data, uploads, saves metadata).
    - 06-setup-identity.ps1 must have been run (OIDC setup, saves identity metadata).

.NOTES
    NEEDS LIVE VALIDATION — The JSON body will be accepted by the frontend (it does
    no field-level validation), but the cleanroom runtime may fail if any of these
    derived values are wrong:

    1. tokenIssuer.url = "https://cgs/oidc" — This is a symbolic CGS reference used
       by cleanroom v5.0.0. The old sample scripts used the actual OIDC issuer URL
       (the static website URL from Step 6). If the cleanroom runtime expects the
       real URL, change this to $Identity.tokenIssuerUrl in New-IdentityObject.

    2. backingResource.id for DEK/KEK = logical names ("$persona-dek-store",
       "$persona-kek-store") — Follows the v5.0.0 convention where secretstore add
       created logical names mapped to Key Vault URLs. If CGS resolves these against
       a registry, they would fail since we didn't register via secretstore add.
       If this fails, try using the Key Vault URL directly as the id value.

    3. store.id = datastore name (e.g., "northwind-input-csv") — Follows the v5.0.0
       config_add_datastore() convention. If CGS expects the ARM resource ID of the
       storage account here, use $Meta.storeId instead.

    4. protection.configuration = "{'KeyType': 'KEK', 'EncryptionMode': 'CPK'}" —
       Follows the v5.0.0 Python string representation. If CGS parses this as JSON,
       the single-quoted Python dict syntax may fail; use proper JSON double quotes.

    Source references used to derive the JSON structure:
    - CleanRoomSpecification model: cleanroom_common/azure_cleanroom_core/models/model.py
    - config_add_datastore: cleanroom_common/azure_cleanroom_core/utilities/datastore_helpers.py:43
    - EncryptionSecrets: cleanroom_common/azure_cleanroom_core/models/model.py:190
    - Frontend DatasetSpecification: src/workloads/frontend/Models/CGS/DatasetSpecification.cs
    - Frontend PrivacyProxySettings: src/workloads/frontend/Models/CGS/PrivacyProxySettings.cs

.PARAMETER collaborationId
    The collaboration ARM resource ID.

.PARAMETER resourceGroup
    The Azure resource group containing provisioned resources.

.PARAMETER persona
    The collaborator persona (northwind or woodgrove).

.PARAMETER outDir
    Output directory for generated configuration files (default: ./generated).

.PARAMETER persona
    Persona (northwind or woodgrove) for naming/logging.
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

# Configure Private CleanRoom cloud for dogfood environment
Write-Host "Configuring Private CleanRoom cloud..." -ForegroundColor Cyan
$env:UsePrivateCleanRoomNamespace = "true"
$privateCloudName = "PrivateCleanroomAzureCloud"

# Register the private cloud if not already registered
$existingCloud = az cloud list --query "[?name=='$privateCloudName']" -o json 2>$null | ConvertFrom-Json
if (-not $existingCloud) {
    Write-Host "  Registering Private CleanRoom cloud..." -ForegroundColor Yellow
    az cloud register --name $privateCloudName --endpoint-resource-manager "https://eastus2euap.management.azure.com/" 2>&1 | Out-Null
    Write-Host "  Private CleanRoom cloud registered." -ForegroundColor Green
}

# Set and login with managed identity
az cloud set --name $privateCloudName 2>&1 | Out-Null
az login --identity --allow-no-subscriptions 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to login with managed identity"
    exit 1
}
Write-Host "Private CleanRoom cloud configuration complete." -ForegroundColor Green

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

# Builds the identity object used in AccessPoint and EncryptionSecretAccessIdentity.
function New-IdentityObject {
    param([PSCustomObject]$Identity)
    return [ordered]@{
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
}

# Builds the DatasetSpecification JSON body for the frontend dataset publish API.
# Structure matches the CleanRoomSpecification AccessPoint model (cleanroom v5.0.0).
# For CPK, includes EncryptionSecrets with DEK/KEK references.
function New-DatasetBody {
    param(
        [PSCustomObject]$Meta,         # Datastore metadata (name, storeType, storeUrl, containerName, schema, encryption)
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

    # Build protection settings with CPK encryption secrets.
    $enc = $Meta.encryption
    $protection = [ordered]@{
        proxyType     = $proxyType
        proxyMode     = "Secure"
        configuration = "{'KeyType': 'KEK', 'EncryptionMode': 'CPK'}"
        encryptionSecrets = [ordered]@{
            dek = [ordered]@{
                name   = $enc.dekSecretName
                secret = [ordered]@{
                    secretType      = "Key"
                    backingResource = [ordered]@{
                        id       = "$persona-dek-store"
                        name     = $enc.dekSecretName
                        type     = "AzureKeyVault"
                        provider = [ordered]@{
                            protocol = "AzureKeyVault_Secret"
                            url      = $enc.dekStoreUrl
                        }
                    }
                }
            }
            kek = [ordered]@{
                name   = $enc.kekName
                secret = [ordered]@{
                    secretType      = "Key"
                    backingResource = [ordered]@{
                        id       = "$persona-kek-store"
                        name     = $enc.kekName
                        type     = "AzureKeyVault"
                        provider = [ordered]@{
                            protocol      = "AzureKeyVault_SecureKey"
                            url           = $enc.kekStoreUrl
                            configuration = ""
                        }
                    }
                }
            }
        }
        encryptionSecretAccessIdentity = (New-IdentityObject -Identity $Identity)
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
                identity   = (New-IdentityObject -Identity $Identity)
                protection = $protection
            }
        }
    }

    return $body | ConvertTo-Json -Depth 20
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
    Write-Host "ERROR: '$datastoreMetadataFile' not found. Run 05-prepare-data-cpk.ps1 first." -ForegroundColor Red
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
Write-Host "=== Publishing input dataset '$($inputMeta.name)' (CPK) ===" -ForegroundColor Cyan

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
        --encryption-mode "CPK" `
        --schema-file "@$inputSchemaFile" `
        --schema-format $inputMeta.schema.format `
        --access-mode "read" `
        --allowed-fields "date,author,mentions" `
        --identity-name $identityMeta.identityName `
        --identity-client-id $identityMeta.clientId `
        --identity-tenant-id $identityMeta.tenantId `
        --identity-issuer-url "https://cgs/oidc" `
        --dek-keyvault-url $inputMeta.encryption.dekStoreUrl `
        --dek-secret-id $inputMeta.encryption.dekSecretName `
        --kek-keyvault-url $inputMeta.encryption.kekStoreUrl `
        --kek-secret-id $inputMeta.encryption.kekName
    Write-Host "Input dataset '$($inputMeta.name)' published." -ForegroundColor Green
}

az managedcleanroom frontend analytics dataset show `
    --collaboration-id $collaborationId `
    --document-id $inputMeta.name

# -- Publish output dataset (Woodgrove only) -----------------------------------
if ($persona -eq "woodgrove" -and $datastoreMeta.output) {
    Write-Host "`n=== Publishing output dataset '$($datastoreMeta.output.name)' (CPK) ===" -ForegroundColor Cyan

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
            --encryption-mode "CPK" `
            --schema-file "@$outputSchemaFile" `
            --schema-format $datastoreMeta.output.schema.format `
            --access-mode "write" `
            --allowed-fields "author,Number_Of_Mentions" `
            --identity-name $identityMeta.identityName `
            --identity-client-id $identityMeta.clientId `
            --identity-tenant-id $identityMeta.tenantId `
            --identity-issuer-url "https://cgs/oidc" `
            --dek-keyvault-url $datastoreMeta.output.encryption.dekStoreUrl `
            --dek-secret-id $datastoreMeta.output.encryption.dekSecretName `
            --kek-keyvault-url $datastoreMeta.output.encryption.kekStoreUrl `
            --kek-secret-id $datastoreMeta.output.encryption.kekName
        Write-Host "Output dataset '$($datastoreMeta.output.name)' published." -ForegroundColor Green
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
