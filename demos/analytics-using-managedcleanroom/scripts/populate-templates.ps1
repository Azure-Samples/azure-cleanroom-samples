<#
.SYNOPSIS
    Generates the dataset publish JSON files by combining user-editable config
    (schema, allowedFields) with infrastructure metadata from Steps 4-6.

.DESCRIPTION
    Reads templates/dataset-config.json for schema and allowedFields, then merges
    with datastore metadata (Step 5) and identity metadata (Step 6) to produce
    ready-to-publish JSON files under generated/datasets/<variant>/.

.PARAMETER variant
    "sse" or "cpk" — determines encryption mode.

.PARAMETER persona
    The collaborator persona (northwind or woodgrove).

.PARAMETER resourceGroup
    The Azure resource group name (used to locate identity metadata).

.PARAMETER outDir
    Output directory for generated metadata (default: ./generated).
#>
param(
    [Parameter(Mandatory)]
    [ValidateSet("sse", "cpk")]
    [string]$variant,

    [Parameter(Mandatory)]
    [ValidateSet("northwind", "woodgrove")]
    [string]$persona,

    [Parameter(Mandatory)]
    [string]$resourceGroup,

    [string]$outDir = "./generated"
)

$ErrorActionPreference = 'Stop'

# -- Load dataset config (user-editable schema + allowedFields) ------------------
$configFile = Join-Path (Split-Path $PSScriptRoot) "templates" "dataset-config.json"
if (-not (Test-Path $configFile)) {
    Write-Host "ERROR: '$configFile' not found." -ForegroundColor Red
    exit 1
}
$config = Get-Content $configFile -Raw | ConvertFrom-Json

# -- Load metadata ---------------------------------------------------------------
$datastoreFile = Join-Path $outDir "datastores" "$persona-datastore-metadata.json"
if (-not (Test-Path $datastoreFile)) {
    Write-Host "ERROR: '$datastoreFile' not found. Run Step 5 first." -ForegroundColor Red
    exit 1
}
$datastoreMeta = Get-Content $datastoreFile -Raw | ConvertFrom-Json

$identityFile = Join-Path $outDir $resourceGroup "identity-metadata.json"
if (-not (Test-Path $identityFile)) {
    Write-Host "ERROR: '$identityFile' not found. Run Step 6 first." -ForegroundColor Red
    exit 1
}
$identityMeta = Get-Content $identityFile -Raw | ConvertFrom-Json

# -- Helpers ---------------------------------------------------------------------
function New-IdentityBlock {
    param([PSCustomObject]$Id)
    return [ordered]@{
        name     = $Id.identityName
        clientId = $Id.clientId
        tenantId = $Id.tenantId
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

function New-DatasetJson {
    param(
        [PSCustomObject]$Meta,
        [PSCustomObject]$Id,
        [string]$AccessMode,
        [PSCustomObject]$SchemaConfig,
        [string[]]$AllowedFields,
        [string]$Variant,
        [string]$Persona
    )

    $isRead  = ($AccessMode -eq "read")
    $apType  = if ($isRead) { "Volume_ReadOnly" } else { "Volume_ReadWrite" }
    $proxyType = if ($isRead) {
        "SecureVolume__ReadOnly__Azure__BlobStorage"
    } else {
        "SecureVolume__ReadWrite__Azure__BlobStorage"
    }

    $protection = [ordered]@{
        proxyType     = $proxyType
        proxyMode     = "Secure"
        configuration = ""
    }

    if ($Variant -eq "cpk" -and $Meta.encryption) {
        $enc = $Meta.encryption
        $protection.configuration = "{'KeyType': 'KEK', 'EncryptionMode': 'CPK'}"
        $protection.encryptionSecrets = [ordered]@{
            dek = [ordered]@{
                name   = $enc.dekSecretName
                secret = [ordered]@{
                    secretType      = "Key"
                    backingResource = [ordered]@{
                        id       = "$Persona-dek-store"
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
                        id       = "$Persona-kek-store"
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
        $protection.encryptionSecretAccessIdentity = (New-IdentityBlock -Id $Id)
    }

    $body = [ordered]@{
        data = [ordered]@{
            name = $Meta.name
            datasetSchema = [ordered]@{
                format = $SchemaConfig.format
                fields = @($SchemaConfig.fields | ForEach-Object {
                    [ordered]@{ fieldName = $_.fieldName; fieldType = $_.fieldType }
                })
            }
            datasetAccessPolicy = [ordered]@{
                accessMode    = $AccessMode
                allowedFields = @($AllowedFields)
            }
            datasetAccessPoint = [ordered]@{
                name = $Meta.name
                type = $apType
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
                identity   = (New-IdentityBlock -Id $Id)
                protection = $protection
            }
        }
    }

    return ($body | ConvertTo-Json -Depth 20)
}

# -- Generate dataset JSON files ---------------------------------------------------
$outputDir = Join-Path $outDir "datasets" $variant
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

# Resolve config entry for this persona
$personaConfig = $config.inputDatasets.$persona
if (-not $personaConfig) {
    Write-Host "ERROR: No entry for '$persona' in dataset-config.json inputDatasets." -ForegroundColor Red
    exit 1
}

# Input dataset
$inputMeta = $datastoreMeta.input
$inputJson = New-DatasetJson `
    -Meta $inputMeta `
    -Id $identityMeta `
    -AccessMode "read" `
    -SchemaConfig $personaConfig.schema `
    -AllowedFields @($personaConfig.allowedFields) `
    -Variant $variant `
    -Persona $persona

$inputFile = Join-Path $outputDir "$persona-input-dataset.json"
$inputJson | Out-File -FilePath $inputFile -Encoding utf8
Write-Host "Generated: $inputFile" -ForegroundColor Green

# Output dataset (woodgrove only)
if ($persona -eq "woodgrove" -and $datastoreMeta.output) {
    $outConfig = $config.outputDataset

    $outputJson = New-DatasetJson `
        -Meta $datastoreMeta.output `
        -Id $identityMeta `
        -AccessMode "write" `
        -SchemaConfig $outConfig.schema `
        -AllowedFields @($outConfig.allowedFields) `
        -Variant $variant `
        -Persona $persona

    $outputFile = Join-Path $outputDir "woodgrove-output-dataset.json"
    $outputJson | Out-File -FilePath $outputFile -Encoding utf8
    Write-Host "Generated: $outputFile" -ForegroundColor Green
}

Write-Host "`nDataset files generated for '$persona' ($variant) in $outputDir. Ready for Step 8." -ForegroundColor Cyan
