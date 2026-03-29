<#
.SYNOPSIS
    Publishes dataset metadata to the collaboration (CPK variant).

.DESCRIPTION
    Run by: Each collaborator (Northwind and Woodgrove).

    This script constructs the dataset specification JSON and publishes via the
    frontend helpers (REST or CLI mode). Replaces direct `az managedcleanroom
    frontend` CLI calls with Publish-FrontendDataset from frontend-helpers.ps1.

    Unlike SSE, CPK (Client-Provided Keys) requires encryption secret references
    (DEK/KEK) in the dataset specification so the cleanroom can decrypt data at
    runtime. These references point to the Key Vault secrets/keys created in
    step 05-prepare-data-cpk.ps1.

    Northwind publishes their input dataset.
    Woodgrove publishes both their input dataset and output dataset.

    Prerequisites:
    - 04-prepare-resources.ps1 must have been run (creates Key Vault with Premium SKU).
    - 05-prepare-data-cpk.ps1 must have been run (encrypts data, uploads, saves metadata).
    - 06-setup-identity.ps1 must have been run (OIDC setup, saves identity metadata).

.PARAMETER collaborationId
    The collaboration ARM resource ID.

.PARAMETER resourceGroup
    The Azure resource group containing provisioned resources.

.PARAMETER persona
    The collaborator persona (northwind or woodgrove).

.PARAMETER frontendEndpoint
    Frontend service URL.

.PARAMETER outDir
    Output directory for generated configuration files (default: ./generated).

.PARAMETER TokenFile
    Optional path to a pre-generated MSAL IdToken file (for persona switching).

.PARAMETER ApiMode
    API mode: "rest" (default) or "cli".
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

    [string]$outDir = "./generated",

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

# Load OIDC issuer URL from generated metadata
$issuerUrlFile = Join-Path $outDir $resourceGroup "issuer-url.txt"
if (-not (Test-Path $issuerUrlFile)) {
    Write-Host "ERROR: '$issuerUrlFile' not found. Run 06-setup-identity.ps1 first." -ForegroundColor Red
    exit 1
}
$oidcIssuerUrl = (Get-Content $issuerUrlFile -Raw).Trim()
Write-Host "Using OIDC issuer URL: $oidcIssuerUrl" -ForegroundColor Cyan

$inputMeta = $datastoreMeta.input

# -- Build dataset publish body -------------------------------------------------
# Body structure matches SSE pattern plus CPK-specific dek/kek fields.
function New-DatasetPublishBody {
    param(
        [PSCustomObject]$Meta,
        [PSCustomObject]$Identity,
        [string]$AccessMode,
        [string[]]$AllowedFields,
        [string]$EncryptionMode = "CPK"
    )

    $body = [ordered]@{
        name = $Meta.name
        datasetSchema = [ordered]@{
            format = $Meta.schema.format
            fields = @($Meta.schema.fields | ForEach-Object {
                [ordered]@{ fieldName = $_.fieldName; fieldType = $_.fieldType }
            })
        }
        datasetAccessPolicy = [ordered]@{
            accessMode = $AccessMode
        }
        store = [ordered]@{
            storageAccountUrl  = $Meta.storeUrl
            containerName      = $Meta.containerName
            storageAccountType = $Meta.storeType
            encryptionMode     = $EncryptionMode
        }
        identity = [ordered]@{
            name      = $Identity.identityName
            clientId  = $Identity.clientId
            tenantId  = $Identity.tenantId
            issuerUrl = $script:oidcIssuerUrl
        }
        # CPK-specific: encryption key references
        dek = [ordered]@{
            keyVaultUrl = $Meta.encryption.dekStoreUrl
            secretId    = $Meta.encryption.dekSecretName
        }
        kek = [ordered]@{
            keyVaultUrl = $Meta.encryption.kekStoreUrl
            secretId    = $Meta.encryption.kekName
            maaUrl      = ""
        }
    }

    if ($AllowedFields -and $AllowedFields.Count -gt 0) {
        $body.datasetAccessPolicy.allowedFields = @($AllowedFields)
    }

    return $body
}

# -- Publish input dataset ------------------------------------------------------
Write-Host "=== Publishing input dataset '$($inputMeta.name)' (CPK) ===" -ForegroundColor Cyan

$existingInput = Get-FrontendDataset -Context $feCtx -CollaborationId $collaborationId -DocumentId $inputMeta.name -TokenFile $TokenFile
if ($existingInput) {
    Write-Host "Input dataset '$($inputMeta.name)' already published (skipped)." -ForegroundColor Yellow
} else {
    $inputBody = New-DatasetPublishBody `
        -Meta $inputMeta `
        -Identity $identityMeta `
        -AccessMode "read" `
        -AllowedFields @("date", "author", "mentions")

    Publish-FrontendDataset -Context $feCtx `
        -CollaborationId $collaborationId `
        -DocumentId $inputMeta.name `
        -Body $inputBody `
        -TokenFile $TokenFile
    Write-Host "Input dataset '$($inputMeta.name)' published." -ForegroundColor Green
}

# Show the dataset
$datasetInfo = Get-FrontendDataset -Context $feCtx -CollaborationId $collaborationId -DocumentId $inputMeta.name -TokenFile $TokenFile
if ($datasetInfo) {
    $datasetInfo | ConvertTo-Json -Depth 10
}

# -- Publish output dataset (Woodgrove only) -----------------------------------
if ($persona -eq "woodgrove" -and $datastoreMeta.output) {
    Write-Host "`n=== Publishing output dataset '$($datastoreMeta.output.name)' (CPK) ===" -ForegroundColor Cyan

    $existingOutput = Get-FrontendDataset -Context $feCtx -CollaborationId $collaborationId -DocumentId $datastoreMeta.output.name -TokenFile $TokenFile
    if ($existingOutput) {
        Write-Host "Output dataset '$($datastoreMeta.output.name)' already published (skipped)." -ForegroundColor Yellow
    } else {
        $outputBody = New-DatasetPublishBody `
            -Meta $datastoreMeta.output `
            -Identity $identityMeta `
            -AccessMode "write" `
            -AllowedFields @("author", "Number_Of_Mentions")

        Publish-FrontendDataset -Context $feCtx `
            -CollaborationId $collaborationId `
            -DocumentId $datastoreMeta.output.name `
            -Body $outputBody `
            -TokenFile $TokenFile
        Write-Host "Output dataset '$($datastoreMeta.output.name)' published." -ForegroundColor Green
    }

    $outputInfo = Get-FrontendDataset -Context $feCtx -CollaborationId $collaborationId -DocumentId $datastoreMeta.output.name -TokenFile $TokenFile
    if ($outputInfo) {
        $outputInfo | ConvertTo-Json -Depth 10
    }
}

Write-Host "`nDataset publishing complete for '$persona'." -ForegroundColor Green

# -- Enable execution consent on published datasets -----------------------------
Write-Host "`n=== Enabling execution consent on datasets ===" -ForegroundColor Cyan
Set-FrontendConsent -Context $feCtx -CollaborationId $collaborationId -DocumentId $inputMeta.name -Action "enable" -TokenFile $TokenFile
Write-Host "Execution consent enabled for input dataset '$($inputMeta.name)'." -ForegroundColor Green

if ($persona -eq "woodgrove" -and $datastoreMeta.output) {
    Set-FrontendConsent -Context $feCtx -CollaborationId $collaborationId -DocumentId $datastoreMeta.output.name -Action "enable" -TokenFile $TokenFile
    Write-Host "Execution consent enabled for output dataset '$($datastoreMeta.output.name)'." -ForegroundColor Green
}
