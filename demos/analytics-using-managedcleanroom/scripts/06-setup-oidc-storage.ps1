<#
.SYNOPSIS
    Sets up OIDC storage infrastructure and saves identity metadata (utility only).

.DESCRIPTION
    Publishes OpenID configuration and JWKS to an OIDC storage account, or reuses
    the exact issuer URL already published by another collaborator in the same
    collaboration. Reuse does not access or modify the publisher's storage.
    Both modes save the current collaborator's managed identity metadata.

    Does NOT make any frontend service calls. The caller (README) is responsible for:
    1. Fetching JWKS from the frontend before publishing (not needed for reuse).
    2. Registering the issuer URL using the current collaborator's frontend token.

    Prerequisites:
    - 04-prepare-resources.ps1 must have been run.
    - For publication, a JWKS file fetched from the frontend must already exist.

.PARAMETER resourceGroup
    Azure resource group containing the managed identity.

.PARAMETER persona
    The collaborator persona (northwind or woodgrove).

.PARAMETER collaborationId
    The collaboration frontend UUID (used as blob path prefix for shared SA).

.PARAMETER JwksFile
    Path to the JWKS JSON file (previously fetched from the frontend).

.PARAMETER OidcStorageUrl
    Optional: the full static-website endpoint URL of a pre-existing
    whitelisted OIDC storage account (e.g. https://<account>.z22.web.core.windows.net).
    If omitted, creates a new SA in the resource group.

.PARAMETER ReuseIssuerUrl
    Exact full issuer URL shared by the publishing collaborator, including its
    path. Use this instead of JwksFile/OidcStorageUrl, including across tenants.
    The published URL must be permitted by each collaborator's tenant policy.

.PARAMETER outDir
    Output directory for generated metadata (default: ./generated).
#>
[CmdletBinding(DefaultParameterSetName = "Publish")]
param(
    [Parameter(Mandatory)]
    [string]$resourceGroup,

    [Parameter(Mandatory)]
    [ValidateSet("northwind", "woodgrove", IgnoreCase = $false)]
    [string]$persona,

    [Parameter(Mandatory)]
    [string]$collaborationId,

    [Parameter(Mandatory, ParameterSetName = "Publish")]
    [string]$JwksFile,

    [Parameter(ParameterSetName = "Publish")]
    [string]$OidcStorageUrl,

    [Parameter(Mandatory, ParameterSetName = "Reuse")]
    [ValidateNotNullOrEmpty()]
    [string]$ReuseIssuerUrl,

    [string]$outDir = "./generated"
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

# -- Validate prerequisites -------------------------------------------------------
$namesFile = Join-Path $outDir $resourceGroup "names.generated.ps1"
if (-not (Test-Path $namesFile)) {
    Write-Host "ERROR: '$namesFile' not found. Run 04-prepare-resources.ps1 first." -ForegroundColor Red
    exit 1
}
. $namesFile

if ($PSCmdlet.ParameterSetName -eq "Reuse") {
    $issuerUri = $null
    if ($ReuseIssuerUrl -cne $ReuseIssuerUrl.Trim() -or
        -not [Uri]::TryCreate($ReuseIssuerUrl, [UriKind]::Absolute, [ref]$issuerUri) -or
        $issuerUri.Scheme -ne "https" -or $issuerUri.UserInfo -or
        $issuerUri.Query -or $issuerUri.Fragment) {
        throw "ReuseIssuerUrl must be an exact HTTPS issuer URL without credentials, query, or fragment."
    }
}
elseif (-not (Test-Path -LiteralPath $JwksFile -PathType Leaf)) {
    Write-Host "ERROR: JWKS file '$JwksFile' not found. Fetch JWKS from frontend first." -ForegroundColor Red
    exit 1
}

$outputDir = Join-Path $outDir $resourceGroup
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

# -- Publish documents, or reuse the publisher's exact issuer URL ----------------
if ($PSCmdlet.ParameterSetName -eq "Reuse") {
    $issuerUrl = $ReuseIssuerUrl
    Write-Host "Reusing published issuer URL: $issuerUrl" -ForegroundColor Cyan
} else {
    $subscriptionId = az account show --query id -o tsv

    $newOidcAccountCreated = $false
    if ($OidcStorageUrl) {
        $staticWebUrl = $OidcStorageUrl.TrimEnd('/')
        # Extract the account name from the static-website hostname.
        $oidcStorageAccountName = ([Uri]$staticWebUrl).Host.Split('.')[0]
        $containerName = $collaborationId
        $issuerUrl = "$staticWebUrl/$containerName"
        Write-Host "Using pre-existing OIDC storage account '$oidcStorageAccountName'." -ForegroundColor Cyan
    } else {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes("$subscriptionId-$resourceGroup")
        $sha = [System.Security.Cryptography.SHA256]::Create()
        $hash = -join ($sha.ComputeHash($bytes) | ForEach-Object { $_.ToString("x2") })
        $oidcStorageAccountName = ("oidc" + $hash).Substring(0, 24).ToLower() -replace '[^a-z0-9]', ''
        $containerName = ("oidc-" + $hash).Substring(0, 24).ToLower() -replace '[^a-z0-9-]', ''

        $location = az group show --name $resourceGroup --query location -o tsv

        Write-Host "Creating OIDC storage account '$oidcStorageAccountName'..." -ForegroundColor Cyan
        az storage account create --name $oidcStorageAccountName --resource-group $resourceGroup `
            --location $location --sku Standard_LRS --min-tls-version TLS1_2 --output none

        az storage blob service-properties update --account-name $oidcStorageAccountName `
            --static-website --auth-mode login --output none

        $oidcStorageId = az storage account show --name $oidcStorageAccountName `
            --resource-group $resourceGroup --query id -o tsv

        $accountInfo = az account show --query "user" -o json 2>$null | ConvertFrom-Json
        $principalType = "User"
        $callerObjectId = $null
        if ($accountInfo.type -eq "servicePrincipal") {
            $PSNativeCommandUseErrorActionPreference = $false
            $callerObjectId = az ad sp show --id $accountInfo.name --query id -o tsv 2>$null
            $PSNativeCommandUseErrorActionPreference = $true
            $principalType = "ServicePrincipal"
        } else {
            $callerObjectId = az ad signed-in-user show --query id -o tsv
        }

        if ($callerObjectId) {
            Invoke-AzSafe @("role", "assignment", "create", "--role", "Storage Blob Data Contributor",
                "--assignee-object-id", $callerObjectId, "--assignee-principal-type", $principalType,
                "--scope", $oidcStorageId, "--output", "none")
        }

        $staticWebUrl = (az storage account show --name $oidcStorageAccountName `
            --resource-group $resourceGroup --query "primaryEndpoints.web" -o tsv).TrimEnd('/')
        $issuerUrl = "$staticWebUrl/$containerName"
        $newOidcAccountCreated = $true
    }

    Write-Host "Issuer URL: $issuerUrl" -ForegroundColor Yellow

    if ($newOidcAccountCreated) {
        Write-Host "Waiting 60 seconds for RBAC role assignment to propagate..." -ForegroundColor Yellow
        Start-Sleep -Seconds 60
    }

    $openidConfig = @{
        issuer                                = $issuerUrl
        jwks_uri                              = "$issuerUrl/openid/v1/jwks"
        response_types_supported              = @("id_token")
        subject_types_supported               = @("public")
        id_token_signing_alg_values_supported = @("RS256")
    } | ConvertTo-Json -Depth 4

    $openidConfigPath = Join-Path $outputDir "openid-configuration.json"
    $openidConfig | Out-File -FilePath $openidConfigPath -Encoding utf8

    # Copy JWKS to output dir (skip if already there).
    $jwksPath = Join-Path $outputDir "jwks.json"
    $jwksSrc = (Resolve-Path $JwksFile).Path
    $jwksDst = Join-Path (Resolve-Path $outputDir).Path "jwks.json"
    if ($jwksSrc -ne $jwksDst) {
        Copy-Item -Path $JwksFile -Destination $jwksPath -Force
    }

    Write-Host "Uploading OpenID configuration to static website..." -ForegroundColor Cyan
    az storage blob upload --account-name $oidcStorageAccountName --container-name '$web' `
        --name "$containerName/.well-known/openid-configuration" --file $openidConfigPath `
        --content-type "application/json" --overwrite --auth-mode login --output none

    Write-Host "Uploading JWKS to static website..." -ForegroundColor Cyan
    az storage blob upload --account-name $oidcStorageAccountName --container-name '$web' `
        --name "$containerName/openid/v1/jwks" --file $jwksPath `
        --content-type "application/json" --overwrite --auth-mode login --output none
}

# -- Step 3: Save issuer URL -----------------------------------------------------
$issuerUrlPath = Join-Path $outputDir "issuer-url.txt"
$issuerUrl | Out-File -FilePath $issuerUrlPath -Encoding utf8
Write-Host "Issuer URL saved to '$issuerUrlPath'." -ForegroundColor Green

# -- Step 4: Get managed identity properties and save metadata --------------------
Write-Host "Retrieving managed identity properties..." -ForegroundColor Cyan
$identityJson = az identity show --name $MANAGED_IDENTITY_NAME `
    --resource-group $resourceGroup --output json | ConvertFrom-Json

$identityMetadata = @{
    identityName    = "$persona-identity"
    clientId        = $identityJson.clientId
    tenantId        = $identityJson.tenantId
    tokenIssuerUrl  = $issuerUrl
    backingIdentity = "cleanroom_cgs_oidc"
}

$identityFile = Join-Path $outputDir "identity-metadata.json"
$identityMetadata | ConvertTo-Json -Depth 5 | Out-File -FilePath $identityFile -Encoding utf8
Write-Host "Identity metadata saved to '$identityFile'." -ForegroundColor Green

Write-Host "`nOIDC storage setup complete for '$persona'." -ForegroundColor Green
