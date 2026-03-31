###############################################################################################
# setup-oidc-issuer.ps1
#
# Sets up an OIDC issuer for a managed cleanroom collaboration. Creates an Azure Storage
# static website to host the OpenID Connect discovery document and JWKS, fetches keys from
# the cleanroom frontend service, and registers the issuer URL with the collaboration.
###############################################################################################

param(
    [Parameter(Mandatory)]
    [string]$resourceGroup,

    [Parameter(Mandatory)]
    [string]$collaborationId,

    [Parameter(Mandatory)]
    [string]$frontendEndpoint,

    [string]$outDir = "./generated",

    [string]$apiVersion = "2026-03-01-preview",

    [string]$TokenFile,

    [ValidateSet("rest", "cli")]
    [string]$ApiMode = "rest",

    # For MSFT-tenant collaborations, use -OidcStorageAccount "cleanroomoidc"
    # (whitelisted SA in azcleanroom-ctest-rg on AzureCleanRoom-NonProd subscription).
    # When set, skips SA creation / static-website / RBAC and uploads OIDC documents
    # to the pre-existing account using $collaborationId as the blob path prefix.
    [string]$OidcStorageAccount
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

# -- Load generated resource names -----------------------------------------------
$namesFile = Join-Path $outDir $resourceGroup "names.generated.ps1"
if (-not (Test-Path $namesFile)) {
    Write-Host "ERROR: '$namesFile' not found. Run prepare-resources.ps1 first." -ForegroundColor Red
    exit 1
}
. $namesFile
. "$PSScriptRoot/frontend-helpers.ps1"

$outputDir = Join-Path $outDir $resourceGroup
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

function Get-ResourceNameHash {
    param([string]$seed)
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($seed)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $hash = $sha.ComputeHash($bytes)
    $hex = -join ($hash | ForEach-Object { $_.ToString("x2") })
    return $hex
}

$subscriptionId = az account show --query id -o tsv

if ($OidcStorageAccount) {
    # -- Using a pre-existing whitelisted OIDC storage account ----------------------
    # Skip SA creation, static website enabling, and RBAC assignment.
    $oidcStorageAccountName = $OidcStorageAccount
    $containerName = $collaborationId
    $staticWebUrl = "https://${oidcStorageAccountName}.z22.web.core.windows.net"
    $issuerUrl = "$staticWebUrl/$containerName"
    Write-Host "Using pre-existing OIDC storage account '$oidcStorageAccountName'." -ForegroundColor Cyan
    Write-Host "Issuer URL: $issuerUrl" -ForegroundColor Yellow
} else {
    # -- Create a new OIDC storage account in the resource group --------------------
    $hash = Get-ResourceNameHash -seed "$subscriptionId-$resourceGroup"
    $oidcStorageAccountName = ("oidc" + $hash).Substring(0, 24).ToLower() -replace '[^a-z0-9]', ''
    $containerName = ("oidc-" + $hash).Substring(0, 24).ToLower() -replace '[^a-z0-9-]', ''

    $location = az group show --name $resourceGroup --query location -o tsv

    Write-Host "Creating OIDC storage account '$oidcStorageAccountName'..." -ForegroundColor Cyan
    az storage account create `
        --name $oidcStorageAccountName `
        --resource-group $resourceGroup `
        --location $location `
        --sku Standard_LRS `
        --output none

    # Enable static website hosting.
    Write-Host "Enabling static website on '$oidcStorageAccountName'..." -ForegroundColor Cyan
    az storage blob service-properties update `
        --account-name $oidcStorageAccountName `
        --static-website `
        --auth-mode login `
        --output none

    # -- RBAC for the caller ---------------------------------------------------------
    $oidcStorageId = az storage account show `
        --name $oidcStorageAccountName `
        --resource-group $resourceGroup `
        --query id -o tsv

    # Detect whether we are logged in as a user or a service principal.
    $accountInfo = az account show --query "user" -o json 2>$null | ConvertFrom-Json
    $principalType = "User"
    $callerObjectId = $null

    if ($accountInfo.type -eq "servicePrincipal") {
        Write-Host "  Detected service principal login." -ForegroundColor Yellow
        $PSNativeCommandUseErrorActionPreference = $false
        $callerObjectId = az ad sp show --id $accountInfo.name --query id -o tsv 2>$null
        $PSNativeCommandUseErrorActionPreference = $true
        $principalType = "ServicePrincipal"
    } else {
        $callerObjectId = az ad signed-in-user show --query id -o tsv
    }

    Write-Host "Assigning 'Storage Blob Data Contributor' on OIDC storage account..." -ForegroundColor Cyan
    if ($callerObjectId) {
        Invoke-AzSafe @("role", "assignment", "create", "--role", "Storage Blob Data Contributor", "--assignee-object-id", $callerObjectId, "--assignee-principal-type", $principalType, "--scope", $oidcStorageId, "--output", "none")
    } else {
        Write-Warning "Could not determine caller object ID - skipping OIDC storage RBAC."
    }

    # -- Retrieve Static Website URL ------------------------------------------------
    Write-Host "Retrieving static website URL..." -ForegroundColor Cyan
    $staticWebUrl = az storage account show `
        --name $oidcStorageAccountName `
        --resource-group $resourceGroup `
        --query "primaryEndpoints.web" -o tsv
    $staticWebUrl = $staticWebUrl.TrimEnd('/')

    $issuerUrl = "$staticWebUrl/$containerName"
    Write-Host "Issuer URL: $issuerUrl" -ForegroundColor Yellow
}

# -- Build openid-configuration.json --------------------------------------------
Write-Host "Building OpenID configuration document..." -ForegroundColor Cyan
$openidConfig = @{
    issuer                 = $issuerUrl
    jwks_uri               = "$issuerUrl/openid/v1/jwks"
    response_types_supported = @("id_token")
    subject_types_supported  = @("public")
    id_token_signing_alg_values_supported = @("RS256")
} | ConvertTo-Json -Depth 4

$openidConfigPath = Join-Path $outputDir "openid-configuration.json"
$openidConfig | Out-File -FilePath $openidConfigPath -Encoding utf8

# -- Fetch JWKS from managed cleanroom frontend ----------------------------------
# Supports two modes:
#   CLI mode:  Uses az managedcleanroom frontend oidc commands
#   REST mode: Makes direct REST calls to the frontend endpoint

# $frontendEndpoint may already include /collaborations — normalize to base URL.
$feBase = $frontendEndpoint.TrimEnd('/')
if ($feBase.EndsWith('/collaborations')) {
    $feBase = $feBase.Substring(0, $feBase.Length - '/collaborations'.Length)
}

Write-Host "Fetching OIDC issuer info from managed cleanroom (mode: $ApiMode)..." -ForegroundColor Cyan

if ($ApiMode -eq "cli") {
    # -- CLI mode: Configure the CLI extension and use oidc commands --
    $token = Get-FrontendToken -TokenFile $TokenFile
    $env:MANAGEDCLEANROOM_ACCESS_TOKEN = $token
    $env:AZURE_CLI_DISABLE_CONNECTION_VERIFICATION = "1"
    # SDK URL templates already include /collaborations/ prefix — use bare base URL.
    az managedcleanroom frontend configure --endpoint $feBase 2>&1 | Out-Null

    # Fetch OIDC issuer info
    try {
        $PSNativeCommandUseErrorActionPreference = $false
        $oidcIssuerInfoRaw = az managedcleanroom frontend oidc issuerinfo show `
            --collaboration-id $collaborationId 2>&1
        $PSNativeCommandUseErrorActionPreference = $true
        # Filter out WARNING lines from stderr captured by 2>&1
        $oidcIssuerInfoJson = $oidcIssuerInfoRaw | Where-Object { $_ -is [string] }
        if ($LASTEXITCODE -eq 0 -and $oidcIssuerInfoJson) {
            $oidcIssuerInfo = $oidcIssuerInfoJson | ConvertFrom-Json
            Write-Host "OIDC issuer info:" -ForegroundColor Yellow
            $oidcIssuerInfo | ConvertTo-Json -Depth 5
        }
    } catch {
        Write-Warning "Failed to fetch OIDC issuer info: $_"
        Write-Host "  Continuing anyway (issuer info is informational only)..." -ForegroundColor Yellow
    }

    # Fetch JWKS
    Write-Host "Fetching JWKS from frontend endpoint (CLI)..." -ForegroundColor Cyan
    $jwksRaw = az managedcleanroom frontend oidc keys `
        --collaboration-id $collaborationId
    if ($LASTEXITCODE -ne 0 -or -not $jwksRaw) {
        throw "Failed to fetch JWKS via CLI"
    }
    $jwksResponse = $jwksRaw | ConvertFrom-Json
    $jwksJson = $jwksResponse | ConvertTo-Json -Depth 10
} else {
    # -- REST mode: Direct REST calls (original approach) --

    $token = Get-FrontendToken -TokenFile $TokenFile
    $headers = @{
        Authorization  = "Bearer $token"
        "Content-Type" = "application/json"
    }

    $oidcInfoUrl = "$feBase/collaborations/$collaborationId/oidc/issuerinfo?api-version=$apiVersion"
    Write-Host "  GET $oidcInfoUrl" -ForegroundColor Gray
    try {
        $oidcIssuerInfo = Invoke-RestMethod -Uri $oidcInfoUrl -Headers $headers -Method Get -SkipCertificateCheck
        Write-Host "OIDC issuer info:" -ForegroundColor Yellow
        $oidcIssuerInfo | ConvertTo-Json -Depth 5
    } catch {
        Write-Warning "Failed to fetch OIDC issuer info: $_"
        Write-Host "  Continuing anyway (issuer info is informational only)..." -ForegroundColor Yellow
    }

    Write-Host "Fetching JWKS from frontend endpoint (direct REST call)..." -ForegroundColor Cyan
    $token = Get-FrontendToken -TokenFile $TokenFile
    $headers = @{
        Authorization  = "Bearer $token"
        "Content-Type" = "application/json"
    }
    $jwksUrl = "$feBase/collaborations/$collaborationId/oidc/keys?api-version=$apiVersion"
    $jwksResponse = Invoke-RestMethod -Uri $jwksUrl -Headers $headers -Method Get -SkipCertificateCheck
    $jwksJson = $jwksResponse | ConvertTo-Json -Depth 10
}

$jwksPath = Join-Path $outputDir "jwks.json"
$jwksJson | Out-File -FilePath $jwksPath -Encoding utf8
Write-Host "JWKS saved to '$jwksPath'." -ForegroundColor Green

# -- Upload documents to $web container ------------------------------------------
Write-Host "Uploading OpenID configuration to static website..." -ForegroundColor Cyan
az storage blob upload `
    --account-name $oidcStorageAccountName `
    --container-name '$web' `
    --name "$containerName/.well-known/openid-configuration" `
    --file $openidConfigPath `
    --content-type "application/json" `
    --overwrite `
    --auth-mode login `
    --output none

Write-Host "Uploading JWKS to static website..." -ForegroundColor Cyan
az storage blob upload `
    --account-name $oidcStorageAccountName `
    --container-name '$web' `
    --name "$containerName/openid/v1/jwks" `
    --file $jwksPath `
    --content-type "application/json" `
    --overwrite `
    --auth-mode login `
    --output none

# -- Register issuer URL with CGS ------------------------------------------------

Write-Host "Registering issuer URL with managed cleanroom (mode: $ApiMode)..." -ForegroundColor Cyan

if ($ApiMode -eq "cli") {
    # CLI mode
    $token = Get-FrontendToken -TokenFile $TokenFile
    $env:MANAGEDCLEANROOM_ACCESS_TOKEN = $token
    $env:AZURE_CLI_DISABLE_CONNECTION_VERIFICATION = "1"
    az managedcleanroom frontend configure --endpoint $feBase 2>&1 | Out-Null

    az managedcleanroom frontend oidc set-issuer-url `
        --collaboration-id $collaborationId `
        --url $issuerUrl
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to register issuer URL via CLI"
    }
} else {
    # REST mode (original approach)
    $token = Get-FrontendToken -TokenFile $TokenFile
    $headers = @{
        Authorization  = "Bearer $token"
        "Content-Type" = "application/json"
    }
    $setIssuerBody = @{ url = $issuerUrl } | ConvertTo-Json
    Invoke-RestMethod `
        -Uri "$feBase/collaborations/$collaborationId/oidc/setIssuerUrl?api-version=$apiVersion" `
        -Headers $headers `
        -Method Post `
        -Body $setIssuerBody `
        -ContentType "application/json" `
        -SkipCertificateCheck
}
Write-Host "Issuer URL registered with CGS." -ForegroundColor Green

# -- Save issuer URL ------------------------------------------------------------
$issuerUrlPath = Join-Path $outputDir "issuer-url.txt"
$issuerUrl | Out-File -FilePath $issuerUrlPath -Encoding utf8
Write-Host "Issuer URL saved to '$issuerUrlPath'." -ForegroundColor Green

Write-Host "OIDC issuer setup complete." -ForegroundColor Green
