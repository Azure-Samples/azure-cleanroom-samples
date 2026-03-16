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

# -- Load generated resource names -----------------------------------------------
$namesFile = Join-Path $outDir $resourceGroup "names.generated.ps1"
if (-not (Test-Path $namesFile)) {
    Write-Host "ERROR: '$namesFile' not found. Run prepare-resources.ps1 first." -ForegroundColor Red
    exit 1
}
. $namesFile

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
$hash = Get-ResourceNameHash -seed "$subscriptionId-$resourceGroup"
$oidcStorageAccountName = ("oidc" + $hash).Substring(0, 24).ToLower() -replace '[^a-z0-9]', ''
$containerName = ("oidc-" + $hash).Substring(0, 24).ToLower() -replace '[^a-z0-9-]', ''

# -- Create OIDC Storage Account ------------------------------------------------
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

# -- RBAC for logged-in user -----------------------------------------------------
$oidcStorageId = az storage account show `
    --name $oidcStorageAccountName `
    --resource-group $resourceGroup `
    --query id -o tsv

$callerObjectId = az ad signed-in-user show --query id -o tsv

Write-Host "Assigning 'Storage Blob Data Contributor' on OIDC storage account..." -ForegroundColor Cyan
Invoke-AzSafe @("role", "assignment", "create", "--role", "Storage Blob Data Contributor", "--assignee-object-id", $callerObjectId, "--assignee-principal-type", "User", "--scope", $oidcStorageId, "--output", "none")

# -- Retrieve Static Website URL ------------------------------------------------
Write-Host "Retrieving static website URL..." -ForegroundColor Cyan
$staticWebUrl = az storage account show `
    --name $oidcStorageAccountName `
    --resource-group $resourceGroup `
    --query "primaryEndpoints.web" -o tsv
$staticWebUrl = $staticWebUrl.TrimEnd('/')

$issuerUrl = "$staticWebUrl/$containerName"
Write-Host "Issuer URL: $issuerUrl" -ForegroundColor Yellow

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

# -- Fetch JWKS from managed cleanroom frontend (direct REST call) ---------------
# TODO: Replace this direct REST call with CLI command when available:
#   az managedcleanroom frontend oidc keys show --collaboration-id $collaborationId
# The frontend exposes JWKS at GET /collaborations/{id}/oidc/keys, but no CLI command
# wraps it yet. We call the endpoint directly using the configured frontend endpoint.

Write-Host "Fetching OIDC issuer info from managed cleanroom..." -ForegroundColor Cyan
$oidcIssuerInfo = az managedcleanroom frontend oidc issuerinfo show `
    --collaboration-id $collaborationId | ConvertFrom-Json

Write-Host "OIDC issuer info:" -ForegroundColor Yellow
$oidcIssuerInfo | ConvertTo-Json -Depth 5

Write-Host "Fetching JWKS from frontend endpoint (direct REST call)..." -ForegroundColor Cyan
$token = az account get-access-token --query accessToken -o tsv
$headers = @{
    Authorization  = "Bearer $token"
    "Content-Type" = "application/json"
}
$jwksUrl = "$($frontendEndpoint.TrimEnd('/'))/collaborations/$collaborationId/oidc/keys"
$jwksResponse = Invoke-RestMethod -Uri $jwksUrl -Headers $headers -Method Get
$jwksJson = $jwksResponse | ConvertTo-Json -Depth 10

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
# TODO: Replace this direct REST call with CLI command when available:
#   az managedcleanroom frontend oidc set-issuer-url --url $issuerUrl --collaboration-id $collaborationId
# The frontend exposes this at POST /collaborations/{id}/oidc/setIssuerUrl, but no CLI
# command wraps it yet (confirmed: only `oidc issuerinfo show` exists in the CLI).

Write-Host "Registering issuer URL with managed cleanroom (direct REST call)..." -ForegroundColor Cyan
$setIssuerBody = @{ url = $issuerUrl } | ConvertTo-Json
Invoke-RestMethod `
    -Uri "$($frontendEndpoint.TrimEnd('/'))/collaborations/$collaborationId/oidc/setIssuerUrl" `
    -Headers $headers `
    -Method Post `
    -Body $setIssuerBody `
    -ContentType "application/json"
Write-Host "Issuer URL registered with CGS." -ForegroundColor Green

# -- Save issuer URL ------------------------------------------------------------
$issuerUrlPath = Join-Path $outputDir "issuer-url.txt"
$issuerUrl | Out-File -FilePath $issuerUrlPath -Encoding utf8
Write-Host "Issuer URL saved to '$issuerUrlPath'." -ForegroundColor Green

Write-Host "OIDC issuer setup complete." -ForegroundColor Green
