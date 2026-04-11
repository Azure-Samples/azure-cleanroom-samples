<#
.SYNOPSIS
    Sets up local user authentication for the analytics E2E scripts.

.DESCRIPTION
    Replaces the VM managed identity boilerplate used in scripts 04-12.
    Instead of `az login --identity`, this helper:
      1. Verifies the user is already logged in via `az account show`
      2. Optionally sets the correct subscription

    Scripts 04-12 operate on standard Azure resources (storage, KV, MI)
    via normal ARM and call the frontend REST API directly.

    Prerequisites: User must be logged in via `az login` before running any script.

.PARAMETER subscription
    Optional subscription ID to set as the active subscription.
    If not specified, the current subscription is kept.

.EXAMPLE
    # Dot-source at the top of any script:
    . "$PSScriptRoot/common/setup-local-auth.ps1"

.EXAMPLE
    # With subscription switching:
    . "$PSScriptRoot/common/setup-local-auth.ps1" -subscription "dd6ae7e0-4013-486b-9aef-c51cf8eb840a"
#>
param(
    [string]$subscription
)

function Initialize-AppAuth {
    <#
    .SYNOPSIS
        Authenticates as a service principal using MSAL SNI (x5c) cert-based auth.
    .DESCRIPTION
        For MSFT tenant apps with trustedCertificateSubjects in the manifest.
        Acquires an access token via Python MSAL and sets it as the frontend token.
    #>
    param(
        [Parameter(Mandatory)][string]$appId,
        [Parameter(Mandatory)][string]$tenantId,
        [Parameter(Mandatory)][string]$certPemPath,
        [string]$subscription
    )

    Write-Host "Configuring app auth (MSAL SNI)..." -ForegroundColor Cyan

    if (-not (Test-Path $certPemPath)) {
        throw "Certificate PEM file not found: $certPemPath"
    }

    $commonDir = if ($PSScriptRoot) { $PSScriptRoot } else { "." }
    $sniScript = Join-Path $commonDir "get-sp-token-sni.ps1"
    if (Test-Path (Join-Path $commonDir "common" "get-sp-token-sni.ps1")) {
        $sniScript = Join-Path $commonDir "common" "get-sp-token-sni.ps1"
    }

    $token = & $sniScript -appId $appId -tenantId $tenantId -certPemPath $certPemPath
    if (-not $token) {
        throw "Failed to acquire SP token via MSAL SNI"
    }

    $env:CLEANROOM_FRONTEND_TOKEN = $token
    Write-Host "  SP token set in CLEANROOM_FRONTEND_TOKEN" -ForegroundColor Green

    if ($subscription) {
        Write-Host "  Subscription: $subscription" -ForegroundColor Yellow
    }

    Write-Host "App auth configuration complete." -ForegroundColor Green
}

# Default: local user auth
Write-Host "Configuring local user auth..." -ForegroundColor Cyan

$PSNativeCommandUseErrorActionPreference = $false

# Verify user is logged in
$accountJson = az account show -o json 2>$null
if ($LASTEXITCODE -ne 0 -or -not $accountJson) {
    Write-Host "ERROR: Not logged in. Run 'az login' first, then re-run this script." -ForegroundColor Red
    exit 1
}
$PSNativeCommandUseErrorActionPreference = $true

$account = $accountJson | ConvertFrom-Json
Write-Host "  Logged in as: $($account.user.name)" -ForegroundColor Green

# Set subscription if specified
if ($subscription) {
    Write-Host "  Setting subscription: $subscription" -ForegroundColor Yellow
    az account set --subscription $subscription 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to set subscription '$subscription'"
        exit 1
    }
    $account = az account show -o json | ConvertFrom-Json
}

Write-Host "  Subscription: $($account.name) ($($account.id))" -ForegroundColor Green
Write-Host "Local auth configuration complete." -ForegroundColor Green
