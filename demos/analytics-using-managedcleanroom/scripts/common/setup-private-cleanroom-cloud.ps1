<#
.SYNOPSIS
    Configures Azure CLI to use Private CleanRoom cloud for dogfood environment.

.DESCRIPTION
    Sets up the Private CleanRoom cloud configuration and logs in with managed identity.
    This is required for accessing the dogfood CleanRoom service in eastus2euap.

.PARAMETER skipLogin
    If specified, skips the login step (useful if already logged in).

.EXAMPLE
    # Setup Private CleanRoom cloud and login
    . ./common/setup-private-cleanroom-cloud.ps1

.EXAMPLE
    # Setup cloud configuration only, skip login
    . ./common/setup-private-cleanroom-cloud.ps1 -skipLogin
#>
param(
    [switch]$skipLogin
)

Write-Host "Configuring Private CleanRoom cloud..." -ForegroundColor Cyan

# Set environment variable for Private CleanRoom namespace
$env:UsePrivateCleanRoomNamespace = "true"
$privateCloudName = "PrivateCleanroomAzureCloud"

# Register the private cloud if not already registered
$existingCloud = az cloud list --query "[?name=='$privateCloudName']" -o json 2>$null | ConvertFrom-Json
if (-not $existingCloud) {
    Write-Host "  Registering Private CleanRoom cloud..." -ForegroundColor Yellow
    az cloud register --name $privateCloudName --endpoint-resource-manager "https://eastus2euap.management.azure.com/" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to register Private CleanRoom cloud"
        return $false
    }
    Write-Host "  Private CleanRoom cloud registered." -ForegroundColor Green
} else {
    Write-Host "  Private CleanRoom cloud already registered." -ForegroundColor Green
}

# Set the cloud to Private CleanRoom
Write-Host "  Switching to Private CleanRoom cloud..." -ForegroundColor Yellow
az cloud set --name $privateCloudName 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to set Private CleanRoom cloud"
    return $false
}
Write-Host "  Cloud set to: $privateCloudName" -ForegroundColor Green

# Login with managed identity if not skipped
if (-not $skipLogin) {
    Write-Host "  Logging in with managed identity..." -ForegroundColor Yellow
    az login --identity --allow-no-subscriptions 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to login with managed identity"
        return $false
    }
    Write-Host "  Logged in successfully." -ForegroundColor Green
}

Write-Host "Private CleanRoom cloud configuration complete." -ForegroundColor Green
return $true
