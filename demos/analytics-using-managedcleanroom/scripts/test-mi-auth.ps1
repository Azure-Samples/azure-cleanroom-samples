#!/usr/bin/env pwsh
$ErrorActionPreference = "Stop"

Write-Host "Testing Managed Identity authentication..." -ForegroundColor Cyan

# Login with managed identity
Write-Host "Logging in with managed identity..." -ForegroundColor Cyan
az login --identity --allow-no-subscriptions 2>&1 | Out-Null

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to login with managed identity" -ForegroundColor Red
    exit 1
}

$currentUser = az account show --query user -o json | ConvertFrom-Json

if ($currentUser.type -eq "servicePrincipal") {
    Write-Host "Running as Managed Identity: $($currentUser.name)" -ForegroundColor Green
    Write-Host "Subscription: $(az account show --query name -o tsv)" -ForegroundColor Green
    exit 0
} else {
    Write-Host "NOT running as Managed Identity (type: $($currentUser.type))" -ForegroundColor Red
    exit 1
}
