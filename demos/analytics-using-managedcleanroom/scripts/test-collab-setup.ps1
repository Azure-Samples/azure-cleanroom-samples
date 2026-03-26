#!/usr/bin/env pwsh
# Test script for collaboration setup with managed identities

$ErrorActionPreference = 'Stop'

# Get MI metadata
$metadata = & "$PSScriptRoot/common/get-mi-metadata.ps1"

# Generate unique timestamp
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"

Write-Host "=== Testing Collaboration Setup with Managed Identities ===" -ForegroundColor Cyan
Write-Host "Timestamp: $timestamp" -ForegroundColor Yellow
Write-Host "Northwind MI Principal ID: $($metadata.northwindPrincipalId)" -ForegroundColor Yellow
Write-Host "Woodgrove MI Principal ID: $($metadata.woodgrovePrincipalId)" -ForegroundColor Yellow
Write-Host "NOTE: Using westus region for Private CleanRoom namespace" -ForegroundColor Yellow
Write-Host ""

# Invoke collaboration setup on Woodgrove VM
& "$PSScriptRoot/common/invoke-vm-script.ps1" `
    -vmName "cleanroom-e2e-woodgrove-vm" `
    -scriptPath "01-setup-collaboration.ps1" `
    -scriptArgs @{
        collaborationName = "c$timestamp"
        resourceGroupName = "cr-e2e-rg"
        location = "westus"
        northwindMIPrincipalId = $metadata.northwindPrincipalId
        woodgroveMIPrincipalId = $metadata.woodgrovePrincipalId
        persona = "woodgrove"
    } `
    -timeout 900

Write-Host ""
Write-Host "=== Test Complete ===" -ForegroundColor Green
