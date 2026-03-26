#!/usr/bin/env pwsh
# Verify the collaboration that was just created

$ErrorActionPreference = 'Stop'

$collaborationName = "c20260324-213519"
$resourceGroupName = "cr-e2e-rg"

Write-Host "=== Verifying Collaboration Creation ===" -ForegroundColor Cyan
Write-Host "Collaboration: $collaborationName" -ForegroundColor Yellow
Write-Host "Resource Group: $resourceGroupName" -ForegroundColor Yellow
Write-Host ""

& "$PSScriptRoot/common/invoke-vm-script.ps1" `
    -vmName "cleanroom-e2e-woodgrove-vm" `
    -scriptPath "verify-collaboration.ps1" `
    -scriptArgs @{
        collaborationName = $collaborationName
        resourceGroupName = $resourceGroupName
    } `
    -timeout 300

Write-Host ""
Write-Host "=== Verification Complete ===" -ForegroundColor Green
