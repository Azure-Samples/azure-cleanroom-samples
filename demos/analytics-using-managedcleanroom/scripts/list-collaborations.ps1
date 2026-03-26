#!/usr/bin/env pwsh
# List all collaborations to see what was created

$ErrorActionPreference = 'Stop'

# Configure Private CleanRoom cloud
$env:UsePrivateCleanRoomNamespace = "true"
$privateCloudName = "PrivateCleanroomAzureCloud"
az cloud set --name $privateCloudName 2>&1 | Out-Null
az login --identity --allow-no-subscriptions 2>&1 | Out-Null

Write-Host "=== Listing All Resource Groups ===" -ForegroundColor Cyan
az group list --query "[].name" -o table

Write-Host ""
Write-Host "=== Checking cr-e2e-rg ===" -ForegroundColor Cyan
$rgExists = az group exists --name "cr-e2e-rg"
Write-Host "cr-e2e-rg exists: $rgExists"

if ($rgExists -eq "true") {
    Write-Host ""
    Write-Host "=== Resources in cr-e2e-rg ===" -ForegroundColor Cyan
    az resource list --resource-group "cr-e2e-rg" --query "[].{Name:name, Type:type}" -o table
}

Write-Host ""
Write-Host "=== Searching for Collaborations ===" -ForegroundColor Cyan
$subscriptionId = az account show --query id -o tsv
$apiUrl = "https://management.azure.com/subscriptions/$subscriptionId/providers/Microsoft.CleanRoom/collaborations?api-version=2025-01-31-preview"
az rest --method GET --url $apiUrl --resource "https://management.azure.com/" --query "value[].{Name:name, ResourceGroup:id, ProvisioningState:properties.provisioningState}" -o table 2>&1
