#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Tests collaboration verification using current Azure CLI session (no MI required).

.DESCRIPTION
    This script verifies collaboration creation by querying ARM API using the 
    Private.CleanRoom namespace. It uses the current Azure CLI session instead
    of Managed Identity login, so it can run locally.
#>

$ErrorActionPreference = 'Stop'

$collaborationName = "c20260324-213519"
$resourceGroupName = "cr-e2e-rg"

Write-Host "=== Verifying Collaboration (Local Test) ===" -ForegroundColor Cyan
Write-Host "Collaboration: $collaborationName" -ForegroundColor Yellow
Write-Host "Resource Group: $resourceGroupName" -ForegroundColor Yellow
Write-Host ""

# Configure Private CleanRoom cloud for dogfood environment
Write-Host "Configuring Private CleanRoom cloud..." -ForegroundColor Cyan
$env:UsePrivateCleanRoomNamespace = "true"
$privateCloudName = "PrivateCleanroomAzureCloud"

# Register the private cloud if not already registered
$existingCloud = az cloud list --query "[?name=='$privateCloudName']" -o json 2>$null | ConvertFrom-Json
if (-not $existingCloud) {
    Write-Host "  Registering Private CleanRoom cloud..." -ForegroundColor Yellow
    az cloud register --name $privateCloudName --endpoint-resource-manager "https://eastus2euap.management.azure.com/" 2>&1 | Out-Null
    Write-Host "  Private CleanRoom cloud registered." -ForegroundColor Green
}

# Set the cloud to Private CleanRoom
Write-Host "  Switching to Private CleanRoom cloud..." -ForegroundColor Yellow
az cloud set --name $privateCloudName 2>&1 | Out-Null

# Need to login again after switching clouds
Write-Host "  Logging in to Private CleanRoom cloud..." -ForegroundColor Yellow
az login --allow-no-subscriptions 2>&1 | Out-Null

Write-Host "Private CleanRoom cloud configuration complete." -ForegroundColor Green
Write-Host ""

# Get subscription ID
$subscriptionId = az account show --query id -o tsv
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to get subscription ID"
    exit 1
}

Write-Host "Subscription ID: $subscriptionId" -ForegroundColor Gray

# Build ARM API URL with Private.CleanRoom namespace
$apiVersion = "2025-01-31-preview"
$apiUrl = "https://management.azure.com/subscriptions/$subscriptionId/resourceGroups/$resourceGroupName/providers/Private.CleanRoom/collaborations/${collaborationName}?api-version=$apiVersion"

Write-Host "API URL: $apiUrl" -ForegroundColor Gray
Write-Host ""

# Call ARM API
Write-Host "Fetching collaboration details..." -ForegroundColor Cyan
$response = az rest --method GET --url $apiUrl --resource "https://management.azure.com/" 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to fetch collaboration" -ForegroundColor Red
    Write-Host $response -ForegroundColor Red
    
    # Try listing resources to compare
    Write-Host ""
    Write-Host "=== Resources in Resource Group (for comparison) ===" -ForegroundColor Yellow
    az resource list --resource-group $resourceGroupName --query "[].{Name:name, Type:type, Location:location}" --output table
    
    exit 1
}

# Parse and display key details
$collaboration = $response | ConvertFrom-Json

Write-Host "=== Collaboration Details ===" -ForegroundColor Green
Write-Host "Name: $($collaboration.name)" -ForegroundColor White
Write-Host "ID: $($collaboration.id)" -ForegroundColor Gray
Write-Host "Location: $($collaboration.location)" -ForegroundColor White
Write-Host "Provisioning State: $($collaboration.properties.provisioningState)" -ForegroundColor $(if ($collaboration.properties.provisioningState -eq 'Succeeded') { 'Green' } else { 'Yellow' })
Write-Host "Consortium Type: $($collaboration.properties.consortiumType)" -ForegroundColor White
Write-Host ""

# Show workloads
if ($collaboration.properties.workloads) {
    Write-Host "=== Workloads ===" -ForegroundColor Green
    foreach ($workload in $collaboration.properties.workloads) {
        Write-Host "  Type: $($workload.workloadType)" -ForegroundColor White
        Write-Host "  Status: $($workload.status)" -ForegroundColor $(if ($workload.status -eq 'Ready') { 'Green' } else { 'Yellow' })
        if ($workload.endpoint) {
            Write-Host "  Endpoint: $($workload.endpoint)" -ForegroundColor Gray
        }
        Write-Host ""
    }
}

# Show collaborators
if ($collaboration.properties.collaborators) {
    Write-Host "=== Collaborators ===" -ForegroundColor Green
    foreach ($collaborator in $collaboration.properties.collaborators) {
        Write-Host "  Tenant ID: $($collaborator.tenantId)" -ForegroundColor White
        Write-Host "  Object ID: $($collaborator.objectId)" -ForegroundColor Gray
        Write-Host ""
    }
}

Write-Host ""
Write-Host "=== Full JSON Response ===" -ForegroundColor Cyan
$response | ConvertFrom-Json | ConvertTo-Json -Depth 10
