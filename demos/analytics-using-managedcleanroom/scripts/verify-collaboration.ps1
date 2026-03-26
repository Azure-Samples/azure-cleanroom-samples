#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Verifies collaboration creation by querying ARM API.

.PARAMETER collaborationName
    Name of the collaboration to verify.

.PARAMETER resourceGroupName
    Resource group containing the collaboration.

.EXAMPLE
    ./verify-collaboration.ps1 -collaborationName "c20260324-213519" -resourceGroupName "cr-e2e-rg"
#>
param(
    [Parameter(Mandatory)]
    [string]$collaborationName,
    
    [Parameter(Mandatory)]
    [string]$resourceGroupName
)

$ErrorActionPreference = 'Stop'

# Configure Private CleanRoom cloud for dogfood environment
Write-Host "Configuring Private CleanRoom cloud..." -ForegroundColor Cyan
$env:UsePrivateCleanRoomNamespace = "true"
$privateCloudName = "PrivateCleanroomAzureCloud"

# Register the private cloud if not already registered
$existingCloud = az cloud list --query "[?name=='$privateCloudName']" -o json 2>$null | ConvertFrom-Json
if (-not $existingCloud) {
    Write-Host "  Registering Private CleanRoom cloud..." -ForegroundColor Yellow
    az cloud register --name $privateCloudName --endpoint-resource-manager "https://eastus2euap.management.azure.com/" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to register Private CleanRoom cloud"
        exit 1
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
    exit 1
}

# Login with managed identity to Private CleanRoom cloud
Write-Host "  Logging in with managed identity..." -ForegroundColor Yellow
az login --identity --allow-no-subscriptions 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to login with managed identity"
    exit 1
}
Write-Host "Private CleanRoom cloud configuration complete." -ForegroundColor Green
Write-Host ""

Write-Host "=== Verifying Collaboration ===" -ForegroundColor Cyan
Write-Host "Collaboration: $collaborationName" -ForegroundColor Yellow
Write-Host "Resource Group: $resourceGroupName" -ForegroundColor Yellow
Write-Host ""

# Get subscription ID
$subscriptionId = az account show --query id -o tsv
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to get subscription ID"
    exit 1
}

Write-Host "Subscription ID: $subscriptionId" -ForegroundColor Gray

# Build ARM API URL (use Private.CleanRoom for dogfood environment)
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
} else {
    Write-Host "No workloads configured" -ForegroundColor Yellow
}

# Show collaborators
if ($collaboration.properties.collaborators) {
    Write-Host "=== Collaborators ===" -ForegroundColor Green
    foreach ($collaborator in $collaboration.properties.collaborators) {
        Write-Host "  Tenant ID: $($collaborator.tenantId)" -ForegroundColor White
        Write-Host "  Object ID: $($collaborator.objectId)" -ForegroundColor Gray
        Write-Host ""
    }
} else {
    Write-Host "No collaborators added yet" -ForegroundColor Yellow
}

# Show resource group for AKS cluster
if ($collaboration.properties.provisioningState -eq 'Succeeded') {
    Write-Host "=== Associated Resources ===" -ForegroundColor Green
    
    # The AKS cluster is typically in a managed resource group
    # Format: MC_<resourceGroup>_<collaborationName>_<location>
    $managedRgName = "MC_${resourceGroupName}_${collaborationName}_$($collaboration.location)"
    Write-Host "Expected Managed Resource Group: $managedRgName" -ForegroundColor Gray
    
    # Check if the managed RG exists
    $rgExists = az group exists --name $managedRgName
    if ($rgExists -eq "true") {
        Write-Host "✓ Managed resource group exists" -ForegroundColor Green
        
        # List resources in the managed RG
        Write-Host ""
        Write-Host "Resources in managed RG:" -ForegroundColor Cyan
        az resource list --resource-group $managedRgName --query "[].{Name:name, Type:type}" --output table
    } else {
        Write-Host "⚠ Managed resource group not found (might be using different naming)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=== Full JSON Response ===" -ForegroundColor Cyan
$response | ConvertFrom-Json | ConvertTo-Json -Depth 10
