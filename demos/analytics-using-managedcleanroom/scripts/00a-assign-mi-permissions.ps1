param(
    [string]$subscriptionId = "",
    [switch]$skipPermissionAssignment
)

$ErrorActionPreference = "Stop"

# Get current subscription if not provided
if ([string]::IsNullOrEmpty($subscriptionId)) {
    $subscriptionId = (az account show --query id -o tsv)
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to get current subscription ID"
        exit 1
    }
}

Write-Host "Using subscription: $subscriptionId" -ForegroundColor Cyan

if ($skipPermissionAssignment) {
    Write-Host "Skipping permission assignment (skipPermissionAssignment flag set)" -ForegroundColor Yellow
    exit 0
}

# Define paths to VM metadata files
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$northwindVmPath = Join-Path $scriptDir "generated/vms/northwind-vm.json"
$woodgroveVmPath = Join-Path $scriptDir "generated/vms/woodgrove-vm.json"

# Check if files exist
if (-not (Test-Path $northwindVmPath)) {
    Write-Error "Northwind VM metadata file not found: $northwindVmPath"
    exit 1
}

if (-not (Test-Path $woodgroveVmPath)) {
    Write-Error "Woodgrove VM metadata file not found: $woodgroveVmPath"
    exit 1
}

# Read VM metadata
Write-Host "Reading VM metadata..." -ForegroundColor Cyan
$northwindVm = Get-Content $northwindVmPath | ConvertFrom-Json
$woodgroveVm = Get-Content $woodgroveVmPath | ConvertFrom-Json

# Extract principal IDs
$northwindPrincipalId = $northwindVm.identity.principalId
$woodgrovePrincipalId = $woodgroveVm.identity.principalId

if ([string]::IsNullOrEmpty($northwindPrincipalId)) {
    Write-Error "Failed to extract principal ID from Northwind VM metadata"
    exit 1
}

if ([string]::IsNullOrEmpty($woodgrovePrincipalId)) {
    Write-Error "Failed to extract principal ID from Woodgrove VM metadata"
    exit 1
}

Write-Host "Northwind VM Principal ID: $northwindPrincipalId" -ForegroundColor Green
Write-Host "Woodgrove VM Principal ID: $woodgrovePrincipalId" -ForegroundColor Green

# Define roles to assign
$roles = @("Contributor", "User Access Administrator")
$scope = "/subscriptions/$subscriptionId"

# Function to check if role assignment exists
function Test-RoleAssignment {
    param(
        [string]$principalId,
        [string]$role
    )
    
    $existing = az role assignment list --assignee $principalId --role $role --scope $scope --query "[0].id" -o tsv 2>$null
    return -not [string]::IsNullOrEmpty($existing)
}

# Function to assign role
function Add-RoleAssignment {
    param(
        [string]$principalId,
        [string]$role,
        [string]$vmName
    )
    
    if (Test-RoleAssignment -principalId $principalId -role $role) {
        Write-Host "  Role '$role' already assigned to $vmName MI" -ForegroundColor Yellow
        return $true
    }
    
    Write-Host "  Assigning role '$role' to $vmName MI..." -ForegroundColor Cyan
    az role assignment create --assignee $principalId --role $role --scope $scope -o none 2>$null
    
    if ($LASTEXITCODE -ne 0) {
        # Try to check if it was created despite the error (race condition)
        if (Test-RoleAssignment -principalId $principalId -role $role) {
            Write-Host "  Role '$role' assigned to $vmName MI (already existed)" -ForegroundColor Yellow
            return $true
        }
        Write-Error "Failed to assign role '$role' to $vmName MI"
        return $false
    }
    
    Write-Host "  Role '$role' assigned to $vmName MI" -ForegroundColor Green
    return $true
}

# Assign roles to Northwind VM
Write-Host "`nAssigning roles to Northwind VM managed identity..." -ForegroundColor Cyan
foreach ($role in $roles) {
    if (-not (Add-RoleAssignment -principalId $northwindPrincipalId -role $role -vmName "Northwind")) {
        exit 1
    }
}

# Assign roles to Woodgrove VM
Write-Host "`nAssigning roles to Woodgrove VM managed identity..." -ForegroundColor Cyan
foreach ($role in $roles) {
    if (-not (Add-RoleAssignment -principalId $woodgrovePrincipalId -role $role -vmName "Woodgrove")) {
        exit 1
    }
}

# Wait for RBAC propagation
Write-Host "`nWaiting 90 seconds for RBAC propagation..." -ForegroundColor Cyan
Start-Sleep -Seconds 90

# Verify role assignments
Write-Host "`nVerifying role assignments..." -ForegroundColor Cyan

Write-Host "`nNorthwind VM role assignments:" -ForegroundColor Cyan
$northwindRoles = az role assignment list --assignee $northwindPrincipalId --scope $scope --query "[].roleDefinitionName" -o tsv
if ($LASTEXITCODE -eq 0) {
    $northwindRoles | ForEach-Object { Write-Host "  $_" -ForegroundColor Green }
} else {
    Write-Error "Failed to verify Northwind VM role assignments"
    exit 1
}

Write-Host "`nWoodgrove VM role assignments:" -ForegroundColor Cyan
$woodgroveRoles = az role assignment list --assignee $woodgrovePrincipalId --scope $scope --query "[].roleDefinitionName" -o tsv
if ($LASTEXITCODE -eq 0) {
    $woodgroveRoles | ForEach-Object { Write-Host "  $_" -ForegroundColor Green }
} else {
    Write-Error "Failed to verify Woodgrove VM role assignments"
    exit 1
}

# Verify both required roles are assigned
foreach ($role in $roles) {
    if ($northwindRoles -notcontains $role) {
        Write-Error "Role '$role' not found in Northwind VM assignments"
        exit 1
    }
    if ($woodgroveRoles -notcontains $role) {
        Write-Error "Role '$role' not found in Woodgrove VM assignments"
        exit 1
    }
}

Write-Host "`nAll role assignments completed and verified successfully" -ForegroundColor Green
