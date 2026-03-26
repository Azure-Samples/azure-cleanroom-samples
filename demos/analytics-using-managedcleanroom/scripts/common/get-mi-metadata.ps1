<#
.SYNOPSIS
    Extracts managed identity metadata from VM JSON files.

.DESCRIPTION
    Reads VM metadata JSON files and returns managed identity principal IDs,
    tenant ID, and VM information for use in other scripts.

.PARAMETER persona
    Persona name (northwind or woodgrove). If specified, returns metadata for that persona only.

.PARAMETER vmMetadataDir
    Directory containing VM JSON files (default: ../generated/vms).

.OUTPUTS
    Returns a hashtable with the following keys:
    - northwindPrincipalId: Northwind MI principal ID
    - woodgrovePrincipalId: Woodgrove MI principal ID
    - tenantId: Azure tenant ID
    - northwindVmName: Northwind VM name
    - woodgroveVmName: Woodgrove VM name
    - northwindResourceGroup: Northwind VM resource group
    - woodgroveResourceGroup: Woodgrove VM resource group
    
    If -persona is specified, returns a hashtable with:
    - principalId: MI principal ID for the specified persona
    - tenantId: Azure tenant ID
    - vmName: VM name
    - resourceGroup: Resource group name
    - location: Azure region

.EXAMPLE
    # Get all metadata
    $metadata = ./common/get-mi-metadata.ps1
    Write-Host "Northwind Principal ID: $($metadata.northwindPrincipalId)"
    
.EXAMPLE
    # Get metadata for specific persona
    $northwind = ./common/get-mi-metadata.ps1 -persona northwind
    Write-Host "Principal ID: $($northwind.principalId)"
#>
param(
    [ValidateSet("northwind", "woodgrove")]
    [string]$persona,
    
    [string]$vmMetadataDir = "$PSScriptRoot/../generated/vms"
)

$ErrorActionPreference = 'Stop'

# Resolve relative path
$vmMetadataDir = Resolve-Path $vmMetadataDir -ErrorAction Stop

# Read VM metadata files
$northwindFile = Join-Path $vmMetadataDir "northwind-vm.json"
$woodgroveFile = Join-Path $vmMetadataDir "woodgrove-vm.json"

if (-not (Test-Path $northwindFile)) {
    throw "Northwind VM metadata file not found: $northwindFile"
}

if (-not (Test-Path $woodgroveFile)) {
    throw "Woodgrove VM metadata file not found: $woodgroveFile"
}

$northwindData = Get-Content $northwindFile -Raw | ConvertFrom-Json
$woodgroveData = Get-Content $woodgroveFile -Raw | ConvertFrom-Json

# Validate tenant IDs match
if ($northwindData.identity.tenantId -ne $woodgroveData.identity.tenantId) {
    throw "Tenant IDs do not match between VMs: Northwind=$($northwindData.identity.tenantId), Woodgrove=$($woodgroveData.identity.tenantId)"
}

$tenantId = $northwindData.identity.tenantId

# Return persona-specific metadata if requested
if ($persona) {
    $data = if ($persona -eq "northwind") { $northwindData } else { $woodgroveData }
    
    return @{
        principalId = $data.identity.principalId
        tenantId = $data.identity.tenantId
        vmName = $data.name
        resourceGroup = $data.resourceGroup
        location = $data.location
    }
}

# Return all metadata
return @{
    northwindPrincipalId = $northwindData.identity.principalId
    woodgrovePrincipalId = $woodgroveData.identity.principalId
    tenantId = $tenantId
    northwindVmName = $northwindData.name
    woodgroveVmName = $woodgroveData.name
    northwindResourceGroup = $northwindData.resourceGroup
    woodgroveResourceGroup = $woodgroveData.resourceGroup
    northwindLocation = $northwindData.location
    woodgroveLocation = $woodgroveData.location
}
