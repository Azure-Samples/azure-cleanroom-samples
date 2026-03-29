#!/usr/bin/env pwsh
<#
.SYNOPSIS
    E2E orchestrator for Azure CleanRoom analytics demo using VM-based managed identities.

.DESCRIPTION
    Executes a full end-to-end test of Azure Managed CleanRoom analytics workflow using
    2 Azure VMs with system-assigned managed identities. Orchestrates all phases of the
    workflow sequentially for simplicity and reliability.

.PARAMETER encryptionType
    Encryption method: SSE (server-side) or CPK (customer-provided keys). Default: SSE

.PARAMETER queryFormat
    Query format: inline (query.txt) or json (JSON template). Default: inline

.PARAMETER skipInfrastructure
    Skip Phase 0 (infrastructure setup). Use if VMs are already provisioned.

.PARAMETER cleanupOnSuccess
    Delete all test resources after successful completion.

.PARAMETER cleanupOnFailure
    Delete all test resources after failure (for cleanup).

.PARAMETER useExistingCollaboration
    Use existing collaboration instead of creating new one.

.PARAMETER existingCollaborationName
    Name of existing collaboration to use (requires -useExistingCollaboration).

.PARAMETER existingCollaborationRG
    Resource group of existing collaboration (requires -useExistingCollaboration).

.EXAMPLE
    ./run-e2e-test-vms.ps1
    Run full E2E test with default settings (SSE encryption, inline query)

.EXAMPLE
    ./run-e2e-test-vms.ps1 -skipInfrastructure
    Run E2E test using existing VMs

.EXAMPLE
    ./run-e2e-test-vms.ps1 -encryptionType CPK -queryFormat json
    Run E2E test with CPK encryption and JSON query format
#>

param(
    [ValidateSet("SSE", "CPK")]
    [string]$encryptionType = "SSE",
    
    [ValidateSet("inline", "json")]
    [string]$queryFormat = "inline",
    
    [ValidateSet("rest", "cli")]
    [string]$apiMode = "rest",
    
    [switch]$skipInfrastructure,
    [switch]$cleanupOnSuccess,
    [switch]$cleanupOnFailure,
    
    [switch]$useExistingCollaboration,
    [string]$existingCollaborationName,
    [string]$existingCollaborationRG
)

$ErrorActionPreference = 'Stop'
$scriptStartTime = Get-Date
$scriptDir = Split-Path -Parent $PSCommandPath

# ============================================================================
# Configuration
# ============================================================================

$config = @{
    # VM names
    northwindVM = "cleanroom-e2e-northwind-vm"
    woodgroveVM = "cleanroom-e2e-woodgrove-vm"
    vmResourceGroup = "cleanroom-e2e-vms-rg"
    
    # Test configuration
    location = "westus"  # Private CleanRoom namespace requires westus
    subscriptionId = "fccb68eb-8ccf-49a6-a69a-7ea3c2867e9c"
    
    # Static frontend endpoint for westus
    frontendEndpoint = "https://dogfood.workload-frontendwestus.cleanroom.cloudapp.azure-test.net/collaborations"
    
    # Timeouts (seconds)
    timeouts = @{
        infrastructure = 900   # 15 min
        collaboration = 900    # 15 min
        resourcePrep = 300     # 5 min
        dataPrep = 300         # 5 min
        identity = 180         # 3 min
        access = 180           # 3 min
        dataset = 300          # 5 min
        query = 180            # 3 min
        vote = 120             # 2 min
        execute = 5400         # 90 min
        results = 120          # 2 min
    }
}

# ============================================================================
# Helper Functions
# ============================================================================

function Write-PhaseHeader {
    param([string]$phase, [string]$description)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "[$timestamp] $phase" -ForegroundColor Cyan
    Write-Host "$description" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
}

function Write-Step {
    param([string]$message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $message" -ForegroundColor Yellow
}

function Write-Success {
    param([string]$message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] ✓ $message" -ForegroundColor Green
}

function Write-Failure {
    param([string]$message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] ✗ $message" -ForegroundColor Red
}

function Invoke-VmScriptWrapper {
    param(
        [string]$vmName,
        [string]$scriptPath,
        [hashtable]$scriptArgs,
        [int]$timeout
    )
    
    $invokeScript = Join-Path (Split-Path -Parent $scriptDir) "common/invoke-vm-script.ps1"
    
    Write-Step "Executing $scriptPath on $vmName..."
    
    & $invokeScript `
        -vmName $vmName `
        -scriptPath $scriptPath `
        -scriptArgs $scriptArgs `
        -timeout $timeout
    
    if ($LASTEXITCODE -ne 0) {
        throw "Script execution failed on $vmName"
    }
}

function Get-MIMetadata {
    $getMetadataScript = Join-Path (Split-Path -Parent $scriptDir) "common/get-mi-metadata.ps1"
    
    if (-not (Test-Path $getMetadataScript)) {
        throw "Metadata script not found: $getMetadataScript"
    }
    
    return & $getMetadataScript
}

function Test-VMExists {
    param([string]$vmName, [string]$resourceGroup)
    
    $vm = az vm show `
        --name $vmName `
        --resource-group $resourceGroup `
        --query "name" `
        --output tsv 2>$null
    
    return ($LASTEXITCODE -eq 0 -and $vm -eq $vmName)
}

# ============================================================================
# Display Configuration
# ============================================================================

Write-Host ""
Write-Host "========================================" -ForegroundColor Magenta
Write-Host "Azure CleanRoom E2E Test - VM Edition" -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta
Write-Host "Configuration:" -ForegroundColor White
Write-Host "  Encryption Type:  $encryptionType" -ForegroundColor Gray
Write-Host "  Query Format:     $queryFormat" -ForegroundColor Gray
Write-Host "  API Mode:         $apiMode" -ForegroundColor Gray
Write-Host "  Location:         $($config.location)" -ForegroundColor Gray
Write-Host "  Northwind VM:     $($config.northwindVM)" -ForegroundColor Gray
Write-Host "  Woodgrove VM:     $($config.woodgroveVM)" -ForegroundColor Gray
Write-Host "========================================" -ForegroundColor Magenta
Write-Host ""

# ============================================================================
# Phase 0: Infrastructure Setup
# ============================================================================

if (-not $skipInfrastructure) {
    Write-PhaseHeader "Phase 0: Infrastructure Setup" "Ensuring VMs are provisioned and accessible"
    
    $parentDir = Split-Path -Parent $scriptDir
    
    # Check if VMs exist
    Write-Step "Checking if VMs exist..."
    $northwindExists = Test-VMExists -vmName $config.northwindVM -resourceGroup $config.vmResourceGroup
    $woodgroveExists = Test-VMExists -vmName $config.woodgroveVM -resourceGroup $config.vmResourceGroup
    
    if (-not $northwindExists -or -not $woodgroveExists) {
        Write-Step "VMs not found. Creating VMs..."
        $createVMScript = Join-Path $parentDir "00-create-test-vms.ps1"
        & $createVMScript
        
        if ($LASTEXITCODE -ne 0) {
            Write-Failure "Failed to create VMs"
            exit 1
        }
        Write-Success "VMs created successfully"
    } else {
        Write-Success "VMs already exist"
    }
    
    # Assign RBAC permissions
    Write-Step "Assigning RBAC permissions to managed identities..."
    $assignRBACScript = Join-Path $parentDir "00a-assign-mi-permissions.ps1"
    & $assignRBACScript
    
    if ($LASTEXITCODE -ne 0) {
        Write-Failure "Failed to assign RBAC permissions"
        exit 1
    }
    Write-Success "RBAC permissions assigned"
    
    Write-Success "Phase 0 complete - Infrastructure ready"
}

# ============================================================================
# Load MI Metadata
# ============================================================================

Write-Step "Loading managed identity metadata..."
$metadata = Get-MIMetadata

if (-not $metadata) {
    Write-Failure "Failed to load MI metadata"
    exit 1
}

Write-Host "  Northwind MI Principal ID: $($metadata.northwindPrincipalId)" -ForegroundColor Gray
Write-Host "  Woodgrove MI Principal ID: $($metadata.woodgrovePrincipalId)" -ForegroundColor Gray
Write-Success "Metadata loaded"

# ============================================================================
# Phase 1: Collaboration Setup
# ============================================================================

$collaborationName = ""
$resourceGroupName = ""

if ($useExistingCollaboration) {
    if (-not $existingCollaborationName -or -not $existingCollaborationRG) {
        Write-Failure "Must specify -existingCollaborationName and -existingCollaborationRG when using -useExistingCollaboration"
        exit 1
    }
    
    $collaborationName = $existingCollaborationName
    $resourceGroupName = $existingCollaborationRG
    Write-Step "Using existing collaboration: $collaborationName in $resourceGroupName"
} else {
    Write-PhaseHeader "Phase 1: Collaboration Setup" "Creating collaboration and adding collaborators"
    
    # Generate short collaboration name to avoid 80-char AKS resource group limit
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $collaborationName = "c$timestamp"
    $resourceGroupName = "cr-e2e-rg"
    
    Write-Step "Collaboration name: $collaborationName"
    Write-Step "Resource group: $resourceGroupName"
    
    try {
        Invoke-VmScriptWrapper `
            -vmName $config.woodgroveVM `
            -scriptPath "01-setup-collaboration.ps1" `
            -scriptArgs @{
                collaborationName = $collaborationName
                resourceGroupName = $resourceGroupName
                location = $config.location
                northwindMIPrincipalId = $metadata.northwindPrincipalId
                woodgroveMIPrincipalId = $metadata.woodgrovePrincipalId
                persona = "woodgrove"
                frontendEndpoint = $config.frontendEndpoint
            } `
            -timeout $config.timeouts.collaboration
        
        Write-Success "Phase 1 complete - Collaboration created: $collaborationName"
    } catch {
        Write-Failure "Phase 1 failed: $_"
        if ($cleanupOnFailure) {
            Write-Step "Cleanup on failure - skipping (resources may not exist)"
        }
        exit 1
    }
}

# Validate collaboration info is available for subsequent phases
if (-not $collaborationName -or -not $resourceGroupName) {
    Write-Failure "Collaboration name and resource group must be specified"
    exit 1
}

# ============================================================================
# Phase 2: Prepare Resources (Northwind)
# ============================================================================

Write-PhaseHeader "Phase 2: Prepare Resources (Northwind)" "Creating storage accounts, key vaults, managed identities"

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$northwindRG = "cr-e2e-northwind-$timestamp"

try {
    Invoke-VmScriptWrapper `
        -vmName $config.northwindVM `
        -scriptPath "04-prepare-resources.ps1" `
        -scriptArgs @{
            persona = "northwind"
            location = $config.location
            resourceGroupName = $northwindRG
        } `
        -timeout $config.timeouts.resourcePrep
    
    Write-Success "Phase 2 (Northwind) complete"
} catch {
    Write-Failure "Phase 2 (Northwind) failed: $_"
    if ($cleanupOnFailure) {
        # Cleanup implemented at end
    }
    exit 1
}

# ============================================================================
# Phase 2: Prepare Resources (Woodgrove)
# ============================================================================

Write-PhaseHeader "Phase 2: Prepare Resources (Woodgrove)" "Creating storage accounts, key vaults, managed identities"

$woodgroveRG = "cr-e2e-woodgrove-$timestamp"

try {
    Invoke-VmScriptWrapper `
        -vmName $config.woodgroveVM `
        -scriptPath "04-prepare-resources.ps1" `
        -scriptArgs @{
            persona = "woodgrove"
            location = $config.location
            resourceGroupName = $woodgroveRG
        } `
        -timeout $config.timeouts.resourcePrep
    
    Write-Success "Phase 2 (Woodgrove) complete"
} catch {
    Write-Failure "Phase 2 (Woodgrove) failed: $_"
    if ($cleanupOnFailure) {
        # Cleanup implemented at end
    }
    exit 1
}

# ============================================================================
# Phase 3: Prepare Data
# ============================================================================

Write-PhaseHeader "Phase 3: Prepare Data" "Generating and uploading sample data"

# Determine which data script to use based on encryption type
$dataScript = if ($encryptionType -eq "SSE") { "05-prepare-data-sse.ps1" } else { "05-prepare-data-cpk.ps1" }

# Northwind
try {
    Invoke-VmScriptWrapper `
        -vmName $config.northwindVM `
        -scriptPath $dataScript `
        -scriptArgs @{
            persona = "northwind"
            location = $config.location
            resourceGroupName = $northwindRG
        } `
        -timeout $config.timeouts.dataPrep
    
    Write-Success "Phase 3 (Northwind) complete"
} catch {
    Write-Failure "Phase 3 (Northwind) failed: $_"
    exit 1
}

# Woodgrove
try {
    Invoke-VmScriptWrapper `
        -vmName $config.woodgroveVM `
        -scriptPath $dataScript `
        -scriptArgs @{
            persona = "woodgrove"
            location = $config.location
            resourceGroupName = $woodgroveRG
        } `
        -timeout $config.timeouts.dataPrep
    
    Write-Success "Phase 3 (Woodgrove) complete"
} catch {
    Write-Failure "Phase 3 (Woodgrove) failed: $_"
    exit 1
}

# ============================================================================
# Phase 4: Setup Identity
# ============================================================================

Write-PhaseHeader "Phase 4: Setup Identity" "Configuring managed identities for CleanRoom access"

# Northwind
try {
    Invoke-VmScriptWrapper `
        -vmName $config.northwindVM `
        -scriptPath "06-setup-identity.ps1" `
        -scriptArgs @{
            persona = "northwind"
            location = $config.location
            resourceGroupName = $northwindRG
            collaborationName = $collaborationName
            collaborationResourceGroup = $resourceGroupName
            frontendEndpoint = $config.frontendEndpoint
        } `
        -timeout $config.timeouts.identity
    
    Write-Success "Phase 4 (Northwind) complete"
} catch {
    Write-Failure "Phase 4 (Northwind) failed: $_"
    exit 1
}

# Woodgrove
try {
    Invoke-VmScriptWrapper `
        -vmName $config.woodgroveVM `
        -scriptPath "06-setup-identity.ps1" `
        -scriptArgs @{
            persona = "woodgrove"
            location = $config.location
            resourceGroupName = $woodgroveRG
            collaborationName = $collaborationName
            collaborationResourceGroup = $resourceGroupName
            frontendEndpoint = $config.frontendEndpoint
        } `
        -timeout $config.timeouts.identity
    
    Write-Success "Phase 4 (Woodgrove) complete"
} catch {
    Write-Failure "Phase 4 (Woodgrove) failed: $_"
    exit 1
}

# ============================================================================
# Phase 5: Grant Access
# ============================================================================

Write-PhaseHeader "Phase 5: Grant Access" "Assigning RBAC permissions for storage access"

# Northwind
try {
    Invoke-VmScriptWrapper `
        -vmName $config.northwindVM `
        -scriptPath "07-grant-access.ps1" `
        -scriptArgs @{
            persona = "northwind"
            location = $config.location
            resourceGroupName = $northwindRG
            collaborationName = $collaborationName
            collaborationResourceGroup = $resourceGroupName
            frontendEndpoint = $config.frontendEndpoint
        } `
        -timeout $config.timeouts.access
    
    Write-Success "Phase 5 (Northwind) complete"
} catch {
    Write-Failure "Phase 5 (Northwind) failed: $_"
    exit 1
}

# Woodgrove
try {
    Invoke-VmScriptWrapper `
        -vmName $config.woodgroveVM `
        -scriptPath "07-grant-access.ps1" `
        -scriptArgs @{
            persona = "woodgrove"
            location = $config.location
            resourceGroupName = $woodgroveRG
            collaborationName = $collaborationName
            collaborationResourceGroup = $resourceGroupName
            frontendEndpoint = $config.frontendEndpoint
        } `
        -timeout $config.timeouts.access
    
    Write-Success "Phase 5 (Woodgrove) complete"
} catch {
    Write-Failure "Phase 5 (Woodgrove) failed: $_"
    exit 1
}

# ============================================================================
# Phase 6: Publish Datasets
# ============================================================================

Write-PhaseHeader "Phase 6: Publish Datasets" "Registering datasets with CleanRoom collaboration"

# Determine which dataset script to use based on encryption type
$datasetScript = if ($encryptionType -eq "SSE") { "08-publish-dataset-sse.ps1" } else { "08-publish-dataset-cpk.ps1" }

# Northwind
try {
    Invoke-VmScriptWrapper `
        -vmName $config.northwindVM `
        -scriptPath $datasetScript `
        -scriptArgs @{
            persona = "northwind"
            location = $config.location
            resourceGroupName = $northwindRG
            collaborationName = $collaborationName
            collaborationResourceGroup = $resourceGroupName
            frontendEndpoint = $config.frontendEndpoint
            ApiMode = $apiMode
        } `
        -timeout $config.timeouts.dataset
    
    Write-Success "Phase 6 (Northwind) complete"
} catch {
    Write-Failure "Phase 6 (Northwind) failed: $_"
    exit 1
}

# Woodgrove
try {
    Invoke-VmScriptWrapper `
        -vmName $config.woodgroveVM `
        -scriptPath $datasetScript `
        -scriptArgs @{
            persona = "woodgrove"
            location = $config.location
            resourceGroupName = $woodgroveRG
            collaborationName = $collaborationName
            collaborationResourceGroup = $resourceGroupName
            frontendEndpoint = $config.frontendEndpoint
            ApiMode = $apiMode
        } `
        -timeout $config.timeouts.dataset
    
    Write-Success "Phase 6 (Woodgrove) complete"
} catch {
    Write-Failure "Phase 6 (Woodgrove) failed: $_"
    exit 1
}

# ============================================================================
# Phase 7: Publish Query
# ============================================================================

Write-PhaseHeader "Phase 7: Publish Query" "Publishing analytics query to collaboration"

# Determine which query script to use
$queryScript = if ($queryFormat -eq "json") { "09-publish-query-json.ps1" } else { "09-publish-query.ps1" }

try {
    Invoke-VmScriptWrapper `
        -vmName $config.woodgroveVM `
        -scriptPath $queryScript `
        -scriptArgs @{
            persona = "woodgrove"
            collaborationName = $collaborationName
            resourceGroupName = $resourceGroupName
            frontendEndpoint = $config.frontendEndpoint
            ApiMode = $apiMode
        } `
        -timeout $config.timeouts.query
    
    Write-Success "Phase 7 complete - Query published"
} catch {
    Write-Failure "Phase 7 failed: $_"
    exit 1
}

# ============================================================================
# Phase 8: Vote on Query
# ============================================================================

Write-PhaseHeader "Phase 8: Vote on Query" "Both collaborators voting to approve query"

# Northwind vote
try {
    Invoke-VmScriptWrapper `
        -vmName $config.northwindVM `
        -scriptPath "10-vote-query.ps1" `
        -scriptArgs @{
            persona = "northwind"
            collaborationName = $collaborationName
            resourceGroupName = $resourceGroupName
            frontendEndpoint = $config.frontendEndpoint
            ApiMode = $apiMode
        } `
        -timeout $config.timeouts.vote
    
    Write-Success "Phase 8 (Northwind vote) complete"
} catch {
    Write-Failure "Phase 8 (Northwind vote) failed: $_"
    exit 1
}

# Woodgrove vote
try {
    Invoke-VmScriptWrapper `
        -vmName $config.woodgroveVM `
        -scriptPath "10-vote-query.ps1" `
        -scriptArgs @{
            persona = "woodgrove"
            collaborationName = $collaborationName
            resourceGroupName = $resourceGroupName
            frontendEndpoint = $config.frontendEndpoint
            ApiMode = $apiMode
        } `
        -timeout $config.timeouts.vote
    
    Write-Success "Phase 8 (Woodgrove vote) complete - Query approved by all collaborators"
} catch {
    Write-Failure "Phase 8 (Woodgrove vote) failed: $_"
    exit 1
}

# ============================================================================
# Phase 9: Execute Query
# ============================================================================

Write-PhaseHeader "Phase 9: Execute Query" "Running approved query (this may take 5-60 minutes)"

try {
    Invoke-VmScriptWrapper `
        -vmName $config.woodgroveVM `
        -scriptPath "11-run-query.ps1" `
        -scriptArgs @{
            persona = "woodgrove"
            collaborationName = $collaborationName
            resourceGroupName = $resourceGroupName
            frontendEndpoint = $config.frontendEndpoint
            ApiMode = $apiMode
        } `
        -timeout $config.timeouts.execute
    
    Write-Success "Phase 9 complete - Query execution finished"
} catch {
    Write-Failure "Phase 9 failed: $_"
    exit 1
}

# ============================================================================
# Phase 10: View Results
# ============================================================================

Write-PhaseHeader "Phase 10: View Results" "Retrieving and displaying query results"

try {
    Invoke-VmScriptWrapper `
        -vmName $config.woodgroveVM `
        -scriptPath "12-view-results.ps1" `
        -scriptArgs @{
            persona = "woodgrove"
            collaborationName = $collaborationName
            resourceGroupName = $resourceGroupName
            frontendEndpoint = $config.frontendEndpoint
            ApiMode = $apiMode
        } `
        -timeout $config.timeouts.results
    
    Write-Success "Phase 10 complete - Results retrieved"
} catch {
    Write-Failure "Phase 10 failed: $_"
    exit 1
}

# ============================================================================
# Final Summary
# ============================================================================

$scriptEndTime = Get-Date
$duration = $scriptEndTime - $scriptStartTime
$minutes = [Math]::Floor($duration.TotalMinutes)
$seconds = $duration.Seconds

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "E2E TEST COMPLETE" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Collaboration:     $collaborationName" -ForegroundColor Green
Write-Host "  Resource Group:    $resourceGroupName" -ForegroundColor Green
Write-Host "  Northwind RG:      $northwindRG" -ForegroundColor Green
Write-Host "  Woodgrove RG:      $woodgroveRG" -ForegroundColor Green
Write-Host "  Encryption:        $encryptionType" -ForegroundColor Green
Write-Host "  Query Format:      $queryFormat" -ForegroundColor Green
Write-Host "  API Mode:          $apiMode" -ForegroundColor Green
Write-Host "  Duration:          $minutes min $seconds sec" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# ============================================================================
# Cleanup (if requested)
# ============================================================================

if ($cleanupOnSuccess) {
    Write-PhaseHeader "Cleanup: Removing Test Resources" "Deleting collaboration and resource groups"
    
    Write-Step "Deleting collaboration: $collaborationName"
    az managedcleanroom collaboration delete `
        --collaboration-name $collaborationName `
        --resource-group $resourceGroupName `
        --yes `
        --no-wait 2>&1 | Out-Null
    
    Write-Step "Deleting resource group: $resourceGroupName"
    az group delete `
        --name $resourceGroupName `
        --yes `
        --no-wait 2>&1 | Out-Null
    
    Write-Step "Deleting Northwind resource group: $northwindRG"
    az group delete `
        --name $northwindRG `
        --yes `
        --no-wait 2>&1 | Out-Null
    
    Write-Step "Deleting Woodgrove resource group: $woodgroveRG"
    az group delete `
        --name $woodgroveRG `
        --yes `
        --no-wait 2>&1 | Out-Null
    
    Write-Success "Cleanup initiated (running in background)"
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Review results in the collaboration" -ForegroundColor Yellow
Write-Host "  2. Manually test via Azure Portal: https://portal.azure.com" -ForegroundColor Yellow
if (-not $cleanupOnSuccess) {
    Write-Host "  3. Delete resources when done:" -ForegroundColor Yellow
    Write-Host "     az group delete --name $resourceGroupName --yes --no-wait" -ForegroundColor Gray
    Write-Host "     az group delete --name $northwindRG --yes --no-wait" -ForegroundColor Gray
    Write-Host "     az group delete --name $woodgroveRG --yes --no-wait" -ForegroundColor Gray
}
Write-Host ""
