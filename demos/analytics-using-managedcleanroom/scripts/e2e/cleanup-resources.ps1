<#
.SYNOPSIS
    Deletes all E2E test resources created during testing.

.DESCRIPTION
    Run by: Test automation or admin.
    Deletes Azure resource groups and optionally a managed cleanroom collaboration
    created during end-to-end testing. Resource groups are deleted in parallel
    for faster cleanup.

.PARAMETER resourceGroups
    Array of resource group names to delete.
    Example: @("rg-northwind-test", "rg-woodgrove-test", "rg-contoso-test")

.PARAMETER collaborationResourceGroup
    Resource group containing the collaboration (if different from resourceGroups).

.PARAMETER collaborationName
    Name of the collaboration to delete.

.PARAMETER subscription
    Optional Azure subscription name or ID.

.PARAMETER force
    Skip confirmation prompts and delete immediately.

.EXAMPLE
    ./cleanup-resources.ps1 -resourceGroups @("rg-test1", "rg-test2") -force

.EXAMPLE
    ./cleanup-resources.ps1 `
        -resourceGroups @("rg-northwind", "rg-woodgrove") `
        -collaborationResourceGroup "rg-owner" `
        -collaborationName "test-collab" `
        -force
#>
param(
    [Parameter(Mandatory)]
    [string[]]$resourceGroups,

    [string]$collaborationResourceGroup,

    [string]$collaborationName,

    [string]$subscription,

    [switch]$force
)

$ErrorActionPreference = 'Stop'

# Helper to log and execute az commands.
function Invoke-AzCommand {
    param([string[]]$Arguments)
    $cmdLine = "az $($Arguments -join ' ')"
    Write-Host "[CMD] $cmdLine" -ForegroundColor DarkGray
    & az @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
}

# Runs an az command, returning $null instead of throwing if it fails.
function Invoke-AzSafe {
    param([string[]]$Arguments)
    $cmdLine = "az $($Arguments -join ' ')"
    Write-Host "[CMD] $cmdLine" -ForegroundColor DarkGray
    $PSNativeCommandUseErrorActionPreference = $false
    $result = & az @Arguments 2>$null
    if ($LASTEXITCODE -eq 0) { return $result }
    return $null
}

# Set subscription if specified.
if ($subscription) {
    Write-Host "Setting subscription to '$subscription'..." -ForegroundColor Yellow
    Invoke-AzCommand @("account", "set", "--subscription", $subscription)
}

# Confirmation prompt unless --force is specified.
if (-not $force) {
    Write-Host "`n========================================" -ForegroundColor Yellow
    Write-Host "WARNING: This will DELETE the following resources:" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
    
    if ($collaborationName -and $collaborationResourceGroup) {
        Write-Host "  Collaboration: $collaborationName (in $collaborationResourceGroup)" -ForegroundColor Red
    }
    
    Write-Host "  Resource Groups:" -ForegroundColor Red
    foreach ($rg in $resourceGroups) {
        Write-Host "    - $rg" -ForegroundColor Red
    }
    
    Write-Host "`nThis action CANNOT be undone." -ForegroundColor Yellow
    $confirmation = Read-Host "`nType 'yes' to continue"
    
    if ($confirmation -ne 'yes') {
        Write-Host "Cleanup cancelled." -ForegroundColor Yellow
        exit 0
    }
}

# Track deletion results.
$deletionResults = @{
    Success = @()
    Failed = @()
    Skipped = @()
}

# Step 1: Delete collaboration first (if specified).
if ($collaborationName -and $collaborationResourceGroup) {
    Write-Host "`n=== Step 1: Deleting collaboration '$collaborationName' ===" -ForegroundColor Cyan
    
    # Check if collaboration exists.
    $collab = Invoke-AzSafe @("managedcleanroom", "collaboration", "show",
        "--collaboration-name", $collaborationName,
        "--resource-group", $collaborationResourceGroup)
    
    if ($collab) {
        Write-Host "Deleting collaboration '$collaborationName' from resource group '$collaborationResourceGroup'..." -ForegroundColor Yellow
        
        try {
            Invoke-AzCommand @("managedcleanroom", "collaboration", "delete",
                "--collaboration-name", $collaborationName,
                "--resource-group", $collaborationResourceGroup,
                "--yes")
            Write-Host "Collaboration '$collaborationName' deleted successfully." -ForegroundColor Green
            $deletionResults.Success += "Collaboration: $collaborationName"
        }
        catch {
            Write-Host "ERROR: Failed to delete collaboration '$collaborationName': $_" -ForegroundColor Red
            $deletionResults.Failed += "Collaboration: $collaborationName ($_)"
        }
    }
    else {
        Write-Host "Collaboration '$collaborationName' not found (may already be deleted)." -ForegroundColor Yellow
        $deletionResults.Skipped += "Collaboration: $collaborationName (not found)"
    }
}
else {
    Write-Host "`n=== Step 1: Skipping collaboration deletion (not specified) ===" -ForegroundColor Cyan
}

# Step 2: Delete resource groups in parallel.
Write-Host "`n=== Step 2: Deleting resource groups in parallel ===" -ForegroundColor Cyan

if ($resourceGroups.Count -eq 0) {
    Write-Host "No resource groups specified for deletion." -ForegroundColor Yellow
}
else {
    Write-Host "Starting deletion of $($resourceGroups.Count) resource group(s)..." -ForegroundColor Yellow
    
    # Create jobs for parallel deletion.
    $jobs = @()
    
    foreach ($rg in $resourceGroups) {
        Write-Host "Initiating deletion for resource group '$rg'..." -ForegroundColor Yellow
        
        # Start a background job for each resource group deletion.
        $job = Start-Job -ScriptBlock {
            param($rgName)
            
            # Check if resource group exists.
            $exists = az group exists --name $rgName 2>$null
            if ($exists -eq "false") {
                return @{
                    ResourceGroup = $rgName
                    Status = "NotFound"
                    Message = "Resource group not found (may already be deleted)"
                }
            }
            
            # Delete the resource group.
            az group delete --name $rgName --yes --no-wait 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                return @{
                    ResourceGroup = $rgName
                    Status = "Initiated"
                    Message = "Deletion initiated successfully"
                }
            }
            else {
                return @{
                    ResourceGroup = $rgName
                    Status = "Failed"
                    Message = "Failed to initiate deletion"
                }
            }
        } -ArgumentList $rg
        
        $jobs += @{
            Job = $job
            ResourceGroup = $rg
        }
    }
    
    # Wait for all jobs to complete.
    Write-Host "`nWaiting for deletion jobs to complete..." -ForegroundColor Yellow
    $jobs | ForEach-Object { $_.Job } | Wait-Job | Out-Null
    
    # Collect results from jobs.
    foreach ($jobInfo in $jobs) {
        $result = Receive-Job -Job $jobInfo.Job
        Remove-Job -Job $jobInfo.Job
        
        $rg = $jobInfo.ResourceGroup
        
        if ($result.Status -eq "Initiated") {
            Write-Host "Resource group '$rg': Deletion initiated." -ForegroundColor Green
            $deletionResults.Success += "Resource Group: $rg (deletion initiated)"
        }
        elseif ($result.Status -eq "NotFound") {
            Write-Host "Resource group '$rg': Not found (may already be deleted)." -ForegroundColor Yellow
            $deletionResults.Skipped += "Resource Group: $rg (not found)"
        }
        else {
            Write-Host "Resource group '$rg': Failed - $($result.Message)" -ForegroundColor Red
            $deletionResults.Failed += "Resource Group: $rg ($($result.Message))"
        }
    }
}

# Step 3: Wait for resource group deletions to complete.
Write-Host "`n=== Step 3: Verifying resource group deletions ===" -ForegroundColor Cyan

$initiatedDeletions = $resourceGroups | Where-Object {
    $rg = $_
    $deletionResults.Success -match "Resource Group: $rg"
}

if ($initiatedDeletions.Count -gt 0) {
    Write-Host "Waiting for resource group deletions to complete (this may take several minutes)..." -ForegroundColor Yellow
    
    $maxWaitSeconds = 600  # 10 minutes max wait
    $checkIntervalSeconds = 15
    $elapsedSeconds = 0
    $stillDeleting = $initiatedDeletions
    
    while ($stillDeleting.Count -gt 0 -and $elapsedSeconds -lt $maxWaitSeconds) {
        Start-Sleep -Seconds $checkIntervalSeconds
        $elapsedSeconds += $checkIntervalSeconds
        
        $newStillDeleting = @()
        foreach ($rg in $stillDeleting) {
            $exists = Invoke-AzSafe @("group", "exists", "--name", $rg)
            if ($exists -eq "true") {
                $newStillDeleting += $rg
            }
            else {
                Write-Host "Resource group '$rg' deletion completed." -ForegroundColor Green
            }
        }
        
        $stillDeleting = $newStillDeleting
        
        if ($stillDeleting.Count -gt 0) {
            Write-Host "  Still deleting: $($stillDeleting -join ', ') (elapsed: ${elapsedSeconds}s)" -ForegroundColor DarkGray
        }
    }
    
    if ($stillDeleting.Count -gt 0) {
        Write-Host "`nWARNING: The following resource groups are still being deleted:" -ForegroundColor Yellow
        foreach ($rg in $stillDeleting) {
            Write-Host "  - $rg" -ForegroundColor Yellow
        }
        Write-Host "Check Azure Portal or run 'az group exists --name <rg-name>' to verify completion." -ForegroundColor Yellow
    }
    else {
        Write-Host "`nAll resource group deletions completed successfully." -ForegroundColor Green
    }
}

# Step 4: Display summary.
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "CLEANUP SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ($deletionResults.Success.Count -gt 0) {
    Write-Host "`nSuccessfully deleted:" -ForegroundColor Green
    foreach ($item in $deletionResults.Success) {
        Write-Host "  ✓ $item" -ForegroundColor Green
    }
}

if ($deletionResults.Skipped.Count -gt 0) {
    Write-Host "`nSkipped (not found):" -ForegroundColor Yellow
    foreach ($item in $deletionResults.Skipped) {
        Write-Host "  - $item" -ForegroundColor Yellow
    }
}

if ($deletionResults.Failed.Count -gt 0) {
    Write-Host "`nFailed to delete:" -ForegroundColor Red
    foreach ($item in $deletionResults.Failed) {
        Write-Host "  ✗ $item" -ForegroundColor Red
    }
    Write-Host "`nCleanup completed with errors." -ForegroundColor Red
    exit 1
}
else {
    Write-Host "`nCleanup completed successfully." -ForegroundColor Green
    exit 0
}
