#!/usr/bin/env pwsh

<#
.SYNOPSIS
    Run complete end-to-end test for Azure Managed CleanRoom with app-based authentication.

.DESCRIPTION
    This script orchestrates a full E2E test of the CleanRoom analytics demo, including:
    - Collaboration creation
    - Resource provisioning
    - App permission assignment
    - Data preparation
    - Identity setup
    - Dataset publishing
    - Query execution
    - Results validation
    - Optional cleanup

.PARAMETER testScenario
    Test scenario to run: "sse-inline" or "sse-json"

.PARAMETER appKeyVaultName
    Key Vault containing app credentials for northwind and woodgrove

.PARAMETER subscription
    Optional Azure subscription ID to use

.PARAMETER location
    Azure region for resources (default: westus)

.PARAMETER frontendEndpoint
    Frontend URL (if not using default)

.PARAMETER outDir
    Output directory for generated files (default: ./generated)

.PARAMETER cleanupOnSuccess
    Delete all resources after successful test completion

.PARAMETER cleanupOnFailure
    Delete all resources even if test fails

.EXAMPLE
    ./run-e2e-test.ps1 -testScenario sse-inline -appKeyVaultName my-kv

.EXAMPLE
    ./run-e2e-test.ps1 -testScenario sse-json -appKeyVaultName my-kv -cleanupOnSuccess
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet("sse-inline", "sse-json")]
    [string]$testScenario,

    [Parameter(Mandatory)]
    [string]$appKeyVaultName,

    [string]$subscription,

    [string]$location = "westus",

    [string]$frontendEndpoint,

    [string]$outDir = "./generated",

    [ValidateSet("rest", "cli")]
    [string]$apiMode = "rest",

    [switch]$cleanupOnSuccess,

    [switch]$cleanupOnFailure
)

Set-StrictMode -Version 3.0
$ErrorActionPreference = "Stop"

# ============================================================================
# Global Variables
# ============================================================================

$script:testStartTime = Get-Date
$script:timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$script:stepsCompleted = 0
$script:totalSteps = 12
$script:testFailed = $false
$script:failedStep = ""
$script:resourcesCreated = @()

# Determine query mode and directory
$queryMode = if ($testScenario -eq "sse-inline") { "inline" } else { "json" }
$queryDir = if ($testScenario -eq "sse-inline") { "query1" } else { "query2" }

# Resource names
$northwindRgName = "cleanroom-e2e-northwind-$timestamp"
$woodgroveRgName = "cleanroom-e2e-woodgrove-$timestamp"
$collaborationName = "e2e-collab-$timestamp"

# Output paths
$logFile = Join-Path $outDir "e2e-test-$testScenario-$timestamp.log"
$collaborationIdFile = Join-Path $outDir "collaboration-id.txt"
$frontendEndpointFile = Join-Path $outDir "frontend-endpoint.txt"

# ============================================================================
# Helper Functions
# ============================================================================

function Write-ColorText {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    
    # Write to console with color
    switch ($Level) {
        "ERROR" { Write-ColorText $Message "Red" }
        "SUCCESS" { Write-ColorText $Message "Green" }
        "WARN" { Write-ColorText $Message "Yellow" }
        default { Write-Host $Message }
    }
    
    # Write to log file
    Add-Content -Path $logFile -Value $logMessage
}

function Write-StepHeader {
    param([string]$StepNumber, [string]$Description)
    
    $script:stepsCompleted++
    $percentComplete = [int](($script:stepsCompleted / $script:totalSteps) * 100)
    $elapsed = (Get-Date) - $script:testStartTime
    $elapsedStr = "{0:hh\:mm\:ss}" -f $elapsed
    
    $header = @"

================================================================================
STEP $StepNumber - $Description
Progress: $script:stepsCompleted / $script:totalSteps ($percentComplete%)
Elapsed Time: $elapsedStr
================================================================================
"@
    Write-Log $header "INFO"
}

function Write-Progress {
    param([string]$Message)
    
    $elapsed = (Get-Date) - $script:testStartTime
    $elapsedStr = "{0:hh\:mm\:ss}" -f $elapsed
    Write-Log "[PROGRESS] $Message (elapsed: $elapsedStr)" "INFO"
}

function Invoke-ScriptStep {
    param(
        [string]$ScriptPath,
        [hashtable]$Parameters,
        [string]$Description
    )
    
    $stepStartTime = Get-Date
    Write-Log "========================================" "INFO"
    Write-Log "Executing: $ScriptPath" "INFO"
    Write-Log "Description: $Description" "INFO"
    Write-Log "Parameters:" "INFO"
    foreach ($key in $Parameters.Keys) {
        Write-Log "  $key = $($Parameters[$key])" "INFO"
    }
    Write-Log "========================================" "INFO"
    
    try {
        $result = & $ScriptPath @Parameters 2>&1 | Tee-Object -Append -FilePath $logFile
        
        if ($LASTEXITCODE -ne 0 -and $null -ne $LASTEXITCODE) {
            throw "Script exited with code $LASTEXITCODE"
        }
        
        $stepDuration = (Get-Date) - $stepStartTime
        $durationStr = "{0:mm\:ss}" -f $stepDuration
        Write-Log "$Description - COMPLETED in $durationStr" "SUCCESS"
        return $result
    }
    catch {
        $stepDuration = (Get-Date) - $stepStartTime
        $durationStr = "{0:mm\:ss}" -f $stepDuration
        Write-Log "$Description - FAILED after $durationStr" "ERROR"
        Write-Log "Error: $_" "ERROR"
        Write-Log "Stack trace: $($_.ScriptStackTrace)" "ERROR"
        
        # Collect diagnostics
        Write-Log "`n=== DIAGNOSTICS ===" "ERROR"
        Write-Log "Failed script: $ScriptPath" "ERROR"
        Write-Log "Working directory: $(Get-Location)" "ERROR"
        Write-Log "PowerShell version: $($PSVersionTable.PSVersion)" "ERROR"
        
        $script:testFailed = $true
        $script:failedStep = $Description
        throw
    }
}

function Start-ParallelJobs {
    param(
        [hashtable[]]$Jobs,
        [string]$Description
    )
    
    Write-Log "Starting parallel execution: $Description" "INFO"
    
    $jobObjects = @()
    
    foreach ($job in $Jobs) {
        $jobName = $job.Name
        $scriptPath = $job.ScriptPath
        $params = $job.Parameters
        
        Write-Log "Starting job: $jobName" "INFO"
        
        $scriptBlock = {
            param($script, $parameters)
            & $script @parameters
        }
        
        $jobObj = Start-Job -Name $jobName -ScriptBlock $scriptBlock -ArgumentList $scriptPath, $params
        $jobObjects += $jobObj
    }
    
    # Wait for all jobs and collect results
    $results = @{}
    $allSucceeded = $true
    
    foreach ($jobObj in $jobObjects) {
        Write-Log "Waiting for job: $($jobObj.Name)..." "INFO"
        $jobResult = Wait-Job $jobObj | Receive-Job
        
        # Log job output
        $jobOutput = $jobResult | Out-String
        Add-Content -Path $logFile -Value "`n=== Job Output: $($jobObj.Name) ===`n$jobOutput"
        
        if ($jobObj.State -eq "Failed") {
            Write-Log "Job failed: $($jobObj.Name)" "ERROR"
            $allSucceeded = $false
        }
        else {
            Write-Log "Job completed: $($jobObj.Name)" "SUCCESS"
        }
        
        $results[$jobObj.Name] = $jobResult
        Remove-Job $jobObj
    }
    
    if (-not $allSucceeded) {
        throw "One or more parallel jobs failed"
    }
    
    return $results
}

function Invoke-Cleanup {
    param([bool]$TestSucceeded)
    
    $shouldCleanup = ($TestSucceeded -and $cleanupOnSuccess) -or (-not $TestSucceeded -and $cleanupOnFailure)
    
    if (-not $shouldCleanup) {
        Write-Log "Skipping cleanup (cleanupOnSuccess=$cleanupOnSuccess, cleanupOnFailure=$cleanupOnFailure)" "INFO"
        return
    }
    
    Write-Log "`nStarting cleanup..." "INFO"
    
    try {
        $cleanupParams = @{
            resourceGroups = @($northwindRgName, $woodgroveRgName)
            collaborationName = $collaborationName
            appKeyVaultName = $appKeyVaultName
        }
        
        if ($subscription) {
            $cleanupParams.subscription = $subscription
        }
        
        & "$PSScriptRoot/cleanup-resources.ps1" @cleanupParams 2>&1 | Tee-Object -Append -FilePath $logFile
        Write-Log "Cleanup completed successfully" "SUCCESS"
    }
    catch {
        Write-Log "Cleanup failed: $_" "ERROR"
    }
}

function Show-TestSummary {
    param([bool]$TestSucceeded)
    
    $duration = (Get-Date) - $script:testStartTime
    $durationFormatted = "{0:hh\:mm\:ss}" -f $duration
    
    $summary = @"

================================================================================
E2E TEST SUMMARY
================================================================================
Test Scenario    : $testScenario
Status           : $(if ($TestSucceeded) { "SUCCESS" } else { "FAILED" })
Duration         : $durationFormatted
Steps Completed  : $script:stepsCompleted / $script:totalSteps
Failed Step      : $(if ($script:failedStep) { $script:failedStep } else { "N/A" })

Resources Created:
  - Northwind RG : $northwindRgName
  - Woodgrove RG : $woodgroveRgName
  - Collaboration: $collaborationName
  - Location     : $location

Output Files:
  - Log File     : $logFile
  - Output Dir   : $outDir

Cleanup Status   : $(if ($cleanupOnSuccess -or $cleanupOnFailure) { "Enabled" } else { "Disabled" })
================================================================================
"@

    if ($TestSucceeded) {
        Write-Log $summary "SUCCESS"
    }
    else {
        Write-Log $summary "ERROR"
    }
}

# ============================================================================
# Main Test Flow
# ============================================================================

try {
    # Initialize
    Write-ColorText "`n=== Azure Managed CleanRoom E2E Test ===" "Cyan"
    Write-ColorText "Test Scenario: $testScenario" "Cyan"
    Write-ColorText "Timestamp: $timestamp`n" "Cyan"
    
    # Create output directory
    if (-not (Test-Path $outDir)) {
        New-Item -ItemType Directory -Path $outDir -Force | Out-Null
    }
    
    # Initialize log file
    Write-Log "E2E Test Started" "INFO"
    Write-Log "Configuration: testScenario=$testScenario, location=$location, outDir=$outDir" "INFO"
    
    $script:resourcesCreated = @($northwindRgName, $woodgroveRgName, $collaborationName)
    
    # ========================================================================
    # STEP 01 - Woodgrove creates collaboration and adds collaborators
    # ========================================================================
    Write-StepHeader "01" "Woodgrove creates collaboration and adds collaborators"
    
    $setupParams = @{
        persona = "woodgrove"
        collaborationName = $collaborationName
        resourceGroupName = $woodgroveRgName
        location = $location
        appKeyVaultName = $appKeyVaultName
        outDir = $outDir
    }
    
    if ($subscription) { $setupParams.subscription = $subscription }
    if ($frontendEndpoint) { $setupParams.frontendEndpoint = $frontendEndpoint }
    
    $setupResult = Invoke-ScriptStep `
        -ScriptPath "$PSScriptRoot/../01-setup-collaboration.ps1" `
        -Parameters $setupParams `
        -Description "Collaboration creation and collaborator addition"
    
    # Extract collaboration ID and frontend endpoint
    if (Test-Path $collaborationIdFile) {
        $collaborationId = Get-Content $collaborationIdFile -Raw | ForEach-Object { $_.Trim() }
        Write-Log "Collaboration ID: $collaborationId" "INFO"
    }
    else {
        throw "Collaboration ID file not found: $collaborationIdFile"
    }
    
    if (Test-Path $frontendEndpointFile) {
        $frontendEndpointDiscovered = Get-Content $frontendEndpointFile -Raw | ForEach-Object { $_.Trim() }
        Write-Log "Frontend Endpoint: $frontendEndpointDiscovered" "INFO"
        
        # Use discovered endpoint if not provided
        if (-not $frontendEndpoint) {
            $frontendEndpoint = $frontendEndpointDiscovered
        }
    }
    
    # ========================================================================
    # STEP 00a - Assign app permissions (AFTER step 01 creates woodgrove RG)
    # ========================================================================
    Write-StepHeader "00a" "Assign app permissions for both personas"
    
    # First, prepare resources for both personas (creates RGs and resources)
    Write-Log "Creating resource groups and initial resources..." "INFO"
    
    $prepareJobs = @(
        @{
            Name = "prepare-northwind"
            ScriptPath = "$PSScriptRoot/../04-prepare-resources.ps1"
            Parameters = @{
                persona = "northwind"
                resourceGroupName = $northwindRgName
                location = $location
                outDir = $outDir
                appKeyVaultName = $appKeyVaultName
            }
        },
        @{
            Name = "prepare-woodgrove"
            ScriptPath = "$PSScriptRoot/../04-prepare-resources.ps1"
            Parameters = @{
                persona = "woodgrove"
                resourceGroupName = $woodgroveRgName
                location = $location
                outDir = $outDir
                appKeyVaultName = $appKeyVaultName
            }
        }
    )
    
    if ($subscription) {
        $prepareJobs[0].Parameters.subscription = $subscription
        $prepareJobs[1].Parameters.subscription = $subscription
    }
    
    Start-ParallelJobs -Jobs $prepareJobs -Description "Resource group and resource preparation"
    
    # Now assign permissions to the apps
    Write-Log "Assigning app permissions..." "INFO"
    
    $permissionJobs = @(
        @{
            Name = "permissions-northwind"
            ScriptPath = "$PSScriptRoot/../00a-assign-app-permissions.ps1"
            Parameters = @{
                persona = "northwind"
                resourceGroupName = $northwindRgName
                appKeyVaultName = $appKeyVaultName
                outDir = $outDir
            }
        },
        @{
            Name = "permissions-woodgrove"
            ScriptPath = "$PSScriptRoot/../00a-assign-app-permissions.ps1"
            Parameters = @{
                persona = "woodgrove"
                resourceGroupName = $woodgroveRgName
                appKeyVaultName = $appKeyVaultName
                outDir = $outDir
            }
        }
    )
    
    if ($subscription) {
        $permissionJobs[0].Parameters.subscription = $subscription
        $permissionJobs[1].Parameters.subscription = $subscription
    }
    
    Start-ParallelJobs -Jobs $permissionJobs -Description "App permission assignment"
    
    # ========================================================================
    # STEPS 05-08 - Both personas in PARALLEL
    # ========================================================================
    Write-StepHeader "05-08" "Parallel data and identity setup for both personas"
    
    # Note: Step 04 already completed during permission assignment above
    Write-Log "Resource preparation (step 04) already completed" "INFO"
    
    # Step 05 - Prepare data (SSE only for now)
    Write-Log "Step 05 - Preparing data..." "INFO"
    $dataJobs = @(
        @{
            Name = "data-northwind"
            ScriptPath = "$PSScriptRoot/../05-prepare-data.ps1"
            Parameters = @{
                persona = "northwind"
                resourceGroupName = $northwindRgName
                encryption = "sse"
                outDir = $outDir
            }
        },
        @{
            Name = "data-woodgrove"
            ScriptPath = "$PSScriptRoot/../05-prepare-data.ps1"
            Parameters = @{
                persona = "woodgrove"
                resourceGroupName = $woodgroveRgName
                encryption = "sse"
                outDir = $outDir
            }
        }
    )
    
    if ($subscription) {
        $dataJobs[0].Parameters.subscription = $subscription
        $dataJobs[1].Parameters.subscription = $subscription
    }
    
    Start-ParallelJobs -Jobs $dataJobs -Description "Data preparation"
    
    # Step 06 - Setup identity
    Write-Log "Step 06 - Setting up identities..." "INFO"
    $identityJobs = @(
        @{
            Name = "identity-northwind"
            ScriptPath = "$PSScriptRoot/../06-setup-identity.ps1"
            Parameters = @{
                persona = "northwind"
                resourceGroupName = $northwindRgName
                collaborationId = $collaborationId
                appKeyVaultName = $appKeyVaultName
                frontendEndpoint = $frontendEndpoint
                outDir = $outDir
            }
        },
        @{
            Name = "identity-woodgrove"
            ScriptPath = "$PSScriptRoot/../06-setup-identity.ps1"
            Parameters = @{
                persona = "woodgrove"
                resourceGroupName = $woodgroveRgName
                collaborationId = $collaborationId
                appKeyVaultName = $appKeyVaultName
                frontendEndpoint = $frontendEndpoint
                outDir = $outDir
            }
        }
    )
    
    if ($subscription) {
        $identityJobs[0].Parameters.subscription = $subscription
        $identityJobs[1].Parameters.subscription = $subscription
    }
    
    Start-ParallelJobs -Jobs $identityJobs -Description "Identity setup"
    
    # Step 07 - Grant access
    Write-Log "Step 07 - Granting access..." "INFO"
    $accessJobs = @(
        @{
            Name = "access-northwind"
            ScriptPath = "$PSScriptRoot/../07-grant-access.ps1"
            Parameters = @{
                persona = "northwind"
                resourceGroupName = $northwindRgName
                encryption = "sse"
                outDir = $outDir
            }
        },
        @{
            Name = "access-woodgrove"
            ScriptPath = "$PSScriptRoot/../07-grant-access.ps1"
            Parameters = @{
                persona = "woodgrove"
                resourceGroupName = $woodgroveRgName
                encryption = "sse"
                outDir = $outDir
            }
        }
    )
    
    if ($subscription) {
        $accessJobs[0].Parameters.subscription = $subscription
        $accessJobs[1].Parameters.subscription = $subscription
    }
    
    Start-ParallelJobs -Jobs $accessJobs -Description "Access grant"
    
    # Step 08 - Publish dataset (SSE only)
    Write-Log "Step 08 - Publishing datasets..." "INFO"
    $publishJobs = @(
        @{
            Name = "publish-northwind"
            ScriptPath = "$PSScriptRoot/../08-publish-dataset.ps1"
            Parameters = @{
                persona = "northwind"
                collaborationId = $collaborationId
                encryption = "sse"
                appKeyVaultName = $appKeyVaultName
                frontendEndpoint = $frontendEndpoint
                outDir = $outDir
                ApiMode = $apiMode
            }
        },
        @{
            Name = "publish-woodgrove"
            ScriptPath = "$PSScriptRoot/../08-publish-dataset.ps1"
            Parameters = @{
                persona = "woodgrove"
                collaborationId = $collaborationId
                encryption = "sse"
                appKeyVaultName = $appKeyVaultName
                frontendEndpoint = $frontendEndpoint
                outDir = $outDir
                ApiMode = $apiMode
            }
        }
    )
    
    if ($subscription) {
        $publishJobs[0].Parameters.subscription = $subscription
        $publishJobs[1].Parameters.subscription = $subscription
    }
    
    Start-ParallelJobs -Jobs $publishJobs -Description "Dataset publishing"
    
    # ========================================================================
    # STEP 09 - Woodgrove publishes query
    # ========================================================================
    Write-StepHeader "09" "Woodgrove publishes query ($queryMode mode)"
    
    if ($testScenario -eq "sse-inline") {
        $queryParams = @{
            persona = "woodgrove"
            collaborationId = $collaborationId
            queryDirectory = $queryDir
            appKeyVaultName = $appKeyVaultName
            frontendEndpoint = $frontendEndpoint
            outDir = $outDir
            ApiMode = $apiMode
        }
        
        if ($subscription) { $queryParams.subscription = $subscription }
        
        Invoke-ScriptStep `
            -ScriptPath "$PSScriptRoot/../09-publish-query.ps1" `
            -Parameters $queryParams `
            -Description "Query publishing (inline)"
    }
    else {
        $queryParams = @{
            persona = "woodgrove"
            collaborationId = $collaborationId
            queryDirectory = $queryDir
            appKeyVaultName = $appKeyVaultName
            frontendEndpoint = $frontendEndpoint
            outDir = $outDir
            ApiMode = $apiMode
        }
        
        if ($subscription) { $queryParams.subscription = $subscription }
        
        Invoke-ScriptStep `
            -ScriptPath "$PSScriptRoot/../09-publish-query-json.ps1" `
            -Parameters $queryParams `
            -Description "Query publishing (JSON)"
    }
    
    # ========================================================================
    # STEP 10 - Both personas vote in PARALLEL
    # ========================================================================
    Write-StepHeader "10" "Both personas vote on query"
    
    $voteJobs = @(
        @{
            Name = "vote-northwind"
            ScriptPath = "$PSScriptRoot/../10-vote-query.ps1"
            Parameters = @{
                persona = "northwind"
                collaborationId = $collaborationId
                appKeyVaultName = $appKeyVaultName
                frontendEndpoint = $frontendEndpoint
                outDir = $outDir
                ApiMode = $apiMode
            }
        },
        @{
            Name = "vote-woodgrove"
            ScriptPath = "$PSScriptRoot/../10-vote-query.ps1"
            Parameters = @{
                persona = "woodgrove"
                collaborationId = $collaborationId
                appKeyVaultName = $appKeyVaultName
                frontendEndpoint = $frontendEndpoint
                outDir = $outDir
                ApiMode = $apiMode
            }
        }
    )
    
    if ($subscription) {
        $voteJobs[0].Parameters.subscription = $subscription
        $voteJobs[1].Parameters.subscription = $subscription
    }
    
    Start-ParallelJobs -Jobs $voteJobs -Description "Query voting"
    
    # ========================================================================
    # STEP 11 - Woodgrove runs query
    # ========================================================================
    Write-StepHeader "11" "Woodgrove runs query"
    
    $runParams = @{
        persona = "woodgrove"
        collaborationId = $collaborationId
        appKeyVaultName = $appKeyVaultName
        frontendEndpoint = $frontendEndpoint
        outDir = $outDir
        ApiMode = $apiMode
    }
    
    if ($subscription) { $runParams.subscription = $subscription }
    
    Invoke-ScriptStep `
        -ScriptPath "$PSScriptRoot/../11-run-query.ps1" `
        -Parameters $runParams `
        -Description "Query execution"
    
    # ========================================================================
    # STEP 12 - View results
    # ========================================================================
    Write-StepHeader "12" "View query results"
    
    $resultsParams = @{
        persona = "woodgrove"
        collaborationId = $collaborationId
        appKeyVaultName = $appKeyVaultName
        frontendEndpoint = $frontendEndpoint
        outDir = $outDir
        ApiMode = $apiMode
    }
    
    if ($subscription) { $resultsParams.subscription = $subscription }
    
    Invoke-ScriptStep `
        -ScriptPath "$PSScriptRoot/../12-view-results.ps1" `
        -Parameters $resultsParams `
        -Description "Results viewing"
    
    # ========================================================================
    # Test completed successfully
    # ========================================================================
    Write-Log "`nAll test steps completed successfully!" "SUCCESS"
    
    # Cleanup if requested
    Invoke-Cleanup -TestSucceeded $true
    
    # Show summary
    Show-TestSummary -TestSucceeded $true
    
    exit 0
}
catch {
    Write-Log "`nTest failed with error: $_" "ERROR"
    Write-Log "Stack trace: $($_.ScriptStackTrace)" "ERROR"
    
    $script:testFailed = $true
    
    # Cleanup if requested
    Invoke-Cleanup -TestSucceeded $false
    
    # Show summary
    Show-TestSummary -TestSucceeded $false
    
    exit 1
}
