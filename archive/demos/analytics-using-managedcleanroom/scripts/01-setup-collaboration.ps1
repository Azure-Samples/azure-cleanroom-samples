<#
.SYNOPSIS
    Sets up a managed cleanroom collaboration for the analytics scenario.

.DESCRIPTION
    Run by: Owner (e.g., Woodgrove).
    Creates a collaboration resource, adds one or more collaborators by email,
    enables the analytics workload, polls for provisioning, and outputs the
    collaboration details (ARM ID, frontend collaboration UUID, and frontend endpoint)
    needed by all subsequent steps.

    Requires the Private CleanRoom cloud (eastus2euap ARM endpoint) and
    UsePrivateCleanRoomNamespace=true for the az managedcleanroom commands.
    The script handles cloud switching automatically and restores AzureCloud afterward.

.PARAMETER collaborationName
    Name for the collaboration resource.

.PARAMETER resourceGroup
    The Azure resource group in which to create the collaboration.

.PARAMETER collaboratorEmails
    One or more email addresses of collaborators to add to the collaboration.
    Example: @("alice@example.com", "bob@example.com")

.PARAMETER location
    Azure region for the collaboration (default: westus).

.PARAMETER subscription
    Optional Azure subscription name or ID for the collaboration resource.

.PARAMETER outDir
    Output directory for generated files (default: ./generated).
#>
param(
    [Parameter(Mandatory)]
    [string]$collaborationName,

    [Parameter(Mandatory)]
    [string]$resourceGroup,

    [Parameter(Mandatory)]
    [string[]]$collaboratorEmails,

    [string]$location = "westus",

    [string]$subscription,

    [string]$outDir = "./generated"
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

# Runs an az command idempotently: skips on "already exists" errors, throws on real failures.
function Invoke-AzIdempotent {
    param(
        [string[]]$Arguments,
        [string]$SkipPattern = "already exists|already added|already enabled|Conflict",
        [string]$ActionName = "operation"
    )
    $cmdLine = "az $($Arguments -join ' ')"
    Write-Host "[CMD] $cmdLine" -ForegroundColor DarkGray
    $PSNativeCommandUseErrorActionPreference = $false
    $result = & az @Arguments 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  $ActionName - done." -ForegroundColor Green
        return $result
    }
    $errText = ($result | Where-Object { $_ -is [System.Management.Automation.ErrorRecord] }) -join "`n"
    if ($errText -match $SkipPattern) {
        Write-Host "  $ActionName - already done (skipped). Server said: $errText" -ForegroundColor Yellow
        return $null
    }
    Write-Host "  ERROR: $errText" -ForegroundColor Red
    throw "$ActionName failed: $errText"
}

# =============================================================================
# Private CleanRoom Cloud Setup
# The az managedcleanroom collaboration commands require:
#   1. UsePrivateCleanRoomNamespace=true (selects Private.CleanRoom RP namespace)
#   2. PrivateCleanroomAzureCloud (points ARM at eastus2euap.management.azure.com)
# =============================================================================
Write-Host "=== Configuring Private CleanRoom cloud ===" -ForegroundColor Cyan
$env:UsePrivateCleanRoomNamespace = "true"
$privateCloudName = "PrivateCleanroomAzureCloud"

# Save current cloud so we can restore it later.
$previousCloud = az cloud show --query name -o tsv 2>$null
if (-not $previousCloud) { $previousCloud = "AzureCloud" }

# Register the private cloud if not already registered.
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

# Switch to Private CleanRoom cloud.
if ($previousCloud -ne $privateCloudName) {
    Write-Host "  Switching to Private CleanRoom cloud..." -ForegroundColor Yellow
    az cloud set --name $privateCloudName 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to set Private CleanRoom cloud"
        exit 1
    }

    # Re-login is required after cloud switch (new ARM endpoint).
    Write-Host "  Logging in (interactive)..." -ForegroundColor Yellow
    Write-Host "  You will be prompted to authenticate. Log in as the collaboration owner." -ForegroundColor Yellow
    az login --allow-no-subscriptions 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to login after cloud switch"
        exit 1
    }
}
Write-Host "  Private CleanRoom cloud active." -ForegroundColor Green

# Set subscription if specified.
if ($subscription) {
    Write-Host "Setting subscription to '$subscription'..." -ForegroundColor Yellow
    Invoke-AzCommand @("account", "set", "--subscription", $subscription)
}

# Ensure Microsoft.CleanRoom provider is registered.
Write-Host "`n=== Checking Microsoft.CleanRoom provider registration ===" -ForegroundColor Cyan
$regState = Invoke-AzCommand @("provider", "show", "--namespace", "Microsoft.CleanRoom", "--query", "registrationState", "-o", "tsv")
if ($regState -ne "Registered") {
    Write-Host "Registering Microsoft.CleanRoom provider..." -ForegroundColor Yellow
    Invoke-AzCommand @("provider", "register", "--namespace", "Microsoft.CleanRoom")
    while ($regState -ne "Registered") {
        Start-Sleep -Seconds 10
        $regState = Invoke-AzCommand @("provider", "show", "--namespace", "Microsoft.CleanRoom", "--query", "registrationState", "-o", "tsv")
        Write-Host "  Registration state: $regState"
    }
    Write-Host "Microsoft.CleanRoom provider registered." -ForegroundColor Green
} else {
    Write-Host "Microsoft.CleanRoom provider already registered." -ForegroundColor Green
}

# Step 0: Ensure resource group exists.
Write-Host "`n=== Step 0: Ensuring resource group '$resourceGroup' exists ===" -ForegroundColor Cyan
$rgExists = Invoke-AzCommand @("group", "exists", "--name", $resourceGroup)
if ($rgExists -eq "false") {
    Write-Host "Creating resource group '$resourceGroup' in '$location'..." -ForegroundColor Yellow
    Invoke-AzCommand @("group", "create", "--name", $resourceGroup, "--location", $location, "--output", "none")
    Write-Host "Resource group created." -ForegroundColor Green
} else {
    Write-Host "Resource group '$resourceGroup' already exists." -ForegroundColor Green
}

# Step 1: Get the owner's identity details for --user-identity.
Write-Host "`n=== Step 1: Getting owner identity ===" -ForegroundColor Cyan
$accountInfo = Invoke-AzCommand @("account", "show") | ConvertFrom-Json
$ownerObjectId = Invoke-AzCommand @("ad", "signed-in-user", "show", "--query", "id", "-o", "tsv")
$ownerTenantId = $accountInfo.tenantId

Write-Host "Owner Object ID: $ownerObjectId"
Write-Host "Owner Tenant ID: $ownerTenantId"

# Step 2: Create the collaboration (skip if already exists).
Write-Host "`n=== Step 2: Creating collaboration '$collaborationName' ===" -ForegroundColor Cyan
$userIdentity = "{tenant-id:$ownerTenantId,object-id:$ownerObjectId,account-type:microsoft}"
$existingCollab = Invoke-AzSafe @("managedcleanroom", "collaboration", "show",
    "--collaboration-name", $collaborationName,
    "--resource-group", $resourceGroup)
if ($existingCollab) {
    Write-Host "Collaboration '$collaborationName' already exists." -ForegroundColor Green
} else {
    Invoke-AzCommand @("managedcleanroom", "collaboration", "create",
        "--collaboration-name", $collaborationName,
        "--resource-group", $resourceGroup,
        "--location", $location,
        "--consortium-type", "ConfidentialACI",
        "--user-identity", $userIdentity)
    Write-Host "Collaboration '$collaborationName' created." -ForegroundColor Green
}

# Step 3: Add collaborators by email (skip if already added).
Write-Host "`n=== Step 3: Adding collaborators ===" -ForegroundColor Cyan
foreach ($email in $collaboratorEmails) {
    Invoke-AzIdempotent @("managedcleanroom", "collaboration", "add-collaborator",
        "--collaboration-name", $collaborationName,
        "--resource-group", $resourceGroup,
        "--user-identifier", $email) -ActionName "Add collaborator '$email'"
}
Write-Host "Collaborator setup complete." -ForegroundColor Green

# Step 4: Enable the analytics workload (skip if already enabled).
Write-Host "`n=== Step 4: Enabling analytics workload ===" -ForegroundColor Cyan
Invoke-AzIdempotent @("managedcleanroom", "collaboration", "enable-workload",
    "--collaboration-name", $collaborationName,
    "--resource-group", $resourceGroup,
    "--workload-type", "analytics") -ActionName "Enable analytics workload"

# Step 5: Poll for collaboration provisioning to complete.
Write-Host "`n=== Step 5: Waiting for collaboration provisioning ===" -ForegroundColor Cyan

$maxRetries = 20
$retryInterval = 30

for ($i = 1; $i -le $maxRetries; $i++) {
    Write-Host "  Checking provisioning state (attempt $i/$maxRetries)..." -ForegroundColor Yellow

    $collaboration = Invoke-AzCommand @("managedcleanroom", "collaboration", "show",
        "--collaboration-name", $collaborationName,
        "--resource-group", $resourceGroup) | ConvertFrom-Json

    $provisioningState = $collaboration.provisioningState
    Write-Host "  Current provisioning state: $provisioningState" -ForegroundColor Yellow

    if ($provisioningState -eq "Succeeded") {
        Write-Host "  Collaboration provisioning completed successfully!" -ForegroundColor Green
        break
    } elseif ($provisioningState -eq "Failed") {
        Write-Host "ERROR: Collaboration provisioning failed!" -ForegroundColor Red
        throw "Collaboration provisioning failed with state: $provisioningState"
    }

    if ($i -eq $maxRetries) {
        Write-Host "WARNING: Collaboration still provisioning after $($maxRetries * $retryInterval) seconds." -ForegroundColor Red
        throw "Collaboration provisioning timeout"
    }

    Write-Host "  Still provisioning. Waiting $retryInterval seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds $retryInterval
}

# Step 6: Retrieve collaboration details.
Write-Host "`n=== Step 6: Retrieving collaboration details ===" -ForegroundColor Cyan
$collaboration = Invoke-AzCommand @("managedcleanroom", "collaboration", "show",
    "--collaboration-name", $collaborationName,
    "--resource-group", $resourceGroup) | ConvertFrom-Json

$collaborationArmId = $collaboration.id

# Extract frontend collaboration UUID (used by all frontend scripts).
$collaborationUuid = $collaboration.collaborationId
if (-not $collaborationUuid) {
    Write-Host "WARNING: Could not extract collaboration UUID from ARM response." -ForegroundColor Red
    Write-Host "You may need to get it from the frontend API after accepting the invitation." -ForegroundColor Red
}

# Extract frontend endpoint from workloads array.
$frontendEndpoint = ($collaboration.workloads | Where-Object { $_.workloadType -eq "analytics" }).endpoint
if (-not $frontendEndpoint) {
    Write-Host "WARNING: Could not extract frontend endpoint. Workload may not be provisioned yet." -ForegroundColor Red
    Write-Host "Run 'az managedcleanroom collaboration show' later and check workloads[].endpoint." -ForegroundColor Red
}

# =============================================================================
# Restore previous cloud
# =============================================================================
Write-Host "`n=== Restoring cloud to '$previousCloud' ===" -ForegroundColor Cyan
az cloud set --name $previousCloud 2>&1 | Out-Null
Write-Host "  Switched back to '$previousCloud'." -ForegroundColor Green
Write-Host "  You may need to run 'az login' to re-authenticate with the restored cloud." -ForegroundColor Yellow

# =============================================================================
# Save outputs
# =============================================================================
Write-Host "`n=== Step 7: Saving collaboration details ===" -ForegroundColor Cyan

if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
}

# Save ARM resource ID
$collaborationArmId | Out-File -FilePath (Join-Path $outDir "collaboration-id.txt") -Encoding utf8 -NoNewline
Write-Host "  Collaboration ARM ID saved to: $outDir/collaboration-id.txt" -ForegroundColor Green

# Save frontend collaboration UUID
if ($collaborationUuid) {
    $collaborationUuid | Out-File -FilePath (Join-Path $outDir "collaboration-uuid.txt") -Encoding utf8 -NoNewline
    Write-Host "  Collaboration UUID saved to: $outDir/collaboration-uuid.txt" -ForegroundColor Green
}

# Save frontend endpoint
if ($frontendEndpoint) {
    $frontendEndpoint | Out-File -FilePath (Join-Path $outDir "frontend-endpoint.txt") -Encoding utf8 -NoNewline
    Write-Host "  Frontend endpoint saved to: $outDir/frontend-endpoint.txt" -ForegroundColor Green
}

# Print summary
Write-Host "`nCollaboration setup complete." -ForegroundColor Green
Write-Host "`n========================================" -ForegroundColor Yellow
Write-Host "COLLABORATION DETAILS" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  Collaboration Name:     $collaborationName" -ForegroundColor Yellow
Write-Host "  Collaboration ARM ID:   $collaborationArmId" -ForegroundColor Yellow
Write-Host "  Collaboration UUID:     $collaborationUuid" -ForegroundColor Yellow
Write-Host "  Frontend Endpoint:      $frontendEndpoint" -ForegroundColor Yellow
Write-Host "  Resource Group:         $resourceGroup" -ForegroundColor Yellow
Write-Host "  Location:               $location" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
