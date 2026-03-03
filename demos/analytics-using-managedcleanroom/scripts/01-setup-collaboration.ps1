<#
.SYNOPSIS
    Sets up a managed cleanroom collaboration for the analytics scenario.

.DESCRIPTION
    Run by: Owner (e.g., Woodgrove).
    Creates a collaboration resource, adds one or more collaborators by email,
    enables the analytics workload, and outputs the collaboration details
    (including the Collaboration ARM ID and Frontend Endpoint needed by all subsequent steps).

.PARAMETER collaborationName
    Name for the collaboration resource.

.PARAMETER resourceGroup
    The Azure resource group in which to create the collaboration.

.PARAMETER collaboratorEmails
    One or more email addresses of collaborators to add to the collaboration.
    Example: @("northwind@company.com", "woodgrove@company.com", "contoso@company.com")

.PARAMETER location
    Azure region for the collaboration (default: westus).

.PARAMETER subscription
    Optional Azure subscription name or ID.
#>
param(
    [Parameter(Mandatory)]
    [string]$collaborationName,

    [Parameter(Mandatory)]
    [string]$resourceGroup,

    [Parameter(Mandatory)]
    [string[]]$collaboratorEmails,

    [string]$location = "westus",

    [string]$subscription
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
}
else {
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

# Step 3: Add collaborators (skip if already added).
Write-Host "`n=== Step 3: Adding collaborators ===" -ForegroundColor Cyan
foreach ($email in $collaboratorEmails) {
    Invoke-AzIdempotent @("managedcleanroom", "collaboration", "add-collaborator",
        "--collaboration-name", $collaborationName,
        "--resource-group", $resourceGroup,
        "--email", $email) -ActionName "Add collaborator '$email'"
}
Write-Host "Collaborator setup complete." -ForegroundColor Green

# Step 4: Enable the analytics workload (skip if already enabled).
Write-Host "`n=== Step 4: Enabling analytics workload ===" -ForegroundColor Cyan
Invoke-AzIdempotent @("managedcleanroom", "collaboration", "enable-workload",
    "--collaboration-name", $collaborationName,
    "--resource-group", $resourceGroup,
    "--workload-type", "analytics") -ActionName "Enable analytics workload"

# Step 5: Retrieve collaboration details.
Write-Host "`n=== Step 5: Retrieving collaboration details ===" -ForegroundColor Cyan
$collaboration = Invoke-AzCommand @("managedcleanroom", "collaboration", "show",
    "--collaboration-name", $collaborationName,
    "--resource-group", $resourceGroup) | ConvertFrom-Json

$collaborationId = $collaboration.id
# Extract frontend endpoint from workloads (confirmed from source: workloads[].endpoint — client_flatten in CLI)
$frontendEndpoint = ($collaboration.workloads | Where-Object { $_.workloadType -eq "analytics" }).endpoint
if (-not $frontendEndpoint) {
    Write-Host "WARNING: Could not extract frontend endpoint from collaboration show. Workload may not be provisioned yet." -ForegroundColor Red
    Write-Host "Run 'az managedcleanroom collaboration show' later and look for properties.workloads[].endpoint." -ForegroundColor Red
}

Write-Host "`nCollaboration setup complete." -ForegroundColor Green
Write-Host "`nCollaboration details:"
$collaboration | ConvertTo-Json -Depth 10

Write-Host "`n========================================" -ForegroundColor Yellow
Write-Host "IMPORTANT - Share the following with all collaborators:" -ForegroundColor Yellow
Write-Host "  Collaboration ARM ID:  $collaborationId" -ForegroundColor Yellow
Write-Host "  Frontend Endpoint:     $frontendEndpoint" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "`nCollaborators need the Collaboration ARM ID for Step 2 (accept invitation)" -ForegroundColor Yellow
Write-Host "and the Frontend Endpoint for Steps 2 and 6." -ForegroundColor Yellow
