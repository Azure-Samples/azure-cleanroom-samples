<#
.SYNOPSIS
    Sets up a managed cleanroom collaboration for the analytics scenario.

.DESCRIPTION
    Run by: Owner (e.g., Woodgrove).
    Creates a collaboration resource, adds collaborators (managed identities),
    enables the analytics workload, waits for provisioning to complete,
    and verifies collaborators were added successfully via the frontend API.

.PARAMETER collaborationName
    Name for the collaboration resource.

.PARAMETER resourceGroupName
    The Azure resource group in which to create the collaboration.

.PARAMETER location
    Azure region for the collaboration (default: westus).

.PARAMETER subscription
    Optional Azure subscription name or ID.

.PARAMETER northwindMIPrincipalId
    Northwind managed identity principal ID.

.PARAMETER woodgroveMIPrincipalId
    Woodgrove managed identity principal ID.

.PARAMETER persona
    Persona (northwind or woodgrove) for naming/logging.

.PARAMETER outDir
    Output directory for generated files (default: ./generated).

.PARAMETER frontendEndpoint
    Optional frontend endpoint URL. Defaults to static westus endpoint.
#>
param(
    [Parameter(Mandatory)]
    [string]$collaborationName,

    [Parameter(Mandatory)]
    [string]$resourceGroupName,

    [string]$location = "westus",

    [string]$subscription,

    [Parameter(Mandatory)]
    [string]$northwindMIPrincipalId,

    [Parameter(Mandatory)]
    [string]$woodgroveMIPrincipalId,

    [string]$persona,

    [string]$outDir = "./generated",

    [string]$frontendEndpoint
)

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
Write-Host "`n=== Step 0: Ensuring resource group '$resourceGroupName' exists ===" -ForegroundColor Cyan
$rgExists = Invoke-AzCommand @("group", "exists", "--name", $resourceGroupName)
if ($rgExists -eq "false") {
    Write-Host "Creating resource group '$resourceGroupName' in '$location'..." -ForegroundColor Yellow
    Invoke-AzCommand @("group", "create", "--name", $resourceGroupName, "--location", $location, "--output", "none")
    Write-Host "Resource group created." -ForegroundColor Green
}
else {
    Write-Host "Resource group '$resourceGroupName' already exists." -ForegroundColor Green
}

# Step 1: Get the owner's identity details for --user-identity.
Write-Host "`n=== Step 1: Getting owner identity ===" -ForegroundColor Cyan
$accountInfo = Invoke-AzCommand @("account", "show") | ConvertFrom-Json

# For managed identity authentication, use the woodgrove MI principal ID as owner
# The account type will be "servicePrincipal" when logged in with MI
if ($accountInfo.user.type -eq "servicePrincipal") {
    Write-Host "Authenticated as Managed Identity" -ForegroundColor Yellow
    # When this script is run by Woodgrove VM, use its MI principal ID as owner
    $ownerObjectId = $woodgroveMIPrincipalId
    $ownerTenantId = $accountInfo.tenantId
    Write-Host "Owner (MI) Principal ID: $ownerObjectId"
    Write-Host "Owner Tenant ID: $ownerTenantId"
} else {
    # For user authentication
    $ownerObjectId = Invoke-AzCommand @("ad", "signed-in-user", "show", "--query", "id", "-o", "tsv")
    $ownerTenantId = $accountInfo.tenantId
    Write-Host "Owner (User) Object ID: $ownerObjectId"
    Write-Host "Owner Tenant ID: $ownerTenantId"
}

# Step 2: Create the collaboration (skip if already exists).
Write-Host "`n=== Step 2: Creating collaboration '$collaborationName' ===" -ForegroundColor Cyan
$userIdentity = "{tenant-id:$ownerTenantId,object-id:$ownerObjectId,account-type:microsoft}"
$existingCollab = Invoke-AzSafe @("managedcleanroom", "collaboration", "show",
    "--collaboration-name", $collaborationName,
    "--resource-group", $resourceGroupName)
if ($existingCollab) {
    Write-Host "Collaboration '$collaborationName' already exists." -ForegroundColor Green
} else {
    Invoke-AzCommand @("managedcleanroom", "collaboration", "create",
        "--collaboration-name", $collaborationName,
        "--resource-group", $resourceGroupName,
        "--location", $location,
        "--consortium-type", "ConfidentialACI",
        "--user-identity", $userIdentity)
    Write-Host "Collaboration '$collaborationName' created." -ForegroundColor Green
}

# Step 3: Add collaborators using REST API (supports managed identities).
Write-Host "`n=== Step 3: Adding collaborators ===" -ForegroundColor Cyan

# Get current subscription ID
$subscriptionId = $accountInfo.id

# Get collaboration resource ID
$collaborationId = "/subscriptions/$subscriptionId/resourceGroups/$resourceGroupName/providers/Microsoft.CleanRoom/collaborations/$collaborationName"

# Get current tenant ID for collaborators
$tenantId = $accountInfo.tenantId

# Add both collaborators using managed identity principal IDs
$collaborators = @(
    @{
        displayName = "Northwind"
        miPrincipalId = $northwindMIPrincipalId
    },
    @{
        displayName = "Woodgrove"
        miPrincipalId = $woodgroveMIPrincipalId
    }
)

foreach ($collaborator in $collaborators) {
    $displayName = $collaborator.displayName
    $miPrincipalId = $collaborator.miPrincipalId
    
    Write-Host "Adding collaborator: $displayName (MI Principal ID: $miPrincipalId)..." -ForegroundColor Yellow
    
    # Construct the REST API payload
    $payload = @{
        Collaborator = @{
            UserIdentifier = $miPrincipalId
            ObjectId = $miPrincipalId
            TenantId = $tenantId
        }
    } | ConvertTo-Json -Compress
    
    # Call the REST API using az rest
    # Note: Using --method POST because add-collaborator is a POST operation
    $restUrl = "https://eastus2euap.management.azure.com${collaborationId}/addCollaborator?api-version=2025-01-31-preview"
    
    $PSNativeCommandUseErrorActionPreference = $false
    $addResult = az rest --method POST --url $restUrl --body $payload 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Collaborator '$displayName' added successfully." -ForegroundColor Green
    } else {
        $errText = $addResult -join "`n"
        if ($errText -match "already exists|already added|Conflict") {
            Write-Host "  Collaborator '$displayName' already added (skipped)." -ForegroundColor Yellow
        } else {
            Write-Host "  ERROR adding collaborator '$displayName': $errText" -ForegroundColor Red
            throw "Failed to add collaborator '$displayName'"
        }
    }
}

Write-Host "Collaborator addition complete." -ForegroundColor Green

# Step 4: Enable the analytics workload (skip if already enabled).
Write-Host "`n=== Step 4: Enabling analytics workload ===" -ForegroundColor Cyan
Invoke-AzIdempotent @("managedcleanroom", "collaboration", "enable-workload",
    "--collaboration-name", $collaborationName,
    "--resource-group", $resourceGroupName,
    "--workload-type", "analytics") -ActionName "Enable analytics workload"

# Step 5: Poll for collaboration provisioning to complete.
Write-Host "`n=== Step 5: Waiting for collaboration provisioning ===" -ForegroundColor Cyan

$maxRetries = 20
$retryInterval = 30
$provisioningComplete = $false

for ($i = 1; $i -le $maxRetries; $i++) {
    Write-Host "  Checking provisioning state (attempt $i/$maxRetries)..." -ForegroundColor Yellow
    
    $collaboration = Invoke-AzCommand @("managedcleanroom", "collaboration", "show",
        "--collaboration-name", $collaborationName,
        "--resource-group", $resourceGroupName) | ConvertFrom-Json
    
    $provisioningState = $collaboration.provisioningState
    Write-Host "  Current provisioning state: $provisioningState" -ForegroundColor Yellow
    
    if ($provisioningState -eq "Succeeded") {
        Write-Host "  Collaboration provisioning completed successfully!" -ForegroundColor Green
        $provisioningComplete = $true
        break
    } elseif ($provisioningState -eq "Failed") {
        Write-Host "ERROR: Collaboration provisioning failed!" -ForegroundColor Red
        throw "Collaboration provisioning failed with state: $provisioningState"
    }
    
    if ($i -lt $maxRetries) {
        Write-Host "  Still provisioning. Waiting $retryInterval seconds..." -ForegroundColor Yellow
        Start-Sleep -Seconds $retryInterval
    }
}

if (-not $provisioningComplete) {
    Write-Host "WARNING: Collaboration still provisioning after $($maxRetries * $retryInterval) seconds." -ForegroundColor Red
    Write-Host "Current state: $provisioningState" -ForegroundColor Red
    throw "Collaboration provisioning timeout"
}

# Use static frontend endpoint for westus region
if (-not $frontendEndpoint) {
    $frontendEndpoint = "https://dogfood.workload-frontendwestus.cleanroom.cloudapp.azure-test.net/collaborations"
    Write-Host "Using static frontend endpoint: $frontendEndpoint" -ForegroundColor Cyan
}

# Step 6: Verify collaborators were added successfully using frontend API.
if ($frontendEndpoint) {
    Write-Host "`n=== Step 6: Verifying collaborators via frontend API ===" -ForegroundColor Cyan
    
    # Setup frontend certificate
    Write-Host "Setting up frontend certificate..." -ForegroundColor Yellow
    & "$PSScriptRoot/common/setup-frontend-certificate.ps1" -frontendEndpoint $frontendEndpoint -outDir $outDir
    
    # Login to frontend
    Write-Host "Logging into frontend..." -ForegroundColor Yellow
    az managedcleanroom frontend login --endpoint $frontendEndpoint 2>&1 | Out-Null
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Frontend login failed. Skipping collaborator verification." -ForegroundColor Yellow
    } else {
        # Poll for collaborator status
        $maxVerifyRetries = 10
        $verifyInterval = 15
        $collaboratorsVerified = $false
        
        for ($i = 1; $i -le $maxVerifyRetries; $i++) {
            Write-Host "  Verifying collaborator status (attempt $i/$maxVerifyRetries)..." -ForegroundColor Yellow
            
            $PSNativeCommandUseErrorActionPreference = $false
            $collabsResponse = az managedcleanroom frontend analytics collaborations list 2>&1
            
            if ($LASTEXITCODE -eq 0) {
                $collabs = $collabsResponse | ConvertFrom-Json
                
                # Look for our collaboration in the list
                $ourCollab = $collabs.collaborations | Where-Object { $_.collaborationName -eq $collaborationName }
                
                if ($ourCollab -and $ourCollab.userStatus -eq "Active") {
                    Write-Host "  Collaborators verified successfully! Status: $($ourCollab.userStatus)" -ForegroundColor Green
                    $collaboratorsVerified = $true
                    break
                } elseif ($ourCollab) {
                    Write-Host "  Collaboration found but status is: $($ourCollab.userStatus)" -ForegroundColor Yellow
                }
            }
            
            if ($i -lt $maxVerifyRetries) {
                Write-Host "  Collaborators not yet active. Waiting $verifyInterval seconds..." -ForegroundColor Yellow
                Start-Sleep -Seconds $verifyInterval
            }
        }
        
        if (-not $collaboratorsVerified) {
            Write-Host "WARNING: Could not verify collaborator status after $($maxVerifyRetries * $verifyInterval) seconds." -ForegroundColor Yellow
            Write-Host "Collaborators may still be propagating. This might not be a critical error." -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "`n=== Step 6: Skipping collaborator verification (no frontend endpoint) ===" -ForegroundColor Yellow
}

# Step 7: Save collaboration details to output files.
Write-Host "`n=== Step 7: Saving collaboration details ===" -ForegroundColor Cyan

# Get final collaboration details
$collaboration = Invoke-AzCommand @("managedcleanroom", "collaboration", "show",
    "--collaboration-name", $collaborationName,
    "--resource-group", $resourceGroupName) | ConvertFrom-Json

$collaborationId = $collaboration.id

# Save collaboration ID to file
$collaborationIdFile = Join-Path $outDir "collaboration-id.txt"
$collaborationId | Out-File -FilePath $collaborationIdFile -Encoding utf8 -NoNewline
Write-Host "Collaboration ID saved to: $collaborationIdFile" -ForegroundColor Green

# Save frontend endpoint to file
if ($frontendEndpoint) {
    $frontendEndpointFile = Join-Path $outDir "frontend-endpoint.txt"
    $frontendEndpoint | Out-File -FilePath $frontendEndpointFile -Encoding utf8 -NoNewline
    Write-Host "Frontend endpoint saved to: $frontendEndpointFile" -ForegroundColor Green
}

Write-Host "`nCollaboration setup complete." -ForegroundColor Green
Write-Host "`n========================================" -ForegroundColor Yellow
Write-Host "COLLABORATION DETAILS" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  Collaboration Name:    $collaborationName" -ForegroundColor Yellow
Write-Host "  Collaboration ARM ID:  $collaborationId" -ForegroundColor Yellow
Write-Host "  Frontend Endpoint:     $frontendEndpoint" -ForegroundColor Yellow
Write-Host "  Resource Group:        $resourceGroupName" -ForegroundColor Yellow
Write-Host "  Location:              $location" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
