<#
.SYNOPSIS
    Accepts a collaboration invitation via the managed cleanroom frontend.

.DESCRIPTION
    Run by: Each collaborator (Northwind and Woodgrove).
    Configures the frontend endpoint, authenticates the user, lists pending
    invitations, and accepts the one matching the given collaboration.

    Prerequisites: The collaboration must have been created and the collaborator
    must have been added (01-setup-collaboration.ps1).

.PARAMETER collaborationId
    The collaboration ARM resource ID (obtained from the collaboration owner).
    Example: /subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.CleanRoom/collaborations/{name}

.PARAMETER frontendEndpoint
    The Analytics Frontend API endpoint URL (obtained from the collaboration owner).
#>
param(
    [Parameter(Mandatory)]
    [string]$collaborationId,

    [Parameter(Mandatory)]
    [string]$frontendEndpoint
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# Runs an az command idempotently: skips on "already exists" errors, throws on real failures.
function Invoke-AzIdempotent {
    param(
        [string[]]$Arguments,
        [string]$SkipPattern = "already exists|already added|already enabled|already accepted|Conflict",
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

# Step 1: Configure the frontend endpoint.
Write-Host "=== Step 1: Configuring frontend endpoint ===" -ForegroundColor Cyan
az managedcleanroom frontend configure --endpoint $frontendEndpoint
Write-Host "Frontend endpoint configured: $frontendEndpoint" -ForegroundColor Green

# Step 2: Authenticate.
Write-Host "`n=== Step 2: Authenticating ===" -ForegroundColor Cyan
Write-Host "You will be prompted to authenticate via device code flow." -ForegroundColor Yellow
az managedcleanroom frontend login
Write-Host "Authentication successful." -ForegroundColor Green

# Step 3: List invitations and find the one for this collaboration.
Write-Host "`n=== Step 3: Listing invitations ===" -ForegroundColor Cyan
$PSNativeCommandUseErrorActionPreference = $false
$response = az managedcleanroom frontend invitation list `
    --collaboration-id $collaborationId 2>$null | ConvertFrom-Json
$PSNativeCommandUseErrorActionPreference = $true

# Response schema: { "invitations": [{ "invitationId": "..." }, ...] }
$invitations = $response.invitations
if (-not $invitations -or $invitations.Count -eq 0) {
    Write-Host "No pending invitations found - likely already accepted." -ForegroundColor Yellow
    Write-Host "You are already a member of the collaboration." -ForegroundColor Green
    exit 0
}

Write-Host "Found $($invitations.Count) invitation(s):" -ForegroundColor Green
$invitations | ForEach-Object { Write-Host "  - Invitation ID: $($_.invitationId)" }

# Step 4: Accept the first pending invitation (skip if already accepted).
$invitationId = $invitations[0].invitationId
Write-Host "`n=== Step 4: Accepting invitation '$invitationId' ===" -ForegroundColor Cyan
Invoke-AzIdempotent @("managedcleanroom", "frontend", "invitation", "accept",
    "--collaboration-id", $collaborationId,
    "--invitation-id", $invitationId) -ActionName "Accept invitation"

# Step 5: Verify.
Write-Host "`n=== Step 5: Verifying invitation ===" -ForegroundColor Cyan
$invitationDetails = az managedcleanroom frontend invitation show `
    --collaboration-id $collaborationId `
    --invitation-id $invitationId | ConvertFrom-Json
$invitationDetails | ConvertTo-Json -Depth 5

Write-Host "`nYou are now a member of the collaboration." -ForegroundColor Green
