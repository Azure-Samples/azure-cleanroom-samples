<#
.SYNOPSIS
    Adds an app (SPN) as a collaborator to a managed cleanroom collaboration.

.DESCRIPTION
    Uses az rest to call the Private CleanRoom RP directly (eastus2euap ARM endpoint)
    without requiring an az cloud switch. The caller must be logged in as the collaboration
    owner (e.g., admin@contoso.com) with access to the collaboration subscription.

    IMPORTANT: The --user-identifier for an SPN is the Application (client) ID,
    NOT the Enterprise App object ID or the App Registration object ID.

.PARAMETER collaborationName
    Name of the collaboration.

.PARAMETER resourceGroup
    Resource group containing the collaboration.

.PARAMETER subscription
    Subscription ID of the collaboration.

.PARAMETER appClientId
    Application (client) ID of the app to add as collaborator.

.PARAMETER apiVersion
    ARM API version (default: 2026-03-31-preview).

.EXAMPLE
    ./scripts/common/add-app-collaborator.ps1 -collaborationName <collaboration-name> -resourceGroup <resource-group> -subscription "<your-subscription-id>" -appClientId "<your-app-client-id>"
#>
param(
    [Parameter(Mandatory)]
    [string]$collaborationName,

    [Parameter(Mandatory)]
    [string]$resourceGroup,

    [Parameter(Mandatory)]
    [string]$subscription,

    [Parameter(Mandatory)]
    [string]$appClientId,

    [string]$apiVersion = "2026-03-31-preview"
)

$ErrorActionPreference = 'Stop'

Write-Host "=== Adding app collaborator to '$collaborationName' ===" -ForegroundColor Cyan
Write-Host "  App Client ID: $appClientId" -ForegroundColor Yellow

# Get a management token from the current az login session
$token = az account get-access-token --resource "https://management.azure.com/" --query accessToken -o tsv
if ($LASTEXITCODE -ne 0 -or -not $token) {
    throw "Failed to get management token. Run 'az login' first."
}

$baseUrl = "https://eastus2euap.management.azure.com"
$collabUrl = "$baseUrl/subscriptions/$subscription/resourceGroups/$resourceGroup/providers/Microsoft.CleanRoom/collaborations/$collaborationName"

# Step 1: Verify the collaboration exists
Write-Host "Verifying collaboration..." -ForegroundColor Cyan
$headers = @{ Authorization = "Bearer $token" }
try {
    $collab = Invoke-RestMethod -Method Get -Uri "$($collabUrl)?api-version=$apiVersion" -Headers $headers
    Write-Host "  Collaboration found: $($collab.name), State: $($collab.properties.provisioningState)" -ForegroundColor Green
    Write-Host "  Current collaborators:" -ForegroundColor Yellow
    foreach ($c in $collab.properties.collaborators) {
        Write-Host "    - $($c.userIdentifier) (type=$($c.identityType), owner=$($c.isCollaborationOwner))" -ForegroundColor Yellow
    }
} catch {
    throw "Cannot read collaboration: $_"
}

# Step 2: Add the app as collaborator
Write-Host "`nAdding app as collaborator..." -ForegroundColor Cyan
$addUrl = "$($collabUrl)/addCollaborator?api-version=$apiVersion"
$body = @{ userIdentifier = $appClientId } | ConvertTo-Json

try {
    $result = Invoke-RestMethod -Method Post -Uri $addUrl -Headers $headers -Body $body -ContentType "application/json"
    Write-Host "  Collaborator added successfully!" -ForegroundColor Green
} catch {
    $errBody = $_.ErrorDetails.Message
    if ($errBody -match "already exists|already added|Conflict") {
        Write-Host "  App already added as collaborator (idempotent)." -ForegroundColor Yellow
    } else {
        Write-Host "  Failed to add collaborator: $errBody" -ForegroundColor Red
        throw "Add collaborator failed: $_"
    }
}

# Step 3: Verify
Write-Host "`nVerifying updated collaborators..." -ForegroundColor Cyan
$collab = Invoke-RestMethod -Method Get -Uri "$($collabUrl)?api-version=$apiVersion" -Headers $headers
foreach ($c in $collab.properties.collaborators) {
    $marker = if ($c.userIdentifier -eq $appClientId) { ">>> NEW" } else { "   " }
    Write-Host "  $marker $($c.userIdentifier) (type=$($c.identityType), objId=$($c.objectId))" -ForegroundColor $(if ($c.userIdentifier -eq $appClientId) { "Green" } else { "Yellow" })
}

Write-Host "`n=== Done ===" -ForegroundColor Green
Write-Host "Next: The app may need to accept an invitation via the frontend API." -ForegroundColor Yellow
