<#
.SYNOPSIS
    Assigns RBAC roles to app service principals.

.DESCRIPTION
    Assigns specified RBAC role to app service principal on resource group scope.
    Run this AFTER creating Azure resources (step 04-prepare-resources.ps1).

.PARAMETER persona
    Persona name (northwind or woodgrove) to load app metadata.

.PARAMETER resourceGroupName
    Resource group to assign permissions on.

.PARAMETER appKeyVaultName
    Key Vault containing app metadata.

.PARAMETER outDir
    Output directory for app metadata (default: ./generated).

.PARAMETER role
    RBAC role to assign (default: Contributor).

.PARAMETER userAccessAdministrator
    Also assign User Access Administrator role (needed for step 07-grant-access).

.PARAMETER subscription
    Optional Azure subscription name or ID.
#>
param(
    [Parameter(Mandatory)]
    [string]$persona,

    [Parameter(Mandatory)]
    [string]$resourceGroupName,

    [Parameter(Mandatory)]
    [string]$appKeyVaultName,

    [string]$outDir = "./generated",

    [string]$role = "Contributor",
    
    [switch]$userAccessAdministrator = $true,
    
    [string]$subscription
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

function Invoke-AzCommand {
    param([string[]]$Arguments)
    $cmdLine = "az $($Arguments -join ' ')"
    Write-Host "[CMD] $cmdLine" -ForegroundColor DarkGray
    & az @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
}

function Invoke-AzSafe {
    param([string[]]$Arguments)
    $PSNativeCommandUseErrorActionPreference = $false
    $result = & az @Arguments 2>$null
    if ($LASTEXITCODE -eq 0) { return $result }
    return $null
}

Write-Host "=== Assigning RBAC Permissions to App ===" -ForegroundColor Cyan

# Construct path to app metadata file
$appMetadataDir = Join-Path $outDir "apps"
$appMetadataFile = Join-Path $appMetadataDir "$persona-app.json"

if (-not (Test-Path $appMetadataFile)) {
    Write-Host "ERROR: App metadata file not found: $appMetadataFile" -ForegroundColor Red
    Write-Host "Run 00-create-apps.ps1 first to create apps." -ForegroundColor Yellow
    exit 1
}

if ($subscription) {
    Write-Host "Setting subscription to '$subscription'..." -ForegroundColor Yellow
    Invoke-AzCommand @("account", "set", "--subscription", $subscription)
}

$appMeta = Get-Content $appMetadataFile | ConvertFrom-Json
$servicePrincipalId = $appMeta.servicePrincipalId
$displayName = $appMeta.displayName

Write-Host "App: $displayName" -ForegroundColor Yellow
Write-Host "Service Principal ID: $servicePrincipalId" -ForegroundColor Yellow
Write-Host "Resource Group: $resourceGroupName" -ForegroundColor Yellow

$rgExists = Invoke-AzCommand @("group", "exists", "--name", $resourceGroupName)
if ($rgExists -eq "false") {
    Write-Host "ERROR: Resource group '$resourceGroupName' does not exist." -ForegroundColor Red
    Write-Host "Run 04-prepare-resources.ps1 first to create resources." -ForegroundColor Yellow
    exit 1
}

$rgId = Invoke-AzCommand @("group", "show", "--name", $resourceGroupName, "--query", "id", "-o", "tsv")

Write-Host "`nAssigning '$role' role..." -ForegroundColor Cyan
$existing = Invoke-AzSafe @("role", "assignment", "list", 
    "--assignee", $servicePrincipalId, 
    "--role", $role, 
    "--scope", $rgId)

if ($existing) {
    Write-Host "Role '$role' already assigned (skipped)." -ForegroundColor Yellow
} else {
    Invoke-AzCommand @("role", "assignment", "create", 
        "--assignee-object-id", $servicePrincipalId, 
        "--assignee-principal-type", "ServicePrincipal", 
        "--role", $role, 
        "--scope", $rgId, 
        "--output", "none")
    Write-Host "Role '$role' assigned successfully." -ForegroundColor Green
}

if ($userAccessAdministrator) {
    Write-Host "`nAssigning 'User Access Administrator' role..." -ForegroundColor Cyan
    $existing = Invoke-AzSafe @("role", "assignment", "list", 
        "--assignee", $servicePrincipalId, 
        "--role", "User Access Administrator", 
        "--scope", $rgId)
    
    if ($existing) {
        Write-Host "Role 'User Access Administrator' already assigned (skipped)." -ForegroundColor Yellow
    } else {
        Invoke-AzCommand @("role", "assignment", "create", 
            "--assignee-object-id", $servicePrincipalId, 
            "--assignee-principal-type", "ServicePrincipal", 
            "--role", "User Access Administrator", 
            "--scope", $rgId, 
            "--output", "none")
        Write-Host "Role 'User Access Administrator' assigned successfully." -ForegroundColor Green
    }
}

Write-Host "`n=== Role Assignment Complete ===" -ForegroundColor Green
Write-Host "App '$displayName' now has access to resource group '$resourceGroupName'." -ForegroundColor Yellow

# Wait for RBAC propagation
Write-Host "`n=== Waiting for RBAC Propagation ===" -ForegroundColor Cyan
Write-Host "RBAC assignments can take 60-120 seconds to propagate across all Azure services..." -ForegroundColor Yellow
Write-Host "Waiting 90 seconds for propagation..." -ForegroundColor Yellow
Start-Sleep -Seconds 90

# Verify role assignments are queryable
Write-Host "Verifying role assignments are queryable..." -ForegroundColor Yellow
$maxRetries = 10
$retryInterval = 15
$rolesVerified = $false

for ($i = 1; $i -le $maxRetries; $i++) {
    $assignments = Invoke-AzSafe @("role", "assignment", "list", 
        "--assignee", $servicePrincipalId, 
        "--scope", $rgId)
    
    if ($assignments) {
        $assignmentsList = $assignments | ConvertFrom-Json
        $contributorFound = $assignmentsList | Where-Object { $_.roleDefinitionName -eq $role }
        
        if ($contributorFound) {
            if ($userAccessAdministrator) {
                $uaaFound = $assignmentsList | Where-Object { $_.roleDefinitionName -eq "User Access Administrator" }
                if ($uaaFound) {
                    Write-Host "All role assignments verified and queryable." -ForegroundColor Green
                    $rolesVerified = $true
                    break
                }
            } else {
                Write-Host "Role assignment verified and queryable." -ForegroundColor Green
                $rolesVerified = $true
                break
            }
        }
    }
    
    if ($i -lt $maxRetries) {
        Write-Host "  Roles not yet queryable. Retrying in $retryInterval seconds (attempt $i/$maxRetries)..." -ForegroundColor Yellow
        Start-Sleep -Seconds $retryInterval
    }
}

if (-not $rolesVerified) {
    Write-Host "WARNING: Could not verify role assignments after waiting. They may still be propagating." -ForegroundColor Yellow
    Write-Host "Proceeding anyway - subsequent steps may need to retry if permissions aren't ready." -ForegroundColor Yellow
}

Write-Host "`n=== RBAC Setup Complete ===" -ForegroundColor Green
