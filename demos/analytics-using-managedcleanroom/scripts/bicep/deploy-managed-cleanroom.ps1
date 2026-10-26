<#
.SYNOPSIS
    Deploys the managed clean-room collaboration via Bicep, then performs the
    ARM *action* steps that Bicep cannot express (enableWorkload, addCollaborator).

.DESCRIPTION
    Run by: Collaboration owner (Woodgrove).

    Existing collaborations are deleted first. Setup waits until ARM reports
    the old resource absent before creating a fresh one; deletion errors stop setup.

    Bicep is declarative and can only create the collaboration *resource*
    (control-plane PUT). The managed clean-room flow also requires imperative
    ARM *action* POSTs which have no declarative Bicep equivalent:

        1. Deploy Bicep            -> creates Microsoft.CleanRoom/collaborations
        2. enableWorkload (POST)   -> enables the Analytics workload
        3. addCollaborator (POST)  -> (optional) invite additional collaborators

    Everything after this (accept invitation, OIDC, publish datasets/queries,
    vote, run) is dataplane/frontend REST -- follow README-API.md Steps 03-12.

    This mirrors README-API.md Steps 02.2 - 02.4 and Appendix F (ARM API).

.PARAMETER resourceGroup
    Resource group for the collaboration control-plane resource.

.PARAMETER collaborationName
    Name of the collaboration.

.PARAMETER location
    ARM location for the collaboration resource (default: westus).

.PARAMETER resourceLocation
    Region where clean-room infra (AKS/CCF/CACI) is provisioned (default: westus).

.PARAMETER targetResourceConfiguration
    Optional target resource settings using ARM property names, including
    iPTagConfiguration. aksSku and nodePoolSize in this object override the
    corresponding script parameters and retain their validation constraints.

.PARAMETER collaborators
    Creation-time collaborators: emails or SPN objects with userIdentifier,
    objectId and tenantId.

.PARAMETER additionalCollaborators
    Optional extra emails or SPN identity objects to invite via addCollaborator.

.PARAMETER deleteExistingCollab
    Defaults to true. Retained for compatibility; false is rejected because fast
    setup always recreates. Use manage-collaboration.ps1 for existing-resource actions.

.PARAMETER apiVersion
    Microsoft.CleanRoom ARM API version (default: 2026-09-30-preview).

.PARAMETER subscription
    Optional subscription ID; defaults to the current Azure CLI account.

.PARAMETER timeoutMinutes
    Polling budget per lifecycle operation (default: 60 minutes), starting before
    submission. Creation is submitted asynchronously and its deployment is polled.
    This is not a whole-script limit or a hard timeout on each Azure CLI request.

.EXAMPLE
    ./deploy-managed-cleanroom.ps1 `
        -resourceGroup cr-collab-rg -collaborationName collab1 `
        -location westus -resourceLocation westus
#>
param(
    [Parameter(Mandatory)]
    [string]$resourceGroup,

    [Parameter(Mandatory)]
    [string]$collaborationName,

    [string]$location = "westus",

    [ValidateSet(
        "centralindia", "eastasia", "eastus", "eastus2", "germanywestcentral",
        "italynorth", "japaneast", "northeurope", "southcentralus",
        "southeastasia", "switzerlandnorth", "uaenorth", "westeurope",
        "westus", "westus2")]
    [string]$resourceLocation = "westus",

    [ValidateSet("Standard_D4ds_v5")]
    [string]$aksSku = "Standard_D4ds_v5",

    [ValidateRange(3, 3)]
    [int]$nodePoolSize = 3,

    [object[]]$collaborators = @(),

    [object[]]$additionalCollaborators = @(),

    [switch]$deleteExistingCollab = $true,

    [string]$apiVersion = "2026-09-30-preview",

    [string]$subscription,

    [ValidateRange(1, 1440)]
    [int]$timeoutMinutes = 60,

    [switch]$dryRun,

    [ValidateNotNull()]
    [hashtable]$targetResourceConfiguration = @{}
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/../common/Invoke-Collaboration.ps1"

if (-not $deleteExistingCollab) {
    throw "Fast setup always deletes and recreates existing collaborations. Use manage-collaboration.ps1 for existing-resource actions."
}
if ($apiVersion -ne "2026-09-30-preview") {
    throw "This Bicep template uses 2026-09-30-preview; use the same API version for setup."
}
if ($targetResourceConfiguration.ContainsKey("aksSku")) {
    $aksSku = $targetResourceConfiguration.aksSku
}
if ($targetResourceConfiguration.ContainsKey("nodePoolSize")) {
    $nodePoolSize = $targetResourceConfiguration.nodePoolSize
}
$effectiveTargetConfiguration = @{
    aksSku = $aksSku
    nodePoolSize = $nodePoolSize
}
foreach ($entry in $targetResourceConfiguration.GetEnumerator()) {
    if ($entry.Key -notin @("aksSku", "nodePoolSize")) {
        if ($entry.Key -cne "iPTagConfiguration") {
            throw "Unknown target resource property '$($entry.Key)'. Use ARM spelling iPTagConfiguration."
        }
        if ([string]::IsNullOrWhiteSpace($entry.Value.type) -or
            [string]::IsNullOrWhiteSpace($entry.Value.value)) {
            throw "iPTagConfiguration requires both type and value."
        }
        $effectiveTargetConfiguration[$entry.Key] = $entry.Value
    }
}
$targetConfigurationJson = $effectiveTargetConfiguration | ConvertTo-Json -Depth 20 -Compress

# Analytics is currently the only supported workload.
$workloadType = "Analytics"

$creationIdentities = @($collaborators | ForEach-Object { ConvertTo-Collaborator -Identity $_ })
$additionalIdentities = @($additionalCollaborators | ForEach-Object {
    ConvertTo-Collaborator -Identity $_
})
$context = Get-CollaborationContext -ResourceGroup $resourceGroup `
    -CollaborationName $collaborationName -Subscription $subscription `
    -ApiVersion $apiVersion -DryRun:$dryRun
$actionArgs = @{
    ResourceGroup = $resourceGroup
    CollaborationName = $collaborationName
    Subscription = $context.Subscription
    ApiVersion = $apiVersion
    TimeoutMinutes = $timeoutMinutes
    DryRun = $dryRun
}

function Get-Collab {
    Invoke-Collaboration -Context $context -AllowNotFound
}

if ($dryRun) {
    Write-Host "[DRY-RUN] Would ensure resource group '$resourceGroup' ($location) and register providers."
    Write-Host "[DRY-RUN] Would delete an existing '$collaborationName' (if present) and wait for it to vanish."
    & "$PSScriptRoot/manage-collaboration.ps1" @actionArgs -Action Delete
    Write-Host "[DRY-RUN] Would then create a fresh collaboration via Bicep."
    Write-Host "[DRY-RUN]   collaborationName=$collaborationName location=$location resourceLocation=$resourceLocation aksSku=$aksSku nodePoolSize=$nodePoolSize"
    Write-Host "[DRY-RUN]   targetResourceConfiguration=$targetConfigurationJson"
    Write-Host "[DRY-RUN]   collaborators=$($creationIdentities | ConvertTo-Json -Depth 10 -Compress -AsArray)"
    & "$PSScriptRoot/manage-collaboration.ps1" @actionArgs -Action EnableWorkload
    if ($additionalIdentities.Count) {
        & "$PSScriptRoot/manage-collaboration.ps1" @actionArgs -Action AddCollaborator `
            -Collaborators $additionalIdentities
    }
    return
}

# ---------------------------------------------------------------------------
# 0. Ensure resource group + providers
# ---------------------------------------------------------------------------
Write-Host "Ensuring resource group '$resourceGroup' ($location)..."
Invoke-CleanroomAz -Arguments @("group", "create", "--name", $resourceGroup,
    "--location", $location, "--subscription", $context.Subscription, "-o", "none") | Out-Null

Write-Host "Registering resource providers (idempotent)..."
foreach ($provider in @("Microsoft.CleanRoom", "Microsoft.ContainerService")) {
    Invoke-CleanroomAz -Arguments @("provider", "register", "--namespace", $provider,
        "--subscription", $context.Subscription, "--wait", "-o", "none") | Out-Null
}

# ---------------------------------------------------------------------------
# 0.5 Handle an existing collaboration of the same name
# ---------------------------------------------------------------------------
$existing = Get-Collab
if ($existing) {
    Write-Warning "Deleting existing collaboration '$collaborationName' for a clean setup."
    & "$PSScriptRoot/manage-collaboration.ps1" @actionArgs -Action Delete
    Write-Host "Existing collaboration deleted."
}

# ---------------------------------------------------------------------------
# 1. Deploy the collaboration (declarative -- Bicep)
# ---------------------------------------------------------------------------
# The owner is added automatically from the caller's ARM token.
Write-Host "Deploying collaboration '$collaborationName' via Bicep (~25 min)..."
$creationDeadline = (Get-Date).AddMinutes($timeoutMinutes)
$targetConfigurationFile = [System.IO.Path]::GetTempFileName()
try {
    $parameters = @{
        targetResourceConfiguration = @{ value = $effectiveTargetConfiguration }
        collaborators = @{ value = $creationIdentities }
    } | ConvertTo-Json -Depth 20
    [System.IO.File]::WriteAllText($targetConfigurationFile, $parameters)
    Invoke-CleanroomAz -Arguments @("deployment", "group", "create",
        "--resource-group", $resourceGroup, "--subscription", $context.Subscription,
        "--name", "deploy-$collaborationName",
        "--template-file", "$PSScriptRoot/managed-cleanroom.bicep",
        "--parameters", "collaborationName=$collaborationName", "location=$location",
        "resourceLocation=$resourceLocation", "aksSku=$aksSku", "nodePoolSize=$nodePoolSize",
        "@$targetConfigurationFile", "--no-wait", "-o", "none") | Out-Null
}
finally {
    Remove-Item -LiteralPath $targetConfigurationFile -Force
}

$collab = Wait-Collaboration -Context $context -Description "collaboration provisioning" `
    -TimeoutMinutes $timeoutMinutes -Deadline $creationDeadline `
    -DeploymentName "deploy-$collaborationName" -AllowNotFound `
    -Condition { param($c) $c.properties.provisioningState -eq "Succeeded" }
Write-Host "Collaboration resource ready: $($collab.id)"

# ---------------------------------------------------------------------------
# 2. Enable the workload (ARM *action* -- no Bicep equivalent)
# ---------------------------------------------------------------------------
Write-Host "Enabling '$workloadType' workload (~7 min)..."
& "$PSScriptRoot/manage-collaboration.ps1" @actionArgs -Action EnableWorkload | Out-Null

# ---------------------------------------------------------------------------
# 3. Add additional collaborators (ARM *action* -- no Bicep equivalent)
# ---------------------------------------------------------------------------
if ($additionalIdentities.Count) {
    & "$PSScriptRoot/manage-collaboration.ps1" @actionArgs -Action AddCollaborator `
        -Collaborators $additionalIdentities
}

Write-Host ""
Write-Host "Done. Collaboration '$collaborationName' is provisioned and healthy."
Write-Host "Next: follow README-API.md Steps 03-12 (accept invitation, OIDC,"
Write-Host "publish datasets/queries, vote, run) using the frontend REST API."
