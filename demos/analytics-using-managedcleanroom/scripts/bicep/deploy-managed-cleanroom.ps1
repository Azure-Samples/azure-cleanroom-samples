<#
.SYNOPSIS
    Deploys the managed clean-room collaboration via Bicep, then performs the
    ARM *action* steps that Bicep cannot express (enableWorkload, addCollaborator).

.DESCRIPTION
    Run by: Collaboration owner (Woodgrove).

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

.PARAMETER additionalCollaborators
    Optional list of extra collaborator identifiers to invite via addCollaborator.

.PARAMETER deleteExistingCollab
    If a collaboration with the same name exists: when set, delete it (and wait
    for its resources to vanish) before provisioning a new one; when not set,
    skip provisioning and leave the existing collaboration as-is.

.PARAMETER apiVersion
    Microsoft.CleanRoom ARM API version (default: 2026-04-30-preview).

.EXAMPLE
    ./deploy-managed-cleanroom.ps1 `
        -resourceGroup cr-collab-rg -collaborationName collab1 -resourceLocation westus
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

    [string[]]$additionalCollaborators = @(),

    [switch]$deleteExistingCollab,

    [string]$apiVersion = "2026-04-30-preview",

    [switch]$dryRun
)

$ErrorActionPreference = "Stop"
$armEndpoint = "https://management.azure.com"

# Analytics is currently the only supported workload.
$workloadType = "Analytics"

$subscription = (az account show -o json | ConvertFrom-Json).id
$collabArmUrl = "$armEndpoint/subscriptions/$subscription/resourceGroups/$resourceGroup/providers/Microsoft.CleanRoom/Collaborations/$collaborationName"

function Get-Collab {
    az rest --method GET --url "${collabArmUrl}?api-version=$apiVersion" --resource $armEndpoint -o json 2>$null | ConvertFrom-Json
}

# Polls a condition until it returns truthy or the timeout elapses.
function Wait-Until {
    param(
        [Parameter(Mandatory)][scriptblock]$Condition,
        [Parameter(Mandatory)][string]$Description,
        [int]$TimeoutMinutes = 20,
        [int]$IntervalSeconds = 30
    )
    $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
    while (-not (& $Condition)) {
        if ((Get-Date) -gt $deadline) { throw "Timed out after $TimeoutMinutes min waiting for: $Description" }
        Start-Sleep -Seconds $IntervalSeconds
    }
}

if ($dryRun) {
    Write-Host "[DRY-RUN] Would ensure resource group '$resourceGroup' ($location) and register providers."
    if ($deleteExistingCollab) {
        Write-Host "[DRY-RUN] Would delete an existing '$collaborationName' (if present) and wait for it to vanish."
    }
    Write-Host "[DRY-RUN] Would deploy Bicep: $PSScriptRoot/managed-cleanroom.bicep (owner from caller token)"
    Write-Host "[DRY-RUN]   collaborationName=$collaborationName location=$location resourceLocation=$resourceLocation"
    Write-Host "[DRY-RUN] Would POST $collabArmUrl/enableWorkload  body {workloadType=$workloadType}"
    foreach ($collaborator in $additionalCollaborators) {
        Write-Host "[DRY-RUN] Would POST $collabArmUrl/addCollaborator  body {collaborator.userIdentifier=$collaborator}"
    }
    return
}

# ---------------------------------------------------------------------------
# 0. Ensure resource group + providers
# ---------------------------------------------------------------------------
Write-Host "Ensuring resource group '$resourceGroup' ($location)..."
az group create --name $resourceGroup --location $location -o none

Write-Host "Registering resource providers (idempotent)..."
az provider register --namespace Microsoft.CleanRoom -o none
az provider register --namespace Microsoft.ContainerService -o none

# ---------------------------------------------------------------------------
# 0.5 Handle an existing collaboration of the same name
# ---------------------------------------------------------------------------
$existing = Get-Collab
if ($existing -and $existing.id) {
    if ($deleteExistingCollab) {
        Write-Host "Deleting existing collaboration '$collaborationName'..."
        az rest --method DELETE --url "${collabArmUrl}?api-version=$apiVersion" --resource $armEndpoint -o none
        Wait-Until -Description "existing '$collaborationName' to be deleted" -TimeoutMinutes 30 -Condition { -not (Get-Collab) }
        Write-Host "Existing collaboration deleted."
    }
    else {
        Write-Host "Collaboration '$collaborationName' already exists (provisioningState=$($existing.properties.provisioningState)); skipping provisioning."
        Write-Host "Pass -DeleteExistingCollab to delete and recreate it."
        return
    }
}

# ---------------------------------------------------------------------------
# 1. Deploy the collaboration (declarative -- Bicep)
# ---------------------------------------------------------------------------
# The owner is added automatically from the caller's ARM token, so no
# collaborators are passed here. az deployment group create blocks until the
# deployment reaches a terminal state, so no separate provisioning poll is needed.
Write-Host "Deploying collaboration '$collaborationName' via Bicep (~25 min)..."
az deployment group create `
    --resource-group $resourceGroup `
    --name "deploy-$collaborationName" `
    --template-file "$PSScriptRoot/managed-cleanroom.bicep" `
    --parameters `
        collaborationName=$collaborationName `
        location=$location `
        resourceLocation=$resourceLocation `
    -o none
if ($LASTEXITCODE -ne 0) {
    throw "Bicep deployment failed (az deployment group create exit code $LASTEXITCODE). See the error above."
}

$collab = Get-Collab
if (-not $collab -or -not $collab.id) {
    throw "Collaboration '$collaborationName' was not found after deployment. Aborting."
}
Write-Host "Collaboration resource ready: $($collab.id)"

# ---------------------------------------------------------------------------
# 2. Enable the workload (ARM *action* -- no Bicep equivalent)
# ---------------------------------------------------------------------------
Write-Host "Enabling '$workloadType' workload (~7 min)..."
$bodyFile = Join-Path $PSScriptRoot "body.json"
[System.IO.File]::WriteAllText($bodyFile, (@{ workloadType = $workloadType } | ConvertTo-Json))
az rest --method POST `
    --url "$collabArmUrl/enableWorkload?api-version=$apiVersion" `
    --resource $armEndpoint `
    --headers "Content-Type=application/json" `
    --body "@$bodyFile" -o none

# az rest does not surface the Azure-AsyncOperation header, so poll the resource
# for the workload endpoint and health, each guarded by a timeout.
Wait-Until -Description "'$workloadType' workload endpoint" -TimeoutMinutes 15 -Condition {
    $c = Get-Collab
    if ($c.properties.provisioningState -eq "Failed") { throw "Workload enablement failed." }
    $wl = $c.properties.workloads | Where-Object { $_.workloadType -eq $workloadType } | Select-Object -First 1
    [bool]$wl.endpoint
}
Wait-Until -Description "collaboration healthState=Ok" -TimeoutMinutes 15 -Condition {
    $c = Get-Collab
    if ($c.properties.health.healthIssues) {
        $c.properties.health.healthIssues | ForEach-Object { Write-Host "  Issue: $($_ | ConvertTo-Json -Compress)" }
    }
    $c.properties.health.healthState -eq "Ok"
}

# ---------------------------------------------------------------------------
# 3. Add additional collaborators (ARM *action* -- no Bicep equivalent)
# ---------------------------------------------------------------------------
foreach ($collaborator in $additionalCollaborators) {
    Write-Host "Adding collaborator '$collaborator'..."
    [System.IO.File]::WriteAllText($bodyFile, (@{ collaborator = @{ userIdentifier = $collaborator } } | ConvertTo-Json))
    az rest --method POST `
        --url "$collabArmUrl/addCollaborator?api-version=$apiVersion" `
        --resource $armEndpoint `
        --headers "Content-Type=application/json" `
        --body "@$bodyFile" -o none

    Wait-Until -Description "collaborator '$collaborator' to be added" -TimeoutMinutes 10 -Condition {
        (Get-Collab).properties.collaborators.userIdentifier -contains $collaborator
    }
}

Remove-Item $bodyFile -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Done. Collaboration '$collaborationName' is provisioned and healthy."

Write-Host ""
Write-Host "Done. Collaboration '$collaborationName' is provisioned and healthy."
Write-Host "Next: follow README-API.md Steps 03-12 (accept invitation, OIDC,"
Write-Host "publish datasets/queries, vote, run) using the frontend REST API."
