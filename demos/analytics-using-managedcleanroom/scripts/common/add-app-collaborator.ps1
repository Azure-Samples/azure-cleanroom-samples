<#
.SYNOPSIS
    Adds an app (SPN) as a collaborator to a managed cleanroom collaboration.

.DESCRIPTION
    Calls the production CleanRoom RP through the public Azure ARM endpoint.
    The caller must be logged in as the collaboration
    owner (e.g., admin@contoso.com) with access to the collaboration subscription.

    IMPORTANT: The --user-identifier for an SPN is the Application (client) ID,
    NOT the Enterprise App object ID or the App Registration object ID.
    Supply the Enterprise Application objectId and tenantId as well.
    SPNs auto-activate and do not need invitation acceptance.

.PARAMETER collaborationName
    Name of the collaboration.

.PARAMETER resourceGroup
    Resource group containing the collaboration.

.PARAMETER subscription
    Subscription ID of the collaboration.

.PARAMETER appClientId
    Application (client) ID of the app to add as collaborator.

.PARAMETER objectId
    Enterprise Application object ID of the service principal.

.PARAMETER tenantId
    Tenant ID of the service principal.

.PARAMETER apiVersion
    ARM API version (default: 2026-09-30-preview).

.PARAMETER NoWait
    Submit without waiting for the collaborator to appear on the resource.

.PARAMETER DryRun
    Preview the request without changing the collaboration.

.EXAMPLE
    ./scripts/common/add-app-collaborator.ps1 -collaborationName <collaboration-name> -resourceGroup <resource-group> -subscription "<your-subscription-id>" -appClientId "<your-app-client-id>" -objectId "<sp-object-id>" -tenantId "<tenant-id>"
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

    [Parameter(Mandatory)]
    [string]$objectId,

    [Parameter(Mandatory)]
    [string]$tenantId,

    [string]$apiVersion = "2026-09-30-preview",
    [switch]$NoWait,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

& "$PSScriptRoot/../bicep/manage-collaboration.ps1" -Action AddCollaborator `
    -ResourceGroup $resourceGroup -CollaborationName $collaborationName `
    -Subscription $subscription -ApiVersion $apiVersion -NoWait:$NoWait -DryRun:$DryRun `
    -Collaborators @(@{ userIdentifier = $appClientId; objectId = $objectId; tenantId = $tenantId })
