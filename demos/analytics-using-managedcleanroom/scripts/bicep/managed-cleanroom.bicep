// =============================================================================
// managed-cleanroom.bicep
// =============================================================================
// Customer-facing Bicep for the Azure Managed Clean Room.
//
// WHAT THIS DEPLOYS (declaratively, via the ARM control plane):
//   Microsoft.CleanRoom/collaborations  -- the durable collaboration resource
//   that provisions the underlying clean-room infrastructure (CCF network,
//   AKS cluster, confidential ACI) in `resourceLocation`.
//
// This is the Bicep equivalent of Step 02.2 "Create Collaboration"
// (the `az rest --method PUT ...Microsoft.CleanRoom/Collaborations/{name}`
// call) in README-API.md. See Appendix F for the ARM endpoint reference.
//
// WHAT THIS DOES *NOT* DO (Bicep cannot express these):
//   - enableWorkload / addCollaborator      -> ARM *action* POSTs (imperative)
//   - accept invitations, OIDC, publish     -> dataplane / frontend REST
//     datasets & queries, vote, run
//   Run deploy-managed-cleanroom.ps1 (action steps) and the README-API.md
//   frontend flow (dataplane steps) after this template succeeds.
//
// KNOWN WARNING:
//   The Microsoft.CleanRoom Bicep *types* are not published to the public
//   type registry yet, so `az deployment` / `bicep build` emit a harmless
//   BCP081 ("resource type ... not available") warning. The deployment still
//   works. The `#disable-next-line BCP081` directive below suppresses it.
//
// PREREQUISITE:
//   The subscription must be allow-listed for the (private-preview)
//   Microsoft.CleanRoom RP, and the provider registered:
//     az provider register --namespace Microsoft.CleanRoom
//     az provider register --namespace Microsoft.ContainerService
// =============================================================================

@description('Name of the collaboration resource.')
param collaborationName string

@description('ARM location for the collaboration control-plane resource (e.g. westus). This is NOT where clean-room infra is deployed.')
param location string = resourceGroup().location

@description('Region where the clean-room infrastructure (CCF network, AKS, confidential ACI) is provisioned. Must be a supported resourceLocation.')
@allowed([
  'centralindia'
  'eastasia'
  'eastus'
  'eastus2'
  'germanywestcentral'
  'italynorth'
  'japaneast'
  'northeurope'
  'southcentralus'
  'southeastasia'
  'switzerlandnorth'
  'uaenorth'
  'westeurope'
  'westus'
  'westus2'
])
param resourceLocation string = 'westus'

@description('Optional additional collaborators added at creation time. Each entry is { userIdentifier: "<email-or-appId>" }. The owner is derived from the caller token automatically. Add further collaborators later via the addCollaborator action (see deploy script).')
param collaborators array = []

// -----------------------------------------------------------------------------
// Collaboration (mirrors the Step 02.2 create body exactly:
//   { location, properties: { resourceLocation, collaborators[] } })
// -----------------------------------------------------------------------------
#disable-next-line BCP081
resource collaboration 'Microsoft.CleanRoom/collaborations@2026-04-30-preview' = {
  name: collaborationName
  location: location
  properties: {
    resourceLocation: resourceLocation
    collaborators: collaborators
  }
}

// -----------------------------------------------------------------------------
// Outputs (consumed by deploy-managed-cleanroom.ps1 for the action steps)
// -----------------------------------------------------------------------------
@description('ARM resource ID of the collaboration.')
output collaborationId string = collaboration.id

@description('Collaboration name (echoed for convenience).')
output collaborationName string = collaboration.name

@description('Region where clean-room infra was provisioned.')
output resourceLocation string = resourceLocation
