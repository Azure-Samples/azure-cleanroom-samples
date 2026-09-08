<#
.SYNOPSIS
    Step 03 [EACH COLLABORATOR] - Accept a collaboration invitation.

.DESCRIPTION
    Lists collaborations visible to the persona, resolves the collaboration
    UUID (frontend id, NOT the ARM resource id), lists pending invitations,
    and accepts one.

    Mirrors README-API.md Step 03 (3.1 Get UUID, 3.2 Accept Invitation).

.PARAMETER Persona
    Collaborator persona (e.g. woodgrove, northwind). Selects the token file.

.PARAMETER CollaborationId
    Optional frontend collaboration UUID. If omitted, the script lists
    collaborations and prompts for a selection.

.PARAMETER InvitationId
    Optional invitation id. If omitted, the first pending invitation is used.

.EXAMPLE
    ./03-accept-invitation.ps1 -Persona northwind
#>
param(
    [Parameter(Mandatory)][string]$Persona,
    [string]$CollaborationId,
    [string]$InvitationId,
    [string]$Frontend,
    [string]$TokenFile,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/Invoke-Frontend.ps1"

$fe = Get-FrontendContext -Persona $Persona -Frontend $Frontend -TokenFile $TokenFile -DryRun:$DryRun

# 3.1 Resolve collaboration UUID.
if (-not $CollaborationId) {
    if ($DryRun) {
        $CollaborationId = "<collaboration-id>"
    }
    else {
        $collabs = (Invoke-Frontend -Context $fe -Path "" -Method GET).collaborations
        if (-not $collabs) { throw "No collaborations visible to persona '$Persona'." }
        $collabs | Format-Table @{L = '#'; E = { [array]::IndexOf($collabs, $_) + 1 } }, collaborationName, collaborationId, userStatus
        $choice = Read-Host "Enter the number of your collaboration"
        $CollaborationId = $collabs[[int]$choice - 1].collaborationId
    }
}
Write-Host "Collaboration: $CollaborationId"

# 3.2 Accept invitation.
if (-not $InvitationId) {
    if ($DryRun) {
        $InvitationId = "<invitation-id>"
    }
    else {
        $invitations = (Invoke-Frontend -Context $fe -Path "$CollaborationId/invitations" -Method GET).invitations
        if (-not $invitations) { throw "No pending invitations for '$Persona' on $CollaborationId." }
        $invitations | Format-Table invitationId, accountType, status
        $InvitationId = $invitations[0].invitationId
    }
}

Invoke-Frontend -Context $fe -Path "$CollaborationId/invitations/$InvitationId/accept" -Method POST
Write-Host "Accepted invitation $InvitationId."
Write-Host "CollaborationId (save for later steps): $CollaborationId"
