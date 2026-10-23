<#
.SYNOPSIS
    Lists or inspects frontend resources without changing them.
.DESCRIPTION
    Returns objects for filtering, formatting, or saving. Run requires the full
    cl-spark-<uuid> JobId. Dataset and Query require DocumentId.
#>
param(
    [Parameter(Mandatory)][string]$Persona,
    [Parameter(Mandatory)]
    [ValidateSet("Collaborations", "Invitations", "Collaborators", "OidcKeys",
        "Datasets", "Dataset", "Queries", "Query", "Run", "RunHistory", "AuditEvents")]
    [string]$Resource,
    [string]$CollaborationId,
    [Alias("QueryName")][string]$DocumentId,
    [string]$JobId,
    [switch]$PendingOnly,
    [switch]$ActiveOnly,
    [string]$Frontend,
    [string]$TokenFile,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/Invoke-Frontend.ps1"
if ($Resource -ne "Collaborations" -and -not $CollaborationId) {
    throw "-CollaborationId is required for $Resource."
}
if ($Resource -in @("Dataset", "Query", "RunHistory") -and -not $DocumentId) {
    throw "-DocumentId (or -QueryName) is required for $Resource."
}
if ($Resource -eq "Run" -and -not $JobId) { throw "-JobId is required for Run." }
if ($PendingOnly -and $Resource -ne "Invitations") {
    throw "-PendingOnly applies only to Invitations."
}
if ($ActiveOnly -and $Resource -ne "Collaborations") {
    throw "-ActiveOnly applies only to Collaborations."
}
$id = [Uri]::EscapeDataString($CollaborationId)
$document = [Uri]::EscapeDataString($DocumentId)
$job = [Uri]::EscapeDataString($JobId)
$path = switch ($Resource) {
    "Collaborations" { if ($ActiveOnly) { "?activeOnly=true" } else { "" } }
    "Invitations" { "$id/invitations" + $(if ($PendingOnly) { "?pendingOnly=true" }) }
    "Collaborators" { "$id/collaborators" }
    "OidcKeys" { "$id/oidc/keys" }
    "Datasets" { "$id/analytics/datasets" }
    "Dataset" { "$id/analytics/datasets/$document" }
    "Queries" { "$id/analytics/queries" }
    "Query" { "$id/analytics/queries/$document" }
    "Run" { "$id/analytics/runs/$job" }
    "RunHistory" { "$id/analytics/queries/$document/runs" }
    "AuditEvents" { "$id/analytics/auditevents" }
}
$fe = Get-FrontendContext -Persona $Persona -Frontend $Frontend -TokenFile $TokenFile -DryRun:$DryRun
Invoke-Frontend -Context $fe -Path $path
