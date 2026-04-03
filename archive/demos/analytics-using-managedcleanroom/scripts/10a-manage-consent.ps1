<#
.SYNOPSIS
    Enables or disables execution consent on a dataset or query.

.DESCRIPTION
    Run by: Any collaborator.
    Manages execution consent for a specific document (dataset or query) in the
    collaboration. Consent can be enabled or disabled independently of voting.

.PARAMETER collaborationId
    The collaboration frontend UUID.

.PARAMETER documentId
    The document ID (dataset name or query name) to manage consent for.

.PARAMETER action
    The consent action: "enable" or "disable".

.PARAMETER frontendEndpoint
    Frontend service URL.

.PARAMETER TokenFile
    Optional path to a pre-generated MSAL IdToken file.

.PARAMETER ApiMode
    API mode: "rest" (default) or "cli".
#>
param(
    [Parameter(Mandatory)]
    [string]$collaborationId,

    [Parameter(Mandatory)]
    [string]$documentId,

    [Parameter(Mandatory)]
    [ValidateSet("enable", "disable")]
    [string]$action,

    [Parameter(Mandatory)]
    [string]$frontendEndpoint,

    [string]$TokenFile,

    [ValidateSet("rest", "cli")]
    [string]$ApiMode = "rest"
)

# Configure Private CleanRoom cloud and verify local user auth
. "$PSScriptRoot/common/setup-local-auth.ps1"

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# Load common frontend helpers
. "$PSScriptRoot/common/frontend-helpers.ps1"
$feCtx = New-FrontendContext -frontendEndpoint $frontendEndpoint -ApiMode $ApiMode

Write-Host "=== $($action.ToUpper()) consent on '$documentId' ===" -ForegroundColor Cyan

Set-FrontendConsent -Context $feCtx `
    -CollaborationId $collaborationId `
    -DocumentId $documentId `
    -Action $action `
    -TokenFile $TokenFile

Write-Host "Consent '$action' applied to document '$documentId'." -ForegroundColor Green
