<#
.SYNOPSIS
    Helper functions for ARM collaboration operations via az managedcleanroom CLI.

.DESCRIPTION
    Wraps `az managedcleanroom collaboration` commands into reusable functions.
    All functions require the Private CleanRoom Cloud to be configured first
    (call Initialize-PrivateCleanRoomCloud before any other function).

.NOTES
    Usage:
        . "./scripts/common/collaboration-helpers.ps1"
        Initialize-PrivateCleanRoomCloud
        New-Collaboration -CollaborationName "mycollab" -ResourceGroup "myrg" -Location "westus"
#>

function Initialize-PrivateCleanRoomCloud {
    <#
    .SYNOPSIS
        Registers and activates the Private CleanRoom Azure Cloud.
    .DESCRIPTION
        Registers a custom cloud pointing to the Private CleanRoom RP ARM endpoint,
        sets it as active, and enables the private namespace env var.
    #>
    param(
        [string]$ArmEndpoint = "https://eastus2euap.management.azure.com/"
    )

    $privateCloudName = "PrivateCleanroomAzureCloud"
    Write-Host "Configuring Private CleanRoom Cloud..." -ForegroundColor Cyan

    # Register (ignore error if already registered)
    az cloud register --name $privateCloudName `
        --endpoint-resource-manager $ArmEndpoint 2>$null
    az cloud set --name $privateCloudName

    $env:UsePrivateCleanRoomNamespace = "true"
    Write-Host "Private CleanRoom Cloud configured (ARM: $ArmEndpoint)." -ForegroundColor Green
}

function New-Collaboration {
    <#
    .SYNOPSIS
        Creates a new managed cleanroom collaboration.
    .DESCRIPTION
        CLI: az managedcleanroom collaboration create
    #>
    param(
        [Parameter(Mandatory)][string]$CollaborationName,
        [Parameter(Mandatory)][string]$ResourceGroup,
        [Parameter(Mandatory)][string]$Location
    )

    Write-Host "Creating collaboration '$CollaborationName'..." -ForegroundColor Cyan
    az managedcleanroom collaboration create `
        --collaboration-name $CollaborationName `
        --resource-group $ResourceGroup `
        --location $Location
}

function Get-Collaboration {
    <#
    .SYNOPSIS
        Shows a managed cleanroom collaboration.
    .DESCRIPTION
        CLI: az managedcleanroom collaboration show
    #>
    param(
        [Parameter(Mandatory)][string]$CollaborationName,
        [Parameter(Mandatory)][string]$ResourceGroup
    )

    az managedcleanroom collaboration show `
        --collaboration-name $CollaborationName `
        --resource-group $ResourceGroup
}

function Enable-CollaborationWorkload {
    <#
    .SYNOPSIS
        Enables a workload type on a collaboration.
    .DESCRIPTION
        CLI: az managedcleanroom collaboration enable-workload
    #>
    param(
        [Parameter(Mandatory)][string]$CollaborationName,
        [Parameter(Mandatory)][string]$ResourceGroup,
        [string]$WorkloadType = "analytics"
    )

    Write-Host "Enabling '$WorkloadType' workload on '$CollaborationName'..." -ForegroundColor Cyan
    az managedcleanroom collaboration enable-workload `
        --collaboration-name $CollaborationName `
        --resource-group $ResourceGroup `
        --workload-type $WorkloadType
}

function Add-Collaborator {
    <#
    .SYNOPSIS
        Adds a collaborator to a collaboration by email.
    .DESCRIPTION
        CLI: az managedcleanroom collaboration add-collaborator
    #>
    param(
        [Parameter(Mandatory)][string]$CollaborationName,
        [Parameter(Mandatory)][string]$ResourceGroup,
        [Parameter(Mandatory)][string]$UserIdentifier
    )

    Write-Host "Adding collaborator '$UserIdentifier' to '$CollaborationName'..." -ForegroundColor Cyan
    az managedcleanroom collaboration add-collaborator `
        --collaboration-name $CollaborationName `
        --resource-group $ResourceGroup `
        --user-identifier $UserIdentifier
}
