<#
.SYNOPSIS
    Shows, enables, adds collaborators to, recovers, or deletes an existing collaboration.
.DESCRIPTION
    Uses ARM 2026-09-30-preview. Mutations wait for completion unless -NoWait is
    supplied. Delete never recreates the collaboration. Recovery is a last resort;
    -ForceRecover sends forceRecover=true. -DryRun only previews requests.
.PARAMETER Collaborators
    Emails or objects with userIdentifier, objectId and tenantId for SPNs.
.PARAMETER TimeoutMinutes
    Polling budget per operation (default: 60 minutes), starting before submission.
    Each added collaborator has its own budget. Timeout does not cancel Azure work.
#>
param(
    [Parameter(Mandatory)][string]$ResourceGroup,
    [Parameter(Mandatory)][string]$CollaborationName,
    [Parameter(Mandatory)]
    [ValidateSet("Show", "EnableWorkload", "AddCollaborator", "Recover", "Delete")]
    [string]$Action,
    [object[]]$Collaborators = @(),
    [string]$Subscription,
    [string]$ApiVersion = "2026-09-30-preview",
    [ValidateRange(1, 1440)][int]$TimeoutMinutes = 60,
    [switch]$ForceRecover,
    [switch]$NoWait,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/../common/Invoke-Collaboration.ps1"
if ($ForceRecover -and $Action -ne "Recover") {
    throw "-ForceRecover applies only to -Action Recover."
}
if ($Collaborators.Count -and $Action -ne "AddCollaborator") {
    throw "-Collaborators applies only to -Action AddCollaborator."
}
if ($Action -eq "AddCollaborator" -and -not $Collaborators.Count) {
    throw "-Collaborators is required for AddCollaborator."
}
$identities = @($Collaborators | ForEach-Object { ConvertTo-Collaborator -Identity $_ })
$context = Get-CollaborationContext -ResourceGroup $ResourceGroup `
    -CollaborationName $CollaborationName -Subscription $Subscription `
    -ApiVersion $ApiVersion -DryRun:$DryRun

switch ($Action) {
    "Show" { return Invoke-Collaboration -Context $context }
    "Delete" {
        $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
        Invoke-Collaboration -Context $context -Method DELETE | Out-Null
        if (-not $DryRun -and -not $NoWait) {
            Wait-Collaboration -Context $context -Description "collaboration deletion" `
                -TimeoutMinutes $TimeoutMinutes -Deadline $deadline -AllowNotFound `
                -Condition { param($c) $null -eq $c } | Out-Null
        }
    }
    "Recover" {
        # Resource provisioningState can stay Succeeded throughout recovery.
        Invoke-CollaborationOperation -Context $context -Action recover `
            -Body @{ forceRecover = [bool]$ForceRecover } `
            -TimeoutMinutes $TimeoutMinutes -NoWait:$NoWait
        if (-not $DryRun -and -not $NoWait) {
            Invoke-Collaboration -Context $context
        }
    }
    "EnableWorkload" {
        $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
        $existing = if (-not $DryRun) { Invoke-Collaboration -Context $context }
        $workload = $existing.properties.workloads |
            Where-Object { $_.workloadType -eq "Analytics" } | Select-Object -First 1
        if (-not $workload.endpoint) {
            Invoke-Collaboration -Context $context -Method POST -Action enableWorkload `
                -Body @{ workloadType = "Analytics" } | Out-Null
        }
        if (-not $DryRun -and -not $NoWait) {
            Wait-Collaboration -Context $context -Description "Analytics endpoint and healthy collaboration" `
                -TimeoutMinutes $TimeoutMinutes -Deadline $deadline -Condition {
                    param($c)
                    $analytics = $c.properties.workloads |
                        Where-Object { $_.workloadType -eq "Analytics" } | Select-Object -First 1
                    $analytics.endpoint -and $c.properties.health.healthState -eq "Ok"
                }
        }
    }
    "AddCollaborator" {
        foreach ($identity in $identities) {
            $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
            $existing = if (-not $DryRun) { Invoke-Collaboration -Context $context }
            $matches = @($existing.properties.collaborators | Where-Object {
                $_.userIdentifier -eq $identity.userIdentifier -or
                    ($identity.objectId -and $_.objectId -eq $identity.objectId -and
                        $_.tenantId -eq $identity.tenantId)
            })
            if (-not $matches.Count) {
                Invoke-Collaboration -Context $context -Method POST -Action addCollaborator `
                    -Body @{ collaborator = $identity } | Out-Null
            }
            if (-not $DryRun -and -not $NoWait) {
                Wait-Collaboration -Context $context -Description "collaborator $($identity.userIdentifier)" `
                    -TimeoutMinutes $TimeoutMinutes -Deadline $deadline -Condition {
                        param($c)
                        @($c.properties.collaborators | Where-Object {
                            $_.userIdentifier -eq $identity.userIdentifier -or
                                ($identity.objectId -and $_.objectId -eq $identity.objectId -and
                                    $_.tenantId -eq $identity.tenantId)
                        }).Count -gt 0
                    } | Out-Null
            }
        }
    }
}
