param(
    [Parameter(Mandatory = $true)]
    [string]$contractId,

    [string]$persona = "$env:PERSONA",

    [string]$cgsClient = "azure-cleanroom-samples-governance-client-$persona"
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../common/common.psm1

Write-Log OperationStarted `
    "Fetching audit events for contract '$contractId'..."
$auditEvents = (az cleanroom governance contract event list `
    --contract-id $contractId `
    --all `
    --query 'value' `
    --governance-client $cgsClient ) | ConvertFrom-Json

if ($auditEvents.Count -eq 0)
{
    Write-Log Warning `
        "No audit events available for contract '$contractId'."
    return
}

Write-Log Verbose `
    "-----BEGIN AUDIT EVENTS-----$([environment]::NewLine)" `
    "$($PSStyle.Reset)"
foreach ($event in $auditEvents)
{
    Write-Log Information `
        "[$($event.timestamp_iso)][$($event.data.source)]"
    Write-Log Warning `
        "$($event.data.message)$([environment]::NewLine)"
}
Write-Log Verbose `
    "-----END AUDIT EVENTS-----"

