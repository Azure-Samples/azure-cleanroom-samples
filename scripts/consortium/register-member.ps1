param(
    [string]$persona = "$env:PERSONA",

    [string[]]$collaborators = ('litware', 'fabrikam', 'contosso'),

    [string]$samplesRoot = "/home/samples",
    [string]$publicDir = "$samplesRoot/demo-resources/public",

    [string]$cgsClient = "azure-cleanroom-samples-governance-client-$persona"
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../common/common.psm1

Write-Log OperationStarted `
    "Adding '$collaborators' to the consortium..." 

foreach ($collaboratorName in $collaborators)
{
    # Makes a proposal for adding the new member.
    $proposalId = (az cleanroom governance member add `
        --certificate $publicDir/$($collaboratorName)_cert.pem `
        --identifier $collaboratorName `
        --tenant-id (Get-Content "$publicDir/$collaboratorName.tenantid") `
        --query "proposalId" `
        --output tsv `
        --governance-client $cgsClient)

    # Vote on the above proposal to accept the membership.
    az cleanroom governance proposal vote `
        --proposal-id $proposalId `
        --action accept `
        --governance-client $cgsClient

    Write-Log OperationCompleted `
        "Added '$collaboratorName' to the consortium."
}