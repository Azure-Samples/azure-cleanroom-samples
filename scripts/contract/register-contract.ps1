param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("cleanroomhello-job", "cleanroomhello-api", "analytics", "inference", "training")]
    [string]$demo,

    [string[]]$collaborators = ('litware', 'fabrikam', 'contosso'),

    [string]$persona = "$env:PERSONA",

    [string]$samplesRoot = "/home/samples",
    [string]$publicDir = "$samplesRoot/demo-resources/public",

    [string]$contractId = "collab-$demo-$((New-Guid).ToString().Substring(0, 8))",
    [string]$cleanroomConfig = "$publicDir/$contractId-cleanroom.config",
    [string]$cgsClient = "azure-cleanroom-samples-governance-client-$persona"
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../common/common.psm1

Write-Log OperationStarted `
    "Generating cleanroom specification for contract '$contractId' at '$cleanroomConfig'..."
az cleanroom config init `
    --cleanroom-config $cleanroomConfig

# Generate the cleanroom config which contains all the datasources, sinks and applications that are
# configured by the collaborators.
$azArgs = "cleanroom config view --cleanroom-config $cleanroomConfig --output-file $cleanroomConfig --configs "
foreach ($collaboratorName in $collaborators)
{
    $fragment = "$publicDir/$collaboratorName-$demo.config"
    Write-Log Verbose `
        "Adding fragment for '$collaboratorName' ('$fragment')..."
    $azArgs = $azArgs + "$fragment "
}

Start-Process az $azArgs -Wait
Write-Log OperationCompleted `
    "Generated cleanroom specification for contract '$contractId' at '$cleanroomConfig'." 

# Validate the contract structure before proposing.
az cleanroom config validate --cleanroom-config $cleanroomConfig

Write-Log OperationStarted `
    "Proposing contract '$contractId' to the consortium..." 

$data = Get-Content -Raw $cleanroomConfig
az cleanroom governance contract create `
    --data "$data" `
    --id $contractId `
    --governance-client $cgsClient

# Submitting a contract proposal.
$version = (az cleanroom governance contract show `
    --id $contractId `
    --query "version" `
    --output tsv `
    --governance-client $cgsClient)

az cleanroom governance contract propose `
    --version $version `
    --id $contractId `
    --query "proposalId" `
    --output tsv `
    --governance-client $cgsClient

Write-Log OperationCompleted `
    "Proposed contract for '$contractId' to the consortium." 