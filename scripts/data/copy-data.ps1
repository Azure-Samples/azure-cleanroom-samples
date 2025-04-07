param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("cleanroomhello-job", "cleanroomhello-api", "analytics", "inference", "training")]
    [string]$demo,
    [string]$persona = "$env:PERSONA",

    [string]$samplesRoot = "/home/samples",
    [string]$privateDir = "$samplesRoot/demo-resources/private",
    [string]$demosRoot = "$samplesRoot/demos",

    [string]$datastoreConfig = "$privateDir/datastores.config",
    [string]$datastoreDir = "$privateDir/datastores",
    [string]$datasinkPath = "$demosRoot/$demo/datasink/$persona"
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../common/common.psm1

Test-AzureAccessToken

if (Test-Path -Path $datasinkPath)
{
    $dirs = Get-ChildItem -Path $datasinkPath -Directory -Name
    foreach ($dir in $dirs)
    {
        $datastoreName = "$demo-$persona-$dir".ToLower()

        # TODO: Understand why this is being copied into a nested folder.
        az cleanroom datastore download `
            --name $datastoreName `
            --config $datastoreConfig `
            --dst $datastoreDir
        $dataDir = "$datastoreDir/$datastoreName"
        Write-Log OperationCompleted `
            "Downloaded data for datasink '$persona-$dir' ($datastoreName)" `
            "to '$dataDir'."
    }
}
else
{
    Write-Log Warning `
        "No datasink available for persona '$persona' in demo '$demo'."
}

