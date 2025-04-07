param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("cleanroomhello-job", "cleanroomhello-api", "analytics", "inference", "training")]
    [string]$demo,

    [string]$persona = "$env:PERSONA",
    [string]$resourceGroup = "$env:RESOURCE_GROUP",

    [string]$samplesRoot = "/home/samples",
    [string]$privateDir = "$samplesRoot/demo-resources/private",
    [string]$secretDir = "$samplesRoot/demo-resources/secret",
    [string]$demosRoot = "$samplesRoot/demos",
    [string]$sa = "",

    [string]$environmentConfig = "$privateDir/$resourceGroup.generated.json",
    [string]$secretstoreConfig = "$privateDir/secretstores.config",
    [string]$datastoreConfig = "$privateDir/datastores.config",
    [string]$datastoreDir = "$privateDir/datastores",
    [string]$demoPath = "$demosRoot/$demo",
    [string]$datasourcePath = "$demoPath/datasource/$persona",
    [string]$datasinkPath = "$demoPath/datasink/$persona"
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../common/common.psm1

Test-AzureAccessToken

if ($sa -eq "")
{
    $initResult = Get-Content $environmentConfig | ConvertFrom-Json
    $sa = $initResult.datasa.id
}

if (Test-Path -Path "$demoPath/generate-data.ps1")
{
    & $demoPath/generate-data.ps1
}

Write-Log OperationStarted `
    "Creating data stores for '$demo' demo in '$sa'..."

if (Test-Path -Path $datasourcePath)
{
    $dirs = Get-ChildItem -Path $datasourcePath -Directory -Name
    foreach ($dir in $dirs)
    {
        $datastoreName = "$demo-$persona-$dir".ToLower()
        Write-Log Verbose `
            "Enumerated datasink '$datastoreName' in '$datasourcePath'..."

        az cleanroom datastore add `
            --name $datastoreName `
            --config $datastoreConfig `
            --secretstore-config $secretStoreConfig `
            --secretstore $persona-local-store `
            --encryption-mode CPK `
            --backingstore-type Azure_BlobStorage `
            --backingstore-id $sa
        $datastorePath = "$datastoreDir/$datastoreName"
        mkdir -p $datastorePath
        Write-Log OperationCompleted `
            "Created data store '$datastoreName' backed by '$sa'."

        cp -r $datasourcePath/$dir/* $datastorePath
        az cleanroom datastore upload `
            --name $datastoreName `
            --config $datastoreConfig `
            --src $datastorePath
        Write-Log OperationCompleted `
            "Published data from '$datasourcePath/$dir' as data store '$datastoreName'."
    }
}
else
{
    Write-Log Warning `
        "No datasource available for persona '$persona' in demo '$demo'."
}

if (Test-Path -Path $datasinkPath)
{
    $dirs = Get-ChildItem -Path $datasinkPath -Directory -Name
    foreach ($dir in $dirs)
    {
        $datastoreName = "$demo-$persona-$dir".ToLower()
        Write-Log Verbose `
            "Enumerated datasink '$datastoreName' in '$datasinkPath'..."

        az cleanroom datastore add `
            --name $datastoreName `
            --config $datastoreConfig `
            --secretstore-config $secretStoreConfig `
            --secretstore $persona-local-store `
            --encryption-mode CPK `
            --backingstore-type Azure_BlobStorage `
            --backingstore-id $sa
        $datastorePath = "$datastoreDir/$datastoreName"
        mkdir -p $datastorePath
        Write-Log OperationCompleted `
            "Created data store '$datastoreName' backed by '$sa'."
    }
}
else
{
    Write-Log Warning `
        "No datasink available for persona '$persona' in demo '$demo'."
}
