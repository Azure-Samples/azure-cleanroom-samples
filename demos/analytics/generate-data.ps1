param(
    [ValidateSet("fabrikam", "contosso")]
    [string]$persona = "$env:PERSONA",
    [string]$demo = "$(Split-Path $PSScriptRoot -Leaf)"
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../../scripts/common/common.psm1

if (-not (("fabrikam", "contosso") -contains $persona))
{

    Write-Log Warning `
        "No action required for persona '$persona' in demo '$demo'."
    return
}

# Use sample dataset at https://github.com/Azure-Samples/Synapse/tree/main/Data/Tweets
$src = "https://github.com/Azure-Samples/Synapse/raw/refs/heads/main/Data/Tweets"

if ("contosso" -eq $persona)
{
    $handles = ("BrigitMurtaughTweets", "FranmerMSTweets", "JeremyLiknessTweets")
}
else
{
    $handles = ("RahulPotharajuTweets", "MikeDoesBigDataTweets", "SQLCindyTweets")
}

$dataDir = "$PSScriptRoot/datasource/$persona/input"
Write-Log OperationStarted `
    "Downloading data for '$persona' in '$demo' demo from '$src'..."

foreach ($handle in $handles)
{
    Write-Log Verbose `
        "Downloading data for '$handle'..."
    curl -L "$src/$handle.csv" -o "$dataDir/$handle.csv"
}

Write-Log OperationCompleted `
    "Downloaded data for '$persona' in '$demo' demo to '$dataDir'."