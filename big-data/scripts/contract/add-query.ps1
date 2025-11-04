param(
    [string]$demo = "$env:DEMO",
    [string]$persona = "$env:PERSONA",

    [string]$samplesRoot = "/home/samples",
    [string]$publicDir = "$samplesRoot/demo-resources/public",
    [string]$privateDir = "$samplesRoot/demo-resources/private",
    [string]$demosRoot = "$samplesRoot/demos",
    [string]$queryPath = "$demosRoot/$demo/query/$persona",
    [string]$cgsClient = "azure-cleanroom-samples-governance-client-$persona"
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../common/common.psm1

Write-Log OperationStarted `
    "Creating query documents in CCF for '$persona' in the '$demo' demo..."

az cleanroom collaboration context set `
    --collaboration-name $cgsClient

$instanceId = (New-Guid).ToString().Substring(0, 8)
if (Test-Path -Path $queryPath) {
    $dirs = Get-ChildItem -Path $queryPath -Directory -Name
    foreach ($dir in $dirs) {
        $queryName = "$("$persona-$dir".ToLower())-$instanceId"
        $contractId = Get-Content $publicDir/analytics.contract-id

        write-Log Verbose `
            "Publishing query document '$queryName'"

        az cleanroom collaboration spark-sql publish `
            --application-name $queryName `
            --application-query $queryPath/$dir/segmentedQuery.yaml `
            --application-input-dataset "publisher_data:$(Get-Content "$publicDir/northwind-input.dataset-id"), consumer_data:$(Get-Content "$publicDir/woodgrove-input.dataset-id")" `
            --application-output-dataset "datasink:$(Get-Content "$publicDir/woodgrove-output.dataset-id")" `
            --contract-id $contractId
        $queryName | Out-File $publicDir/analytics.query-id
        Write-Log OperationCompleted `
            "Query document '$queryName' is proposed in CCF. ProposalId: $proposalId."
    }
}
else {
    Write-Log Warning `
        "No query specified for persona '$persona' in demo '$demo'."
}

