param(
    [string]$demo = "$env:DEMO",

    [string]$persona = "$env:PERSONA",
    [string]$resourceGroup = "$env:RESOURCE_GROUP",
    [string]$kekName = $($($(New-Guid).Guid) -replace '-').ToLower(),

    [string]$samplesRoot = "/home/samples",
    [string]$publicDir = "$samplesRoot/demo-resources/public",
    [string]$privateDir = "$samplesRoot/demo-resources/private",
    [string]$demosRoot = "$samplesRoot/demos",
    [string]$governanceClient = "azure-cleanroom-samples-governance-client-$persona",

    [string]$secretstoreConfig = "$privateDir/secretstores.config",
    [string]$datastoreConfig = "$privateDir/datastores.config",
    [string]$datasourcePath = "$demosRoot/$demo/datasource/$persona",
    [string]$datasinkPath = "$demosRoot/$demo/datasink/$persona"
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../common/common.psm1

Write-Log OperationStarted `
    "Adding datasources and datasinks for '$persona' in the '$demo' demo"

az cleanroom collaboration context set `
    --collaboration-name $governanceClient

$contractId = Get-Content $publicDir/analytics.contract-id
if (Test-Path -Path $datasourcePath) {
    $dirs = Get-ChildItem -Path $datasourcePath -Directory -Name
    foreach ($dir in $dirs) {
        $datastoreName = "$demo-$persona-$dir".ToLower()
        $datasourceName = "$persona-$dir".ToLower()

        if ($persona -eq "woodgrove" -and $demo -eq "analytics-s3-sse") {
            az cleanroom collaboration dataset publish `
                --contract-id $contractId `
                --dataset-name $datasourceName `
                --datastore-name $datastoreName `
                --identity-name cleanroom_cgs_oidc `
                --policy-access-mode read `
                --policy-allowed-fields "date,author,mentions" `
                --datastore-config-file $datastoreConfig
        }
        else {
            az cleanroom collaboration dataset publish `
                --contract-id $contractId `
                --dataset-name $datasourceName `
                --datastore-name $datastoreName `
                --dek-secret-store-name $persona-dek-store `
                --kek-secret-store-name $persona-kek-store `
                --identity-name $persona-identity `
                --policy-access-mode read `
                --policy-allowed-fields "date,author,mentions" `
                --datastore-config-file $datastoreConfig `
                --secretstore-config-file $secretstoreConfig
        }

        Write-Log OperationCompleted `
            "Added datasource '$datasourceName' ($datastoreName)."

        $datasourceName | Out-File $publicDir/$datasourceName.dataset-id
    }
}
else {
    Write-Log Warning `
        "No datasource required for persona '$persona' in demo '$demo'."
}

if (Test-Path -Path $datasinkPath) {
    $dirs = Get-ChildItem -Path $datasinkPath -Directory -Name
    foreach ($dir in $dirs) {
        $datastoreName = "$demo-$persona-$dir".ToLower()
        $datasinkName = "$persona-$dir".ToLower()

        if ($persona -eq "woodgrove" -and $demo -eq "analytics-s3-sse") {
            az cleanroom collaboration dataset publish `
                --contract-id $contractId `
                --dataset-name $datasinkName `
                --datastore-name $datastoreName `
                --identity-name cleanroom_cgs_oidc `
                --policy-access-mode write `
                --policy-allowed-fields "author,Number_Of_Mentions" `
                --datastore-config-file $datastoreConfig
        }
        else {
            az cleanroom collaboration dataset publish `
                --contract-id $contractId `
                --dataset-name $datasinkName `
                --datastore-name $datastoreName `
                --dek-secret-store-name $persona-dek-store `
                --kek-secret-store-name $persona-kek-store `
                --identity-name $persona-identity `
                --policy-access-mode write `
                --policy-allowed-fields "author,Number_Of_Mentions" `
                --datastore-config-file $datastoreConfig `
                --secretstore-config-file $secretstoreConfig
        }

        Write-Log OperationCompleted `
            "Added datasink '$datasinkName' ($datastoreName)."

        $datasinkName | Out-File $publicDir/$datasinkName.dataset-id
    }
}
else {
    Write-Log Warning `
        "No datasink required for persona '$persona' in demo '$demo'."
}