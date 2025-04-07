param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("cleanroomhello-job", "cleanroomhello-api", "analytics", "inference", "training")]
    [string]$demo,

    [Parameter(Mandatory = $true)]
    [string]$contractId,

    [string]$persona = "$env:PERSONA",
    [string]$resourceGroup = "$env:RESOURCE_GROUP",

    [string]$cleanRoomName = "cleanroom-$contractId",

    [string]$samplesRoot = "/home/samples",
    [string]$privateDir = "$samplesRoot/demo-resources/private",
    [string]$publicDir = "$samplesRoot/demo-resources/public",
    [string]$telemetryDir = "$samplesRoot/demo-resources/telemetry",

    [string]$cleanroomEndpoint = (Get-Content "$publicDir/$cleanRoomName.endpoint"),

    [string]$contractConfig = "$privateDir/$resourceGroup-$demo.generated.json",
    [string]$datastoreConfig = "$privateDir/datastores.config",
    [string]$dashboardName = "azure-cleanroom-samples-telemetry"
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../common/common.psm1

$infrastructureDir = "$telemetryDir/infrastructure-telemetry/$contractId"
$applicationDir = "$telemetryDir/application-telemetry/$contractId"

if ($cleanroomEndpoint -eq '')
{
    Write-Log Warning `
        "No endpoint details available for cleanroom '$cleanRoomName' at" `
        "'$publicDir/$cleanRoomName.endpoint'."
    return
}

Write-Log Information "Exporting logs..."
$response = curl -X POST -s -k https://${cleanroomEndpoint}:8200/gov/exportLogs
$expectedResponse = '{"message":"Application telemetry data exported successfully."}'
if ($response -ne $expectedResponse) {
    Write-Log Critical "Did not get expected response. Received: $response."
    exit 1
}

Write-Log Information "Exporting telemetry..."
$response = curl -X POST -s -k https://${cleanroomEndpoint}:8200/gov/exportTelemetry
$expectedResponse = '{"message":"Infrastructure telemetry data exported successfully."}'
if ($response -ne $expectedResponse) {
    Write-Log Critical "Did not get expected response. Received: $response."
    exit 1
}

if ("litware" -eq $persona)
{
    Test-AzureAccessToken

    $contractConfigResult = Get-Content $contractConfig | ConvertFrom-Json

    #
    # Download infrastructure telemetry.
    #
    Write-Log OperationStarted `
        "Downloading infrastructure telemetry to '$infrastructureDir'."
    mkdir -p $infrastructureDir
    az cleanroom telemetry download `
        --cleanroom-config $contractConfigResult.contractFragment `
        --datastore-config $datastoreConfig `
        --target-folder $infrastructureDir
    # TODO: Fetch exact name of the backing datastore from the clean room spec.
    $dataDir = "$infrastructureDir/infrastructure-telemetry-*"
    Write-Log OperationCompleted `
        "Downloaded infrastructure telemetry to '$dataDir'."

    #
    # Download application telemetry.
    #
    Write-Log OperationStarted `
        "Downloading application telemetry to '$applicationDir'."
    mkdir -p $applicationDir
    az cleanroom logs download `
        --cleanroom-config $contractConfigResult.contractFragment `
        --datastore-config $datastoreConfig `
        --target-folder $applicationDir
    # TODO: Fetch exact name of the backing datastore from the clean room spec.
    $dataDir = "$applicationDir/application-telemetry-*"
    Write-Log OperationCompleted `
        "Downloaded application telemetry to '$dataDir'."
}

# Display application logs.
if (Test-Path $applicationDir/application-telemetry-*/**/demoapp-$demo.log)
{
    Write-Log Warning `
        "$([environment]::NewLine)Application logs:"
    Write-Log Verbose `
        "-----BEGIN OUTPUT-----" `
        "$($PSStyle.Reset)"
    cat $applicationDir/application-telemetry-*/**/demoapp-$demo.log
    Write-Log Verbose `
        "-----END OUTPUT-----"
}
else
{
    Write-Log Warning `
        "$([environment]::NewLine)No application logs are available."
}

# Display dashboard details.
$dashboardUrl = docker compose -p $dashboardName port "aspire" 18888
$dashboardPort = ($dashboardUrl -split ":")[1]
Write-Log Warning `
    "$([environment]::NewLine)Open telemetry dashboard at http://localhost:$dashboardPort" `
    "to view infrastructure telemetry."
