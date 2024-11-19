param(
    [ValidateSet("litware")]
    [string]$persona = "$env:PERSONA",
    [string]$resourceGroup = "$env:RESOURCE_GROUP",

    [string]$samplesRoot = "/home/samples",
    [string]$privateDir = "$samplesRoot/demo-resources/private",

    [string]$demo = "$(Split-Path $PSScriptRoot -Leaf)",
    [string]$environmentConfig = "$privateDir/$resourceGroup.generated.json",
    [string]$contractConfig = "$privateDir/$resourceGroup-$demo.generated.json"
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../../scripts/common/common.psm1

if (-not (("litware") -contains $persona))
{
    Write-Log Warning `
        "No action required for persona '$persona' in demo '$demo'."
    return
}

$environmentConfigResult = Get-Content $environmentConfig | ConvertFrom-Json
$acrName = $environmentConfigResult.acr.name
$acr = $environmentConfigResult.acr.loginServer
$image = "$acr/azure-cleanroom-samples/demos/$demo"

#
# Build application if required.
#
& {
    Write-Log OperationStarted `
        "Generating application container for '$demo'..." 

    Write-Log Verbose `
        "Building application container for '$demo''..." 
    docker build -f $PSScriptRoot/application/Dockerfile.$demo -t $image $PSScriptRoot/application

    Write-Log Verbose `
        "Publishing application container for '$demo' to '$acr'..."
    az acr login --name $acrName
    docker push $image

    Write-Log Verbose `
        "Published application container '$image'." 
}

$configResult = Get-Content $contractConfig | ConvertFrom-Json
Write-Log OperationStarted `
    "Adding application details for '$persona' in the '$demo' demo to" `
    "'$($configResult.contractFragment)'..."

az cleanroom config add-application `
    --cleanroom-config $configResult.contractFragment `
    --name demoapp-$demo `
    --image $image `
    --command "python3.10 ./analytics.py" `
    --mounts "src=fabrikam-input,dst=/mnt/remote/fabrikam-input" `
             "src=contosso-input,dst=/mnt/remote/contosso-input" `
    --env-vars STORAGE_PATH_1=/mnt/remote/fabrikam-input `
               STORAGE_PATH_2=/mnt/remote/contosso-input `
    --cpu 0.5 `
    --memory 4

az cleanroom config add-application-endpoint `
    --cleanroom-config $configResult.contractFragment `
    --application-name demoapp-$demo `
    --port 8310

Write-Log OperationCompleted `
    "Added application 'demoapp-$demo' ($image)."