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
$image = "$acr/azure-cleanroom-samples/demos/${demo}:latest"

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
    --command "python3 ./app/host_model.py --model-path=/mnt/remote/fabrikam-model/onnx --data-path=/mnt/remote/contosso-data" `
    --datasources "fabrikam-model=/mnt/remote/fabrikam-model" `
        "contosso-data=/mnt/remote/contosso-data" `
    --ports 8000 `
    --cpu 1 `
    --memory 3

# Note: This will allow all incoming connections to the application.
# TODO: Add a policy to restrict traffic to the application.
az cleanroom config network http enable `
    --cleanroom-config $configResult.contractFragment `
    --direction inbound

Write-Log OperationCompleted `
    "Added application 'demoapp-$demo' ($image)."