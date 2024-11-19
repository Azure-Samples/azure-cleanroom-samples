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

$configResult = Get-Content $contractConfig | ConvertFrom-Json
Write-Log OperationStarted `
    "Adding application details for '$persona' in the '$demo' demo to" `
    "'$($configResult.contractFragment)'..."

#
# Use inline code instead of packaging application into a container.
#
$image = "docker.io/golang@sha256:f43c6f049f04cbbaeb28f0aad3eea15274a7d0a7899a617d0037aec48d7ab010"
$inline_code = $(cat $PSScriptRoot/application/main.go | base64 -w 0)
az cleanroom config add-application `
    --cleanroom-config $configResult.contractFragment `
    --name demoapp-$demo `
    --image $image `
    --command "bash -c 'echo `$CODE | base64 -d > main.go; go run main.go'" `
    --mounts "src=fabrikam-input,dst=/mnt/remote/fabrikam-input" `
             "src=fabrikam-output,dst=/mnt/remote/fabrikam-output" `
    --env-vars OUTPUT_LOCATION=/mnt/remote/fabrikam-output `
               INPUT_LOCATION=/mnt/remote/fabrikam-input `
               CODE="$inline_code" `
    --cpu 0.5 `
    --memory 4

Write-Log OperationCompleted `
    "Added application 'demoapp-$demo' ($image)."