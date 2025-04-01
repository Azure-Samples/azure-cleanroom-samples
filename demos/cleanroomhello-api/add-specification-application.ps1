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

#$image = "$acr/azure-cleanroom-samples/demos/$demo"
$policy = "$acr/azure-cleanroom-samples/demos/$demo-policy:latest"
$image = "docker.io/nginxdemos/nginx-hello:plain-text@sha256:d976f016b32fc381dfb74119cc421d42787b5a63a6b661ab57891b7caa5ad12e"

& {
    Write-Log OperationStarted `
        "Generating application endpoint policy for the '$demo' demo..."

    Push-Location
    Set-Location $PSScriptRoot/policy
    Write-Log Verbose `
        "Building OPA policy bundle for '$demo''..." 
    opa build . --bundle -o $privateDir/$demo-policy.tar.gz
    Pop-Location

    Write-Log Verbose `
        "Performing ORAS login on '$acr' using az acr token..." 
    $u ="00000000-0000-0000-0000-000000000000"
    $p =$(az acr login --name $acrName --expose-token --output tsv --query accessToken)
    oras login --username $u --password $p $acr

    # Push the bundle to the registry. Need to Set-Location as need to use "./*-policy.tar.gz"
    # as the path in the orash push command. If giving a path like /some/dir/*-policy.tar.gz
    # then oras pull fails with "Error: failed to resolve path for writing: path traversal disallowed"
    Push-Location
    Set-Location $privateDir
    Write-Log Verbose `
        "Pushing OPA policy bundle to '$acr''..." 
    oras push $policy `
        --config $PSScriptRoot/policy/config.json:application/vnd.oci.image.config.v1+json `
        ./$demo-policy.tar.gz:application/vnd.oci.image.layer.v1.tar+gzip
    Pop-Location

    Write-Log OperationStarted `
        "Generated application endpoint policy for the '$demo' demo at '$policy'."
}

$configResult = Get-Content $contractConfig | ConvertFrom-Json
Write-Log OperationStarted `
    "Adding application details for '$persona' in the '$demo' demo to" `
    "'$($configResult.contractFragment)'..."

az cleanroom config add-application `
    --cleanroom-config $configResult.contractFragment `
    --name demoapp-$demo `
    --image $image `
    --cpu 0.5 `
    --ports 8080 `
    --memory 4

az cleanroom config network http enable `
    --cleanroom-config $configResult.contractFragment `
    --direction inbound `
    --policy $policy

Write-Log OperationCompleted `
    "Added application 'demoapp-$demo' ($image)."