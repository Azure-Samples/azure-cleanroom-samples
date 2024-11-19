param(
    [string]$persona = "$env:PERSONA",

    [string]$samplesRoot = "/home/samples",
    [string]$secretDir = "$samplesRoot/demo-resources/secret",
    [string]$publicDir = "$samplesRoot/demo-resources/public",

    [string]$cgsClient = "azure-cleanroom-samples-governance-client-$persona"
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../common/common.psm1

$ccfEndpoint = (Get-Content "$publicDir/ccfEndpoint.json" | ConvertFrom-Json)
Write-Log OperationStarted `
    "Joining consortium '$($ccfEndpoint.name)' ($($ccfEndpoint.url))..."

# Deploy client-side containers to interact with the governance service as the new member.
az cleanroom governance client deploy `
    --ccf-endpoint $($ccfEndpoint.url) `
    --signing-cert $secretDir/$($persona)_cert.pem `
    --signing-key $secretDir/$($persona)_privk.pem `
    --service-cert "$publicDir/$($ccfEndpoint.serviceCert)" `
    --name $cgsClient

# Accept the invitation and becomes an active member in the consortium.
az cleanroom governance member activate --governance-client $cgsClient

Write-Log OperationCompleted `
    "Joined consortium '$($ccfEndpoint.name)' ($($ccfEndpoint.url)) and deployed CGS client '$cgsClient'."
