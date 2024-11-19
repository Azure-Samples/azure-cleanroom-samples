param(
    [Parameter(Mandatory = $true)]
    [string]$contractId,

    [ValidateSet("cached", "cached-debug", "generate", "generate-debug", "allow-all")]
    [string]$securityPolicy = "cached-debug",

    [string]$persona = "$env:PERSONA",

    [string]$samplesRoot = "/home/samples",
    [string]$privateDir = "$samplesRoot/demo-resources/private",
    [string]$artefactsDir = "$privateDir/$contractId-artefacts",
    [string]$cgsClient = "azure-cleanroom-samples-governance-client-$persona"
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../common/common.psm1

Write-Log OperationStarted `
    "Generating deployment artefacts for contract '$contractId' in '$artefactsDir'..." 

mkdir -p $artefactsDir
az cleanroom governance deployment generate `
    --contract-id $contractId `
    --governance-client $cgsClient `
    --security-policy-creation-option $securityPolicy `
    --output-dir $artefactsDir

Write-Log OperationStarted `
    "Proposing deployment artefacts for contract '$contractId' to the consortium..." 

az cleanroom governance deployment template propose `
    --template-file $artefactsDir/cleanroom-arm-template.json `
    --contract-id $contractId `
    --governance-client $cgsClient

az cleanroom governance deployment policy propose `
    --policy-file $artefactsDir/cleanroom-governance-policy.json `
    --contract-id $contractId `
    --governance-client $cgsClient

# Propose enabling CA cert for usage during cleanroom execution.
az cleanroom governance ca propose-enable `
    --contract-id $contractId `
    --governance-client $cgsClient

# Propose enabling log and telemetry collection during cleanroom execution.
az cleanroom governance contract runtime-option propose `
    --option logging `
    --action enable `
    --contract-id $contractId `
    --governance-client $cgsClient

az cleanroom governance contract runtime-option propose `
    --option telemetry `
    --action enable `
    --contract-id $contractId `
    --governance-client $cgsClient

Write-Log OperationCompleted `
    "Proposed deployment artefacts for contract '$contractId' to the consortium." 

