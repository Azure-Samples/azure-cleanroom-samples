param(
    [Parameter(Mandatory = $true)]
    [string]$contractId,

    [string]$persona = "$env:PERSONA",

    [string]$cgsClient = "azure-cleanroom-samples-governance-client-$persona"
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../common/common.psm1

Write-Log OperationStarted `
    "Accepting deployment artefacts for '$contractId'..." 

# Vote on the proposed deployment template.
$proposalId = az cleanroom governance deployment template show `
    --contract-id $contractId `
    --governance-client $cgsClient `
    --query "proposalIds[0]" `
    --output tsv
Write-Log Verbose `
    "Accepting deployment template proposal '$proposalId'..."
az cleanroom governance proposal show-actions `
    --proposal-id $proposalId `
    --query "actions[0].args.spec.data" `
    --governance-client $cgsClient
# TODO: Logic to showcase template verification is pending.
az cleanroom governance proposal vote `
    --proposal-id $proposalId `
    --action accept `
    --governance-client $cgsClient
Write-Log OperationCompleted `
    "Accepted deployment template for '$contractId'."

# Vote on the proposed cce policy.
$proposalId = az cleanroom governance deployment policy show `
    --contract-id $contractId `
    --governance-client $cgsClient `
    --query "proposalIds[0]" `
    --output tsv
Write-Log Verbose `
    "Accepting deployment policy proposal '$proposalId'..."
az cleanroom governance proposal show-actions `
    --proposal-id $proposalId `
    --query "actions[0].args" `
    --governance-client $cgsClient
# TODO: Logic to showcase policy verification is pending.
az cleanroom governance proposal vote `
    --proposal-id $proposalId `
    --action accept `
    --governance-client $cgsClient
Write-Log OperationCompleted `
    "Accepted deployment policy for '$contractId'."

# Vote on the proposed CA enable.
$proposalId = az cleanroom governance ca show `
    --contract-id $contractId `
    --governance-client $cgsClient `
    --query "proposalIds[0]" `
    --output tsv
Write-Log Verbose `
    "Accepting CA proposal '$proposalId'..."
az cleanroom governance proposal vote `
    --proposal-id $proposalId `
    --action accept `
    --governance-client $cgsClient
Write-Log OperationCompleted `
    "Accepted CA for '$contractId'."

# Vote on the enable logging proposal.
$proposalId = az cleanroom governance contract runtime-option get `
    --option logging `
    --contract-id $contractId `
    --governance-client $cgsClient `
    --query "proposalIds[0]" `
    --output tsv
Write-Log Verbose `
    "Accepting enable logging proposal '$proposalId'..."
az cleanroom governance proposal vote `
    --proposal-id $proposalId `
    --action accept `
    --governance-client $cgsClient
Write-Log OperationCompleted `
    "Accepted enabling application telemetry for '$contractId'."

# Vote on the enable telemetry proposal.
$proposalId = az cleanroom governance contract runtime-option get `
    --option telemetry `
    --contract-id $contractId `
    --governance-client $cgsClient `
    --query "proposalIds[0]" `
    --output tsv
Write-Log Verbose `
    "Accepting enable telemetry proposal '$proposalId'..."
az cleanroom governance proposal vote `
    --proposal-id $proposalId `
    --action accept `
    --governance-client $cgsClient
Write-Log OperationCompleted `
    "Accepted enabling infrastructure telemetry for '$contractId'."

Write-Log OperationStarted `
    "Accepting documents for '$contractId'..." 

Write-Log Verbose `
    "Enumerating all documents..."
$documentIds = @(az cleanroom governance document show `
    --governance-client $cgsClient `
    --query "[*].id" `
    --output json | ConvertFrom-Json)

foreach ($documentId in $documentIds)
{
    Write-Log Verbose `
        "Fetching document '$documentId'..."
    $document = (az cleanroom governance document show `
        --id $documentId `
        --governance-client $cgsClient `
        --output json | ConvertFrom-Json)

    if ($contractId -eq $document.contractId)
    {
        Write-Log Verbose `
            "Document '$documentId' for contract '$contractId' is in state '$($document.state)':"
        Write-Log Information `
            "$($document.data)"
        az cleanroom governance document vote `
            --id $documentId `
            --proposal-id $document.proposalId `
            --action accept `
            --governance-client $cgsClient

        Write-Log OperationCompleted `
            "Accepted document '$documentId' for contract '$contractId'."
    }
    else
    {
        Write-Log Verbose `
            "Skipped '$documentId' for contract '$($document.contractId)'."
    }
}