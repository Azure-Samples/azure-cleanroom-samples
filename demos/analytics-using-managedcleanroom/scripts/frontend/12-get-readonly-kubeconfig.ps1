<#
.SYNOPSIS
    Step 12.1 [OWNER] - Get the read-only kubeconfig for Grafana access.

.DESCRIPTION
    Calls the ARM *action* getReadonlyKubeConfig on the collaboration and writes
    a decoded kubeconfig to disk. Unlike Steps 03-11 (frontend/dataplane), this
    is an ARM control-plane action, so it uses `az rest` against management.azure.com
    and needs the ARM collaboration coordinates (resource group + name).

    After this, open Grafana with the existing demo script:
        ./scripts/12-open-grafana-dashboard.ps1 -KubeConfigPath ./readonly.kubeconfig

    Mirrors README-API.md Step 12.1 "Get Readonly Kubeconfig".

.PARAMETER resourceGroup
    Resource group of the collaboration ARM resource ($collabRg).

.PARAMETER collaborationName
    ARM collaboration resource name ($collabName).

.PARAMETER OutFile
    Path to write the decoded kubeconfig (default: ./readonly.kubeconfig).

.PARAMETER apiVersion
    Microsoft.CleanRoom ARM API version (default: 2026-04-30-preview).

.EXAMPLE
    ./12-get-readonly-kubeconfig.ps1 -resourceGroup cr-collab-rg -collaborationName collab1
#>
param(
    [Parameter(Mandatory)][string]$resourceGroup,
    [Parameter(Mandatory)][string]$collaborationName,
    [string]$OutFile = "./readonly.kubeconfig",
    [string]$apiVersion = "2026-04-30-preview",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$armEndpoint = "https://management.azure.com"

$subscription = (az account show -o json | ConvertFrom-Json).id
$collabArmUrl = "$armEndpoint/subscriptions/$subscription/resourceGroups/$resourceGroup/providers/Microsoft.CleanRoom/Collaborations/$collaborationName"

if ($DryRun) {
    Write-Host "[DRY-RUN] POST $collabArmUrl/getReadonlyKubeConfig?api-version=$apiVersion"
    Write-Host "[DRY-RUN] Would decode kubeconfig and write to $OutFile"
    return
}

$kc = az rest --method POST `
    --url "$collabArmUrl/getReadonlyKubeConfig`?api-version=$apiVersion" `
    --resource $armEndpoint -o json | ConvertFrom-Json

$bytes = [Convert]::FromBase64String($kc.kubeconfig)
[System.Text.Encoding]::UTF8.GetString($bytes) | Out-File $OutFile -Encoding utf8

Write-Host "Read-only kubeconfig written to $OutFile"
Write-Host "Next: ./scripts/12-open-grafana-dashboard.ps1 -KubeConfigPath $OutFile"
