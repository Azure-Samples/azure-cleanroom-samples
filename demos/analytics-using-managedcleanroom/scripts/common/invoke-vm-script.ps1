param(
    [Parameter(Mandatory=$true)]
    [string]$vmName,
    
    [Parameter(Mandatory=$true)]
    [string]$scriptPath,
    
    [Parameter(Mandatory=$false)]
    [hashtable]$scriptArgs = @{},
    
    [Parameter(Mandatory=$false)]
    [int]$timeout = 600,
    
    [Parameter(Mandatory=$false)]
    [switch]$showLiveOutput,

    [Parameter(Mandatory=$false)]
    [switch]$skipSync
)

$ErrorActionPreference = "Stop"

# Read VM metadata - extract short name from full VM name
$scriptDir = Split-Path -Parent $PSCommandPath
$shortName = if ($vmName -match "cleanroom-e2e-(\w+)-vm") { $Matches[1] } else { $vmName }
$metadataPath = Join-Path $scriptDir "../generated/vms/$shortName-vm.json"
if (-not (Test-Path $metadataPath)) {
    throw "VM metadata not found at: $metadataPath"
}

$metadata = Get-Content $metadataPath | ConvertFrom-Json
$resourceGroup = $metadata.resourceGroup

# Resolve scripts root directory (parent of common/)
$scriptsRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path

# Verify the target script exists locally
$fullScriptPath = Join-Path $scriptsRoot $scriptPath
if (-not (Test-Path $fullScriptPath)) {
    throw "Script not found at: $fullScriptPath"
}

$remoteScriptsDir = "/opt/cleanroom-scripts"

# Step 1: Sync scripts to VM if not skipped
# We sync the scripts directory so $PSScriptRoot references work correctly
if (-not $skipSync) {
    Write-Host "Syncing scripts to VM $vmName..." -ForegroundColor Cyan
    
    # Get all .ps1 files we need (workflow + common scripts)
    $filesToSync = @()
    
    # Add the target script
    $filesToSync += $fullScriptPath
    
    # Add all common/ scripts (always needed)
    $commonDir = Join-Path $scriptsRoot "common"
    if (Test-Path $commonDir) {
        $filesToSync += Get-ChildItem -Path $commonDir -Filter "*.ps1" | Select-Object -ExpandProperty FullName
    }
    
    # Build sync script using base64 encoding for robustness.
    # Each file is base64-encoded and decoded on the VM, avoiding heredoc escaping issues.
    $syncParts = @()
    $syncParts += "#!/bin/bash"
    $syncParts += "set -e"
    $syncParts += "mkdir -p '$remoteScriptsDir/common'"
    
    $fileCount = 0
    foreach ($file in $filesToSync) {
        $relativePath = $file.Substring($scriptsRoot.Length + 1)
        $remoteFile = "$remoteScriptsDir/$relativePath"
        $remoteDir = [System.IO.Path]::GetDirectoryName($remoteFile) -replace '\\', '/'
        $contentBytes = [System.IO.File]::ReadAllBytes($file)
        $b64 = [Convert]::ToBase64String($contentBytes)
        
        $syncParts += "mkdir -p '$remoteDir'"
        $syncParts += "echo '$b64' | base64 -d > '$remoteFile'"
        $fileCount++
    }
    
    $syncParts += "echo 'Synced $fileCount scripts to $remoteScriptsDir'"
    
    $syncScript = $syncParts -join "`n"
    
    # Write sync script to temp file to avoid command-line length issues
    $tempSyncFile = Join-Path ([System.IO.Path]::GetTempPath()) "sync_$(Get-Random).sh"
    $syncScript | Out-File -FilePath $tempSyncFile -Encoding utf8 -NoNewline
    
    try {
        $fileRef = "@$tempSyncFile"
        $syncResult = az vm run-command invoke `
            --resource-group $resourceGroup `
            --name $vmName `
            --command-id RunShellScript `
            --scripts $fileRef `
            --query "value[0].message" `
            --output tsv 2>&1
        
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Script sync may have issues: $syncResult"
        } else {
            # Extract just success message
            if ($syncResult -match "Synced \d+ scripts") {
                Write-Host "  $($Matches[0])" -ForegroundColor Green
            } else {
                Write-Host "  Scripts synced." -ForegroundColor Green
            }
        }
    } finally {
        Remove-Item -Path $tempSyncFile -Force -ErrorAction SilentlyContinue
    }
}

# Step 2: Execute the target script from the synced location
$argsString = ""
foreach ($key in $scriptArgs.Keys) {
    $value = $scriptArgs[$key]
    $escapedValue = $value -replace "'", "''"
    $argsString += "-$key '$escapedValue' "
}
$argsString = $argsString.TrimEnd()

$remoteScriptPath = "$remoteScriptsDir/$scriptPath"

Write-Host "Executing $scriptPath on VM $vmName..." -ForegroundColor Cyan
if ($argsString) {
    Write-Host "Arguments: $argsString" -ForegroundColor Gray
}

    # Write execution command to a temp script file so quoting is preserved.
    # cd into the remote scripts dir so relative paths (e.g. ./generated) are consistent.
    $execScript = "#!/bin/bash`nset -e`ncd '$remoteScriptsDir'`npwsh -File '$remoteScriptPath' $argsString"
    $tempExecFile = Join-Path ([System.IO.Path]::GetTempPath()) "exec_$(Get-Random).sh"
    $execScript | Out-File -FilePath $tempExecFile -Encoding utf8 -NoNewline

    try {
        # Execute via az vm run-command using file reference
        $fileRef = "@$tempExecFile"
        $result = az vm run-command invoke `
            --resource-group $resourceGroup `
            --name $vmName `
            --command-id RunShellScript `
            --scripts $fileRef `
            --query "value[0].message" `
            --output tsv

        if ($LASTEXITCODE -ne 0) {
            throw "Failed to execute script on VM $vmName (exit code: $LASTEXITCODE)"
        }
    } finally {
        Remove-Item -Path $tempExecFile -Force -ErrorAction SilentlyContinue
    }

# Parse result
if ($result -match "Enable succeeded:") {
    $lines = $result -split "`n"
    $inStdout = $false
    $inStderr = $false
    $stdoutLines = @()
    $stderrLines = @()
    
    foreach ($line in $lines) {
        if ($line -match "^\[stdout\]") {
            $inStdout = $true
            $inStderr = $false
            continue
        }
        if ($line -match "^\[stderr\]") {
            $inStdout = $false
            $inStderr = $true
            continue
        }
        
        if ($inStdout -and $line.Trim() -ne "") {
            $stdoutLines += $line
        }
        if ($inStderr -and $line.Trim() -ne "") {
            $stderrLines += $line
        }
    }
    
    # Output results
    if ($stdoutLines.Count -gt 0) {
        $stdoutLines | ForEach-Object { Write-Host $_ }
    }
    
    if ($stderrLines.Count -gt 0) {
        $stderrLines | ForEach-Object { Write-Warning $_ }
    }
    
    Write-Host "Script execution completed on VM $vmName" -ForegroundColor Green
} else {
    throw "Unexpected output from run-command: $result"
}
