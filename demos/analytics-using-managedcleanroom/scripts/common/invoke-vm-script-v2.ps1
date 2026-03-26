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
    [switch]$async
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

# Read script content
$fullScriptPath = Join-Path $scriptDir ".." $scriptPath
if (-not (Test-Path $fullScriptPath)) {
    throw "Script not found at: $fullScriptPath"
}

$scriptContent = Get-Content $fullScriptPath -Raw

# Convert scriptArgs hashtable to PowerShell parameter string
$argsString = ""
foreach ($key in $scriptArgs.Keys) {
    $value = $scriptArgs[$key]
    # Escape single quotes for PowerShell (double them)
    $escapedValue = $value -replace "'", "''"
    $argsString += "-$key '$escapedValue' "
}
$argsString = $argsString.TrimEnd()

# Create temp script file on VM that includes the script content
$tempScriptName = "/tmp/script_$(Get-Random).ps1"
$logFile = "/tmp/script_$(Get-Random).log"
$remoteCommand = @"
cat > $tempScriptName << 'EOFSCRIPT'
$scriptContent
EOFSCRIPT
chmod +x $tempScriptName
echo "[SCRIPT START] \$(date)" | tee -a $logFile
pwsh $tempScriptName $argsString 2>&1 | tee -a $logFile
EXIT_CODE=\$?
echo "[SCRIPT END] Exit code: \$EXIT_CODE, Time: \$(date)" | tee -a $logFile
rm -f $tempScriptName
exit \$EXIT_CODE
"@

Write-Host "Executing script on VM $vmName via run-command..." -ForegroundColor Cyan
Write-Host "Script: $scriptPath" -ForegroundColor Gray
if ($argsString) {
    Write-Host "Arguments: $argsString" -ForegroundColor Gray
}
Write-Host "Timeout: $timeout seconds" -ForegroundColor Gray
Write-Host "" 

# Show progress indicator
$progressJob = Start-Job -ScriptBlock {
    param($vmName, $duration)
    $elapsed = 0
    while ($elapsed -lt $duration) {
        Start-Sleep -Seconds 10
        $elapsed += 10
        Write-Host "  ⏱️  Waiting for VM command... ${elapsed}s elapsed" -ForegroundColor DarkGray
    }
} -ArgumentList $vmName, $timeout

try {
    # Execute via az vm run-command with timeout
    $startTime = Get-Date
    
    $result = az vm run-command invoke `
        --resource-group $resourceGroup `
        --name $vmName `
        --command-id RunShellScript `
        --scripts $remoteCommand `
        --query "value[0].message" `
        --output tsv
    
    $elapsed = ((Get-Date) - $startTime).TotalSeconds
    
    # Stop progress indicator
    Stop-Job $progressJob -ErrorAction SilentlyContinue
    Remove-Job $progressJob -ErrorAction SilentlyContinue
    
    Write-Host ""
    Write-Host "Command completed in $([math]::Round($elapsed, 1)) seconds" -ForegroundColor Cyan
    
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to execute script on VM $vmName (exit code: $LASTEXITCODE)"
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
        
        Write-Host ""
        Write-Host "Script execution completed on VM $vmName" -ForegroundColor Green
    } else {
        throw "Unexpected output from run-command: $result"
    }
} catch {
    # Stop progress indicator on error
    Stop-Job $progressJob -ErrorAction SilentlyContinue
    Remove-Job $progressJob -ErrorAction SilentlyContinue
    throw
}
