param(
    [Parameter(Mandatory=$true)]
    [string]$vmName,
    
    [Parameter(Mandatory=$false)]
    [string]$sshKeyPath = "demos/analytics-using-managedcleanroom/scripts/generated/vms/cleanroom-e2e-ssh-key",
    
    [Parameter(Mandatory=$false)]
    [string]$localScriptPath = "demos/analytics-using-managedcleanroom/scripts"
)

$ErrorActionPreference = "Stop"

Write-Host "Uploading scripts to VM: $vmName"

# Read VM metadata JSON file
$vmMetadataPath = "demos/analytics-using-managedcleanroom/scripts/generated/vms/$vmName.json"
if (-not (Test-Path $vmMetadataPath)) {
    throw "VM metadata file not found: $vmMetadataPath"
}

$vmMetadata = Get-Content $vmMetadataPath | ConvertFrom-Json
$publicIp = $vmMetadata.publicIp

if ([string]::IsNullOrEmpty($publicIp)) {
    throw "Public IP not found in VM metadata"
}

Write-Host "VM Public IP: $publicIp"

# Verify SSH key exists
if (-not (Test-Path $sshKeyPath)) {
    throw "SSH key not found: $sshKeyPath"
}

# Verify local script path exists
if (-not (Test-Path $localScriptPath)) {
    throw "Local script path not found: $localScriptPath"
}

# Create remote directories
Write-Host "Creating remote directories..."
$createDirCmd = "mkdir -p /home/azureuser/cleanroom-e2e/scripts /home/azureuser/cleanroom-e2e/generated"
ssh -i $sshKeyPath -o StrictHostKeyChecking=no azureuser@$publicIp $createDirCmd

if ($LASTEXITCODE -ne 0) {
    throw "Failed to create remote directories"
}

# Upload scripts directory via SCP
Write-Host "Uploading scripts directory..."
scp -r -i $sshKeyPath -o StrictHostKeyChecking=no "$localScriptPath/*" azureuser@${publicIp}:/home/azureuser/cleanroom-e2e/scripts/

if ($LASTEXITCODE -ne 0) {
    throw "Failed to upload scripts"
}

# Make all .ps1 files executable
Write-Host "Making PowerShell scripts executable..."
$chmodCmd = "find /home/azureuser/cleanroom-e2e/scripts -name '*.ps1' -exec chmod +x {} \;"
ssh -i $sshKeyPath -o StrictHostKeyChecking=no azureuser@$publicIp $chmodCmd

if ($LASTEXITCODE -ne 0) {
    throw "Failed to make scripts executable"
}

# Verify upload by comparing file counts
Write-Host "Verifying upload..."
$localFileCount = (Get-ChildItem -Path $localScriptPath -Recurse -File | Measure-Object).Count
$remoteFileCountCmd = "find /home/azureuser/cleanroom-e2e/scripts -type f | wc -l"
$remoteFileCount = ssh -i $sshKeyPath -o StrictHostKeyChecking=no azureuser@$publicIp $remoteFileCountCmd

if ($LASTEXITCODE -ne 0) {
    throw "Failed to get remote file count"
}

$remoteFileCount = [int]$remoteFileCount.Trim()

Write-Host "Local file count: $localFileCount"
Write-Host "Remote file count: $remoteFileCount"

if ($localFileCount -ne $remoteFileCount) {
    throw "File count mismatch! Local: $localFileCount, Remote: $remoteFileCount"
}

Write-Host "Scripts uploaded successfully to $vmName at $publicIp"
