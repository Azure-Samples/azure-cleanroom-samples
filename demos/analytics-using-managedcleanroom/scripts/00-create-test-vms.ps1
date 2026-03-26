param(
    [string]$location = "westus",
    [switch]$skipVmCreation,
    [string]$vmSize = "Standard_D2s_v3"
)

$ErrorActionPreference = "Stop"

$resourceGroup = "cleanroom-e2e-vms-rg"
$northwindVmName = "cleanroom-e2e-northwind-vm"
$woodgroveVmName = "cleanroom-e2e-woodgrove-vm"
$sshKeyDir = "$PSScriptRoot/generated/vms"
$sshKeyPath = "$sshKeyDir/cleanroom-e2e-ssh-key"
$metadataDir = "$PSScriptRoot/generated/vms"

# Create directories if they don't exist
Write-Host "Creating directories..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $sshKeyDir | Out-Null
New-Item -ItemType Directory -Force -Path $metadataDir | Out-Null

# Get current public IP
Write-Host "Getting current public IP..." -ForegroundColor Cyan
$currentPublicIp = (curl -s https://api.ipify.org)
Write-Host "Current public IP: $currentPublicIp" -ForegroundColor Green

if (-not $skipVmCreation) {
    # Create resource group
    Write-Host "Creating resource group: $resourceGroup in $location..." -ForegroundColor Cyan
    az group create --name $resourceGroup --location $location --output none
    Write-Host "Resource group created successfully" -ForegroundColor Green

    # Generate SSH key pair if it doesn't exist
    if (-not (Test-Path $sshKeyPath)) {
        Write-Host "Generating SSH key pair..." -ForegroundColor Cyan
        ssh-keygen -t rsa -b 4096 -f $sshKeyPath -N '""' -C "cleanroom-e2e-test"
        Write-Host "SSH key pair generated successfully" -ForegroundColor Green
    } else {
        Write-Host "SSH key pair already exists, skipping generation" -ForegroundColor Yellow
    }

    # Create cloud-init configuration
    $cloudInitPath = "$sshKeyDir/cloud-init.yaml"
    $cloudInitContent = @"
#cloud-config
package_update: true
package_upgrade: true
packages:
  - jq
runcmd:
  - wget -q https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb
  - dpkg -i packages-microsoft-prod.deb
  - apt-get update
  - apt-get install -y powershell
  - curl -sL https://aka.ms/InstallAzureCLIDeb | bash
"@
    Write-Host "Creating cloud-init configuration..." -ForegroundColor Cyan
    Set-Content -Path $cloudInitPath -Value $cloudInitContent
    Write-Host "Cloud-init configuration created" -ForegroundColor Green

    # Create NSG
    $nsgName = "cleanroom-e2e-nsg"
    Write-Host "Creating Network Security Group: $nsgName..." -ForegroundColor Cyan
    az network nsg create `
        --resource-group $resourceGroup `
        --name $nsgName `
        --location $location `
        --output none
    Write-Host "NSG created successfully" -ForegroundColor Green

    # Add SSH rule to NSG
    Write-Host "Adding SSH rule to NSG (allowing access from $currentPublicIp)..." -ForegroundColor Cyan
    az network nsg rule create `
        --resource-group $resourceGroup `
        --nsg-name $nsgName `
        --name AllowSSH `
        --priority 1000 `
        --source-address-prefixes "$currentPublicIp/32" `
        --source-port-ranges '*' `
        --destination-address-prefixes '*' `
        --destination-port-ranges 22 `
        --access Allow `
        --protocol Tcp `
        --output none
    Write-Host "SSH rule added successfully" -ForegroundColor Green

    # Create Northwind VM
    Write-Host "Creating VM: $northwindVmName..." -ForegroundColor Cyan
    az vm create `
        --resource-group $resourceGroup `
        --name $northwindVmName `
        --location $location `
        --image "Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest" `
        --size $vmSize `
        --admin-username azureuser `
        --ssh-key-values "$sshKeyPath.pub" `
        --assign-identity `
        --nsg $nsgName `
        --public-ip-sku Standard `
        --custom-data $cloudInitPath `
        --output none
    Write-Host "VM $northwindVmName created successfully" -ForegroundColor Green

    # Create Woodgrove VM
    Write-Host "Creating VM: $woodgroveVmName..." -ForegroundColor Cyan
    az vm create `
        --resource-group $resourceGroup `
        --name $woodgroveVmName `
        --location $location `
        --image "Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest" `
        --size $vmSize `
        --admin-username azureuser `
        --ssh-key-values "$sshKeyPath.pub" `
        --assign-identity `
        --nsg $nsgName `
        --public-ip-sku Standard `
        --custom-data $cloudInitPath `
        --output none
    Write-Host "VM $woodgroveVmName created successfully" -ForegroundColor Green
}

# Get VM metadata
Write-Host "Retrieving VM metadata..." -ForegroundColor Cyan

# Get Northwind VM details
$northwindVmJson = az vm show --resource-group $resourceGroup --name $northwindVmName --show-details --output json | ConvertFrom-Json
$northwindPublicIp = $northwindVmJson.publicIps
$northwindIdentity = $northwindVmJson.identity

$northwindMetadata = @{
    name = $northwindVmName
    resourceGroup = $resourceGroup
    location = $northwindVmJson.location
    publicIp = $northwindPublicIp
    identity = @{
        principalId = $northwindIdentity.principalId
        tenantId = $northwindIdentity.tenantId
    }
}

# Get Woodgrove VM details
$woodgroveVmJson = az vm show --resource-group $resourceGroup --name $woodgroveVmName --show-details --output json | ConvertFrom-Json
$woodgrovePublicIp = $woodgroveVmJson.publicIps
$woodgroveIdentity = $woodgroveVmJson.identity

$woodgroveMetadata = @{
    name = $woodgroveVmName
    resourceGroup = $resourceGroup
    location = $woodgroveVmJson.location
    publicIp = $woodgrovePublicIp
    identity = @{
        principalId = $woodgroveIdentity.principalId
        tenantId = $woodgroveIdentity.tenantId
    }
}

Write-Host "VM metadata retrieved successfully" -ForegroundColor Green

# Verify VMs are running and SSH is accessible
Write-Host "Verifying VMs are running..." -ForegroundColor Cyan
$northwindVmState = az vm get-instance-view --resource-group $resourceGroup --name $northwindVmName --query "instanceView.statuses[?starts_with(code, 'PowerState/')].displayStatus" --output tsv
$woodgroveVmState = az vm get-instance-view --resource-group $resourceGroup --name $woodgroveVmName --query "instanceView.statuses[?starts_with(code, 'PowerState/')].displayStatus" --output tsv

if ($northwindVmState -ne "VM running") {
    Write-Host "Warning: $northwindVmName is not running (state: $northwindVmState)" -ForegroundColor Yellow
} else {
    Write-Host "$northwindVmName is running" -ForegroundColor Green
}

if ($woodgroveVmState -ne "VM running") {
    Write-Host "Warning: $woodgroveVmName is not running (state: $woodgroveVmState)" -ForegroundColor Yellow
} else {
    Write-Host "$woodgroveVmName is running" -ForegroundColor Green
}

# Test SSH connectivity
Write-Host "Testing SSH connectivity..." -ForegroundColor Cyan
$sshTestNorthwind = ssh -i $sshKeyPath -o StrictHostKeyChecking=no -o ConnectTimeout=10 azureuser@$northwindPublicIp "echo 'SSH test successful'" 2>$null
$sshTestWoodgrove = ssh -i $sshKeyPath -o StrictHostKeyChecking=no -o ConnectTimeout=10 azureuser@$woodgrovePublicIp "echo 'SSH test successful'" 2>$null

if ($sshTestNorthwind -eq "SSH test successful") {
    Write-Host "SSH to $northwindVmName is accessible" -ForegroundColor Green
} else {
    Write-Host "Warning: SSH to $northwindVmName may not be accessible yet (cloud-init may still be running)" -ForegroundColor Yellow
}

if ($sshTestWoodgrove -eq "SSH test successful") {
    Write-Host "SSH to $woodgroveVmName is accessible" -ForegroundColor Green
} else {
    Write-Host "Warning: SSH to $woodgroveVmName may not be accessible yet (cloud-init may still be running)" -ForegroundColor Yellow
}

# Save metadata to JSON files
Write-Host "Saving VM metadata to JSON files..." -ForegroundColor Cyan
$northwindMetadata | ConvertTo-Json -Depth 10 | Set-Content -Path "$metadataDir/northwind-vm.json"
$woodgroveMetadata | ConvertTo-Json -Depth 10 | Set-Content -Path "$metadataDir/woodgrove-vm.json"
Write-Host "Metadata saved successfully" -ForegroundColor Green

# Output summary
Write-Host ""
Write-Host "===== VM Creation Summary =====" -ForegroundColor Cyan
Write-Host "Resource Group: $resourceGroup" -ForegroundColor White
Write-Host "Location: $location" -ForegroundColor White
Write-Host ""
Write-Host "Northwind VM:" -ForegroundColor Yellow
Write-Host "  Name: $northwindVmName" -ForegroundColor White
Write-Host "  Public IP: $northwindPublicIp" -ForegroundColor White
Write-Host "  Principal ID: $($northwindIdentity.principalId)" -ForegroundColor White
Write-Host "  SSH Command: ssh -i $sshKeyPath azureuser@$northwindPublicIp" -ForegroundColor White
Write-Host ""
Write-Host "Woodgrove VM:" -ForegroundColor Yellow
Write-Host "  Name: $woodgroveVmName" -ForegroundColor White
Write-Host "  Public IP: $woodgrovePublicIp" -ForegroundColor White
Write-Host "  Principal ID: $($woodgroveIdentity.principalId)" -ForegroundColor White
Write-Host "  SSH Command: ssh -i $sshKeyPath azureuser@$woodgrovePublicIp" -ForegroundColor White
Write-Host ""
Write-Host "Metadata files:" -ForegroundColor Yellow
Write-Host "  $metadataDir/northwind-vm.json" -ForegroundColor White
Write-Host "  $metadataDir/woodgrove-vm.json" -ForegroundColor White
Write-Host "================================" -ForegroundColor Cyan
