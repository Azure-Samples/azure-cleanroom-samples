#!/usr/bin/env pwsh
# Monitor VM run-command operations in real-time

param(
    [string]$vmName = "cleanroom-e2e-woodgrove-vm",
    [string]$resourceGroup = "cleanroom-e2e-vms-rg",
    [int]$intervalSeconds = 15
)

Write-Host "=== Monitoring VM Run-Command Operations ===" -ForegroundColor Cyan
Write-Host "VM: $vmName" -ForegroundColor Yellow
Write-Host "Resource Group: $resourceGroup" -ForegroundColor Yellow
Write-Host "Refresh Interval: ${intervalSeconds}s" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop monitoring" -ForegroundColor Gray
Write-Host ""

$lastStatus = ""

while ($true) {
    $timestamp = Get-Date -Format "HH:mm:ss"
    
    # Get recent run-command operations
    $operations = az monitor activity-log list `
        --resource-group $resourceGroup `
        --start-time (Get-Date).AddMinutes(-30).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ") `
        --query "[?contains(operationName.value, 'runCommand') && contains(resourceId, '$vmName')].{Time:eventTimestamp, Status:status.localizedValue, SubStatus:subStatus.localizedValue}" `
        --output json 2>$null | ConvertFrom-Json
    
    if ($operations) {
        $latest = $operations | Sort-Object -Property Time -Descending | Select-Object -First 1
        $status = "$($latest.Status) - $($latest.SubStatus)"
        $statusTime = [DateTime]::Parse($latest.Time).ToLocalTime().ToString("HH:mm:ss")
        
        if ($status -ne $lastStatus) {
            Write-Host "[$timestamp] Status changed: $status (since $statusTime)" -ForegroundColor Green
            $lastStatus = $status
        } else {
            Write-Host "[$timestamp] Current status: $status (since $statusTime)" -ForegroundColor DarkGray
        }
        
        # Show summary of recent operations
        $recentOps = $operations | Select-Object -First 5
        if ($recentOps.Count -gt 1) {
            Write-Host "  Recent operations:" -ForegroundColor DarkGray
            foreach ($op in $recentOps) {
                $opTime = [DateTime]::Parse($op.Time).ToLocalTime().ToString("HH:mm:ss")
                Write-Host "    $opTime - $($op.Status)" -ForegroundColor DarkGray
            }
        }
    } else {
        Write-Host "[$timestamp] No recent run-command operations found" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Start-Sleep -Seconds $intervalSeconds
}
