param(
    [Parameter(Mandatory = $true)]
    [string]$contractId,

    [Parameter(Mandatory = $true)]
    [ValidateSet("cleanroomhello-job", "cleanroomhello-api", "analytics", "inference")]
    [string]$demo,

    [string]$resourceGroup = "$env:RESOURCE_GROUP",

    [string]$cleanRoomName = "cleanroom-$contractId",
    
    [string]$samplesRoot = "/home/samples",
    [string]$publicDir = "$samplesRoot/demo-resources/public",
    [string]$cleanroomEndpoint = (Get-Content "$publicDir/$cleanRoomName.endpoint"),

    [string]$application = "demoapp-$demo",
    [switch]$job,
    [switch]$skipStart
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../common/common.psm1

Test-AzureAccessToken

function Get-TimeStamp {
    return "[{0:MM/dd/yy} {0:HH:mm:ss}]" -f (Get-Date)
}

if ($cleanroomEndpoint -eq '')
{
    Write-Log Warning `
        "No endpoint details available for cleanroom '$cleanRoomName' at" `
        "'$publicDir/$cleanRoomName.endpoint'."
    return
}

# wait for code-launcher endpoint to be up.
$timeout = New-TimeSpan -Minutes 30
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
while ((curl -o /dev/null -w "%{http_code}" -s -k https://${cleanroomEndpoint}:8200/gov/doesnotexist/status) -ne "404") {
    Write-Host "Waiting for code-launcher endpoint to be up at https://${cleanroomEndpoint}:8200"
    Start-Sleep -Seconds 3
    if ($stopwatch.elapsed -gt $timeout) {
        throw "Hit timeout waiting for code-launcher endpoint to be up."
    }
}

if (!$skipStart)
{
    Write-Log OperationStarted `
        "$(Get-TimeStamp) Starting clean room application '$application' in $cleanRoomName'..."
    
    curl -X POST -s -k https://${cleanroomEndpoint}:8200/gov/$application/start
    Write-Log OperationCompleted `
        "$(Get-TimeStamp) Clean room application '$application' in '$cleanRoomName' started."
}

Write-Log Verbose `
    "$(Get-TimeStamp) Waiting for clean room '$cleanRoomName' ('$resourceGroup')..."

do {
    $applicationStatus = curl -s -k https://${cleanroomEndpoint}:8200/gov/$application/status
    Write-Log Verbose "Got application status: $applicationStatus"

    if ($applicationStatus -ne "" -and $null -ne $applicationStatus) {
        $status = $applicationStatus | ConvertFrom-Json
        if ($status.status -contains "exited") {
            Write-Log Information "$(Get-TimeStamp) Application is terminated. Checking exit code."
            if ($status.exit_code -ne 0) {
                Write-Log Critical "$(Get-TimeStamp) Application exited with non-zero exit code $($status.exit_code)"
                exit $status.exit_code
            }

            Write-Log OperationCompleted "$(Get-TimeStamp) Application exited successfully."
            exit 0
        }
        # Container status is 'started' when the container is present but is not running. This is an unexpected state.
        elseif ($status.status -contains "started") {
            Write-Log Error "$(Get-TimeStamp) Application is in started state. Only expected states are 'running' and 'exited'"
            exit 1
        }
        elseif ($status.status -contains "running") {
            Write-Log OperationCompleted "$(Get-TimeStamp) Application is running."
            if ($false -eq $job)
            {
                exit 0
            }
        }
        else {
            Write-Log Warning "$(Get-TimeStamp) Application is in unknown state:"
            Write-Log Warning $applicationStatus
            throw "Application is in unknown state."
        }
    }
    else {
        throw "Application container status not found. This is unexpected."
    }

    Write-Log Information "Waiting for 30 seconds before checking status again..."
    Start-Sleep -Seconds 30
} while ($true)
