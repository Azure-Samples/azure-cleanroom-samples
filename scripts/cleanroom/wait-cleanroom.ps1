param(
    [Parameter(Mandatory = $true)]
    [string]$contractId,

    [string]$resourceGroup = "$env:RESOURCE_GROUP",

    [string]$cleanRoomName = "cleanroom-$contractId",

    [string]$samplesRoot = "/home/samples",
    [string]$publicDir = "$samplesRoot/demo-resources/public",
    [string]$cleanroomEndpoint = "$publicDir/$cleanRoomName.endpoint",

    [switch]$job
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../common/common.psm1

Test-AzureAccessToken

function Get-TimeStamp {
    return "[{0:MM/dd/yy} {0:HH:mm:ss}]" -f (Get-Date)
}

Write-Log Verbose `
    "$(Get-TimeStamp) Waiting for clean room '$cleanRoomName' ('$resourceGroup')..."

do {
    $cleanroom = az container show --name $cleanRoomName --resource-group $resourceGroup
    $cleanroomState = $cleanroom | jq -r ".instanceView.state"

    # If the cleanroom deployment failed, exit.
    if ($cleanroomState -eq "Failed") {
        Write-Log Critical `
            "$(Get-TimeStamp) Clean room '$cleanRoomName' has encountered an error."
        exit 1
    }
    elseif ($cleanroomState -eq "Running") {
        Write-Log Verbose `
            "$(Get-TimeStamp) Clean room '$cleanRoomName' is running..."

        #
        # Check if any containers have encountered an error.
        # Whe
        $errorContainers = ( `
            $cleanroom | `
            jq -r '[.containers | .[] | {name:.name, status:.instanceView.currentState.detailStatus} | select(.status == "Error")]' | `
            ConvertFrom-Json `
        )

        if (0 -ne $errorContainers.Count) {
            Write-Log Critical `
                "$(Get-TimeStamp) Clean room '$cleanRoomName' has encountered an error in one" `
                "or more containers:"
            foreach ($container in $errorContainers)
            {
                $azCmd = "az container logs --name $cleanRoomName --resource-group $resourceGroup --container-name $($container.name)"
                Write-Log Error `
                    "Container '$($container.name)' has encountered an error."
                Write-Log Information `
                    "$azCmd"
                Write-Log Verbose `
                    "Logs:"
                az container logs --name $cleanRoomName --resource-group $resourceGroup --container-name $container.name
            }

            exit 1
        }

        # Fetch code launcher sidecar status.
        $codeLauncherState = $cleanroom | jq '.containers | .[] | select(.name | contains("code-launcher")) | .instanceView.currentState' | ConvertFrom-Json
        if ($codeLauncherState.state -eq "Running") {
            Write-Log Verbose `
                "$(Get-TimeStamp) Clean room application is running..."
            if ($false -eq $job)
            {
                Write-Log OperationCompleted `
                    "$(Get-TimeStamp) Clean room application started successfully."

                $ccrIP =  $cleanroom | jq -r ".ipAddress.ip"
                $ccrIP | Out-File "$cleanroomEndpoint"
                Write-Log OperationCompleted `
                    "CCR endpoint details {IP: '$ccrIp'} written to '$cleanroomEndpoint'."
                exit 0
            }
        }
        elseif ($codeLauncherState.state -eq "Terminated") {
            Write-Log OperationStarted `
                "$(Get-TimeStamp) Clean room application has terminated. Checking exit code..."
            $exitCode = $codeLauncherState.exitCode
            if ($exitCode -ne 0) {
                Write-Log Critical `
                    "$(Get-TimeStamp) Clean room application exited with non-zero exit code '$exitCode'."
                exit $exitCode
            }
            else {
                Write-Log OperationCompleted `
                    "$(Get-TimeStamp) Clean room application exited successfully."

                $ccrIP =  $cleanroom | jq -r ".ipAddress.ip"
                $ccrIP | Out-File "$cleanroomEndpoint"
                Write-Log OperationCompleted `
                    "CCR endpoint details {IP: '$ccrIp'} written to '$cleanroomEndpoint'."
                exit 0
            }
        }
        else {
            Write-Log Information `
                "$(Get-TimeStamp) Clean room application is in state '$($codeLauncherState.state)'"
        }
    }
    else {
        Write-Log Information `
            "$(Get-TimeStamp) Clean room '$cleanRoomName' is in state '$cleanroomState'"
    }

    Write-Log Verbose `
        "$(Get-TimeStamp) Waiting for 20 seconds before checking status again..."
    Start-Sleep -Seconds 20
} while ($true)
