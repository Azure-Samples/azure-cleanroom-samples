param(
    [string]$demo = "$env:DEMO",

    [string]$persona = "$env:PERSONA",

    [string]$samplesRoot = "/home/samples",
    [string]$publicDir = "$samplesRoot/demo-resources/public",
    [string]$privateDir = "$samplesRoot/demo-resources/private",
    [string]$cgsClient = "azure-cleanroom-samples-governance-client-$persona",
    [Nullable[DateTimeOffset]]$startDate = $null,
    [Nullable[DateTimeOffset]]$endDate = $null
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../common/common.psm1

function Get-TimeStamp {
    return "[{0:MM/dd/yy} {0:HH:mm:ss}]" -f (Get-Date)
}


function Invoke-SqlJobAndWait {
    param (
        [string]$queryDocumentId,
        [string]$collaborationContext,
        [string]$analyticsEndpoint,
        [Nullable[DateTimeOffset]]$startDate,
        [Nullable[DateTimeOffset]]$endDate
    )

    Write-Host "Setting collaboration context to '$collaborationContext'"
    az cleanroom collaboration context set --collaboration-name $collaborationContext

    $token = (az cleanroom governance client get-access-token --query accessToken -o tsv --name $collaborationContext)
    $script:submissionJson = $null
    & {
        # Disable $PSNativeCommandUseErrorActionPreference for this scriptblock
        $PSNativeCommandUseErrorActionPreference = $false

        # Additional Local-Authorization header support is added in agent as kubectl proxy command drops Authorization header.
        $runId = (New-Guid).ToString().Substring(0, 8)
        $body = @{ runId = $runId }
        if ($startDate) { $body.startDate = $startDate }
        if ($endDate) { $body.endDate = $endDate }

        $script:submissionJson = curl -k -s --fail-with-body -X POST "${analyticsEndpoint}/queries/$queryDocumentId/run" `
            -H "content-type: application/json" `
            -H "Local-Authorization: Bearer $token" `
            -d ($body | ConvertTo-Json -Compress)

        if ($LASTEXITCODE -ne 0) {
            Write-Output $script:submissionJson | jq
            throw "/queries/$queryDocumentId/run failed. Check the output above for details."
        }
    }

    $submissionResult = $script:submissionJson | ConvertFrom-Json
    $jobId = $submissionResult.id
    Write-Output "Job submitted with ID: $jobId"

    $applicationTimeout = New-TimeSpan -Minutes 30
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    Write-Output "Waiting for job execution to complete..."
    $jobStatus = $null
    do {
        Write-Host "$(Get-TimeStamp) Checking status of job: $jobId"

        $token = (az cleanroom governance client get-access-token --query accessToken -o tsv --name $cgsClient)
        $script:jobStatusResponse = ""
        & {
            # Disable $PSNativeCommandUseErrorActionPreference for this scriptblock
            $PSNativeCommandUseErrorActionPreference = $false
            $script:jobStatusResponse = $(curl -k -s --fail-with-body -X GET "${analyticsEndpoint}/status/$jobId" `
                    -H "Local-Authorization: Bearer $token")
            if ($LASTEXITCODE -ne 0) {
                $script:jobStatusResponse | jq
                throw "/status/$jobId failed. Check the output above for details."
            }
        }

        $script:jobStatusResponse | jq
        $jobStatus = $script:jobStatusResponse | ConvertFrom-Json

        if ($jobStatus.status.applicationState.state -eq "COMPLETED") {
            Write-Host -ForegroundColor Green "$(Get-TimeStamp) Application has completed execution."
            break
        }
        
        if ($jobStatus.status.applicationState.state -eq "FAILED") {
            Write-Host -ForegroundColor Red "$(Get-TimeStamp) Application has failed execution."
            throw "Application has failed execution."
        }

        Write-Host "Application $jobId state is: $($jobStatus.status.applicationState.state)"
    
        if ($stopwatch.elapsed -gt $applicationTimeout) {
            throw "Hit timeout waiting for application $jobId to complete execution."
        }
        
        Write-Host "Waiting for 15 seconds before checking status again..."
        Start-Sleep -Seconds 15
    } while ($true)
}

$queryDocumentId = Get-Content $publicDir/analytics.query-id
$contractId = Get-Content $publicDir/analytics.contract-id

$kubeConfig = "$publicDir/k8s-credentials.yaml"
if (-not (Test-Path -Path $kubeConfig)) {
    throw "$kubeConfig was not found."
}
Get-Job -Command "*kubectl proxy --port 8181*" | Stop-Job
Get-Job -Command "*kubectl proxy --port 8181*" | Remove-Job
kubectl proxy --port 8181 --kubeconfig $kubeConfig &

$deploymentInformation = (az cleanroom governance deployment information show `
        --contract-id $contractId `
        --governance-client $cgsClient | ConvertFrom-Json)

Write-Output "Submitting SQL job to analytics endpoint: $($deploymentInformation.data.url)"
$analyticsEndpoint = $deploymentInformation.data.url

$timeout = New-TimeSpan -Minutes 1
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
& {
    # Disable $PSNativeCommandUseErrorActionPreference for this scriptblock
    $PSNativeCommandUseErrorActionPreference = $false
    while ((curl -o /dev/null -w "%{http_code}" -k -s ${analyticsEndpoint}/ready) -ne "200") {
        Write-Output "Waiting for analytics endpoint to be ready at ${analyticsEndpoint}/ready"
        Start-Sleep -Seconds 3
        if ($stopwatch.elapsed -gt $timeout) {
            # Re-run the command once to log its output.
            curl -k -s ${analyticsEndpoint}/ready
            throw "Hit timeout waiting for analytics endpoint to be ready."
        }
    }
}

if ((-not $startDate -and $endDate) -or (-not $endDate -and $startDate)) {
    throw "Both startDate and endDate should be specified together."
}

Write-Output "Executing query '$queryDocumentId' as '$persona'..."
Invoke-SqlJobAndWait `
    -queryDocumentId $queryDocumentId `
    -collaborationContext $cgsClient `
    -analyticsEndpoint $analyticsEndpoint `
    -startDate $startDate `
    -endDate $endDate
