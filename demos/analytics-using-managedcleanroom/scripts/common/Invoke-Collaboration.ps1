function Invoke-CleanroomAz {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string[]]$Arguments,
        [switch]$AllowNotFound
    )

    $PSNativeCommandUseErrorActionPreference = $false
    $output = & az @Arguments 2>&1
    $exitCode = $LASTEXITCODE
    $text = ($output | ForEach-Object { "$_" }) -join "`n"
    if ($exitCode -ne 0) {
        # az rest wraps its JSON error in "Not Found(...)", unlike other az commands.
        $notFoundPattern = '\((ResourceNotFound|ResourceGroupNotFound)\)|' +
            'Not Found\(\s*\{\s*"error"\s*:\s*\{\s*"code"\s*:\s*' +
            '"(ResourceNotFound|ResourceGroupNotFound)"'
        if ($AllowNotFound -and $text -match $notFoundPattern) {
            return $null
        }
        throw "Azure CLI failed (exit $exitCode): $text"
    }
    return $text
}

function Get-CollaborationContext {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$ResourceGroup,
        [Parameter(Mandatory)][string]$CollaborationName,
        [string]$Subscription,
        [string]$ApiVersion = "2026-09-30-preview",
        [switch]$DryRun
    )

    if (-not $Subscription) {
        $account = Invoke-CleanroomAz -Arguments @("account", "show", "-o", "json") |
            ConvertFrom-Json
        $Subscription = $account.id
    }
    if ([string]::IsNullOrWhiteSpace($Subscription)) {
        throw "No Azure subscription selected. Run az login or pass -Subscription."
    }
    $armEndpoint = "https://management.azure.com"
    $resourceId = "/subscriptions/$Subscription/resourceGroups/$ResourceGroup" +
        "/providers/Microsoft.CleanRoom/Collaborations/$CollaborationName"
    return [pscustomobject]@{
        ResourceGroup = $ResourceGroup
        CollaborationName = $CollaborationName
        Subscription = $Subscription
        ArmEndpoint = $armEndpoint
        ResourceUri = "$armEndpoint$resourceId"
        ApiVersion = $ApiVersion
        DryRun = [bool]$DryRun
    }
}

function ConvertTo-Collaborator {
    [CmdletBinding()]
    param([Parameter(Mandatory)][object]$Identity)

    $identifier = if ($Identity -is [string]) { $Identity } else { $Identity.userIdentifier }
    if ([string]::IsNullOrWhiteSpace($identifier)) {
        throw "Each collaborator requires a nonempty userIdentifier (email or app client ID)."
    }
    $result = @{ userIdentifier = $identifier }
    $appId = [guid]::Empty
    $isApp = [guid]::TryParse($identifier, [ref]$appId)
    if ($isApp -or $Identity.objectId -or $Identity.tenantId) {
        foreach ($field in @("objectId", "tenantId")) {
            $value = [guid]::Empty
            if (-not [guid]::TryParse([string]$Identity.$field, [ref]$value) -or
                $value -eq [guid]::Empty) {
                throw "SPN collaborator '$identifier' requires a valid $field."
            }
            $result[$field] = [string]$Identity.$field
        }
    }
    return $result
}

function Invoke-Collaboration {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][pscustomobject]$Context,
        [ValidateSet("GET", "POST", "DELETE")][string]$Method = "GET",
        [string]$Action,
        [object]$Body,
        [switch]$AllowNotFound
    )

    $url = $Context.ResourceUri
    if ($Action) { $url += "/$Action" }
    $url += "?api-version=$($Context.ApiVersion)"
    $bodyJson = if ($null -ne $Body) { $Body | ConvertTo-Json -Depth 20 -Compress }
    if ($Context.DryRun) {
        Write-Host "[DRY-RUN] $Method $url"
        if ($bodyJson) { Write-Host "[DRY-RUN] body: $bodyJson" }
        return
    }
    $arguments = @("rest", "--method", $Method, "--url", $url,
        "--resource", $Context.ArmEndpoint, "-o", "json")
    $bodyFile = $null
    try {
        if ($bodyJson) {
            $bodyFile = [IO.Path]::GetTempFileName()
            [IO.File]::WriteAllText($bodyFile, $bodyJson)
            $arguments += @("--headers", "Content-Type=application/json", "--body", "@$bodyFile")
        }
        $json = Invoke-CleanroomAz -Arguments $arguments -AllowNotFound:$AllowNotFound
        if (-not [string]::IsNullOrWhiteSpace($json)) { return $json | ConvertFrom-Json }
    }
    finally {
        if ($bodyFile) { Remove-Item -LiteralPath $bodyFile -Force }
    }
}

function Invoke-CollaborationOperation {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][pscustomobject]$Context,
        [Parameter(Mandatory)][string]$Action,
        [Parameter(Mandatory)][object]$Body,
        [ValidateRange(1, 1440)][int]$TimeoutMinutes = 60,
        [switch]$NoWait
    )

    if ($Context.DryRun) {
        Invoke-Collaboration -Context $Context -Method POST -Action $Action -Body $Body
        return
    }
    $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
    $timeoutMessage = "Timed out waiting for '$Action'. The Azure operation was not cancelled; " +
        "inspect its state before retrying."
    function Send-OperationRequest {
        param([string]$Uri, [string]$Method, [string]$Json)
        if ((Get-Date) -ge $deadline) { throw $timeoutMessage }
        $auth = Invoke-CleanroomAz -Arguments @("account", "get-access-token",
            "--subscription", $Context.Subscription,
            "--resource", "$($Context.ArmEndpoint)/", "-o", "json") | ConvertFrom-Json
        if (-not $auth.accessToken) { throw "Azure CLI returned no ARM access token." }
        $remaining = [Math]::Ceiling(($deadline - (Get-Date)).TotalSeconds)
        if ($remaining -le 0) { throw $timeoutMessage }
        $parameters = @{
            Uri = $Uri
            Method = $Method
            Headers = @{ Authorization = "Bearer $($auth.accessToken)" }
            TimeoutSec = [Math]::Min(120, $remaining)
            MaximumRedirection = 0
        }
        if ($Json) { $parameters.Body = $Json; $parameters.ContentType = "application/json" }
        $result = Invoke-WebRequest @parameters
        if ((Get-Date) -ge $deadline) { throw $timeoutMessage }
        return $result
    }

    $url = "$($Context.ResourceUri)/${Action}?api-version=$($Context.ApiVersion)"
    $response = Send-OperationRequest -Uri $url -Method POST `
        -Json ($Body | ConvertTo-Json -Depth 20 -Compress)
    if ($NoWait) { return }
    $trackingUrl = [string]@($response.Headers["Azure-AsyncOperation"])[0]
    if (-not $trackingUrl) {
        $trackingUrl = [string]@($response.Headers["Operation-Location"])[0]
    }
    $statusEndpoint = -not [string]::IsNullOrWhiteSpace($trackingUrl)
    if (-not $trackingUrl) { $trackingUrl = [string]@($response.Headers["Location"])[0] }
    if (-not $trackingUrl) {
        if ($response.StatusCode -eq 202) {
            throw "ARM accepted '$Action' without an operation tracking URL. Inspect state before retrying."
        }
        return
    }
    $trackingUri = $null
    $armHost = ([Uri]$Context.ArmEndpoint).Host
    if (-not [Uri]::TryCreate($trackingUrl, [UriKind]::Absolute, [ref]$trackingUri) -or
        $trackingUri.Scheme -ne "https" -or $trackingUri.UserInfo -or $trackingUri.Port -ne 443 -or
        ($trackingUri.Host -ne $armHost -and
            -not $trackingUri.Host.EndsWith(".management.azure.com", [StringComparison]::OrdinalIgnoreCase))) {
        throw "ARM returned an unexpected operation tracking URL."
    }
    do {
        $response = Send-OperationRequest -Uri $trackingUrl -Method GET
        if ($statusEndpoint) {
            $operation = $response.Content | ConvertFrom-Json
            if ([string]::IsNullOrWhiteSpace($operation.status)) {
                throw "ARM operation status response has no status."
            }
            if ($operation.status -eq "Succeeded") { return }
            if ($operation.status -in @("Failed", "Canceled", "Cancelled")) {
                $details = $operation.error | ConvertTo-Json -Depth 20 -Compress
                throw "ARM operation '$Action' ended in state '$($operation.status)': $details"
            }
            Write-Host "Waiting for $Action (operation status=$($operation.status))..."
        } elseif ($response.StatusCode -ne 202) {
            return
        }
        $remaining = ($deadline - (Get-Date)).TotalSeconds
        if ($remaining -le 0) { throw $timeoutMessage }
        $retryAfter = 30
        $headerDelay = 0
        if ([int]::TryParse([string]@($response.Headers["Retry-After"])[0], [ref]$headerDelay) -and
            $headerDelay -gt 0) { $retryAfter = $headerDelay }
        Start-Sleep -Seconds ([Math]::Min($retryAfter, $remaining))
    } while ($true)
}

function Wait-Collaboration {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][pscustomobject]$Context,
        [Parameter(Mandatory)][scriptblock]$Condition,
        [Parameter(Mandatory)][string]$Description,
        [ValidateRange(1, 1440)][int]$TimeoutMinutes = 60,
        [ValidateRange(1, 3600)][int]$IntervalSeconds = 30,
        [switch]$AllowNotFound,
        [datetime]$Deadline,
        [string]$DeploymentName
    )

    if (-not $PSBoundParameters.ContainsKey("Deadline")) {
        $Deadline = (Get-Date).AddMinutes($TimeoutMinutes)
    }
    $timeoutMessage = "Timed out after $TimeoutMinutes minutes waiting for $Description. " +
        "The Azure operation was not cancelled; inspect its state before retrying."
    do {
        if ((Get-Date) -ge $Deadline) { throw $timeoutMessage }
        $deploymentReady = $true
        if ($DeploymentName) {
            $deployment = Invoke-CleanroomAz -Arguments @("deployment", "group", "show",
                "--subscription", $Context.Subscription, "--resource-group", $Context.ResourceGroup,
                "--name", $DeploymentName, "-o", "json") | ConvertFrom-Json
            $deploymentState = $deployment.properties.provisioningState
            if ([string]::IsNullOrWhiteSpace($deploymentState)) {
                throw "ARM deployment '$DeploymentName' returned no provisioning state."
            }
            if ($deploymentState -in @("Failed", "Canceled", "Cancelled")) {
                $details = $deployment.properties.error | ConvertTo-Json -Depth 20 -Compress
                throw "ARM deployment '$DeploymentName' ended in state '${deploymentState}': $details"
            }
            $deploymentReady = $deploymentState -eq "Succeeded"
        }
        $collaboration = Invoke-Collaboration -Context $Context -AllowNotFound:$AllowNotFound
        if ($collaboration.properties.provisioningState -in @("Failed", "Canceled", "Cancelled")) {
            $details = $collaboration.properties | ConvertTo-Json -Depth 10 -Compress
            throw "Failed waiting for ${Description}: $details"
        }
        if ((Get-Date) -ge $Deadline) { throw $timeoutMessage }
        if ($deploymentReady -and (& $Condition $collaboration)) { return $collaboration }
        Write-Host "Waiting for $Description (state=$($collaboration.properties.provisioningState))..."
        if ($collaboration.properties.health.healthIssues) {
            $collaboration.properties.health.healthIssues | ConvertTo-Json -Depth 10 | Out-Host
        }
        $remainingSeconds = [Math]::Max(0, ($Deadline - (Get-Date)).TotalSeconds)
        Start-Sleep -Seconds ([Math]::Min($IntervalSeconds, $remainingSeconds))
    } while ($true)
}
