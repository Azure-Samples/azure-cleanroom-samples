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

function Wait-Collaboration {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][pscustomobject]$Context,
        [Parameter(Mandatory)][scriptblock]$Condition,
        [Parameter(Mandatory)][string]$Description,
        [ValidateRange(1, 1440)][int]$TimeoutMinutes = 45,
        [ValidateRange(1, 3600)][int]$IntervalSeconds = 30,
        [switch]$AllowNotFound
    )

    $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
    do {
        $collaboration = Invoke-Collaboration -Context $Context -AllowNotFound:$AllowNotFound
        if ($collaboration.properties.provisioningState -in @("Failed", "Canceled", "Cancelled")) {
            $details = $collaboration.properties | ConvertTo-Json -Depth 10 -Compress
            throw "Failed waiting for ${Description}: $details"
        }
        if (& $Condition $collaboration) { return $collaboration }
        if ((Get-Date) -ge $deadline) {
            throw "Timed out after $TimeoutMinutes minutes waiting for $Description. Check resource state before retrying."
        }
        Write-Host "Waiting for $Description (state=$($collaboration.properties.provisioningState))..."
        if ($collaboration.properties.health.healthIssues) {
            $collaboration.properties.health.healthIssues | ConvertTo-Json -Depth 10 | Out-Host
        }
        Start-Sleep -Seconds $IntervalSeconds
    } while ($true)
}
