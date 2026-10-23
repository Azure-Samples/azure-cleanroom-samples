<#
.SYNOPSIS
    Shared helper for the managed clean-room *frontend* (dataplane) REST API.

.DESCRIPTION
    Steps 03-12 of README-API.md operate against the ACCR frontend service
    (NOT ARM). This helper centralizes the `Invoke-Frontend` wrapper and token
    resolution so every step script stays small and consistent.

    Dot-source it at the top of a step script:

        . "$PSScriptRoot/Invoke-Frontend.ps1"
        $fe = Get-FrontendContext -Persona woodgrove
        $collabs = (Invoke-Frontend -Context $fe -Path "" -Method GET).collaborations

    Token resolution order (first match wins):
      1. -TokenFile <path> (explicit override)
      2. $env:CLEANROOM_FRONTEND_TOKEN (SPN / CI)
      3. $env:MANAGEDCLEANROOM_ACCESS_TOKEN (CLI session)
      4. Temp file msal-idtoken-<persona>.txt (per README Step 1.5)

    Acquire the per-persona token first (README Step 1.5), e.g.:
        az account get-access-token --resource "https://management.azure.com/" `
            --query accessToken -o tsv |
            Out-File (Join-Path $env:TEMP "msal-idtoken-woodgrove.txt") -NoNewline
#>

function Get-FrontendContext {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Persona,

        [string]$Frontend,

        [string]$ApiVersion = "2026-03-01-preview",

        [string]$TokenFile,

        # When set, Invoke-Frontend prints the request it *would* send and
        # returns $null instead of calling the service. Lets you validate URL
        # and body construction with no allow-listed subscription / live infra.
        [switch]$DryRun
    )

    if (-not $Frontend) {
        $Frontend = "https://prod-nonattested.workload-frontendwestus.cleanroom.cloudapp.azure.net"
    }
    $endpoint = $null
    if (-not [Uri]::TryCreate($Frontend, [UriKind]::Absolute, [ref]$endpoint) -or
        $endpoint.Scheme -ne "https" -or $endpoint.UserInfo -or
        $endpoint.Query -or $endpoint.Fragment) {
        throw "Frontend must be an absolute HTTPS URL without credentials, query, or fragment."
    }

    # Resolve the bearer token (skipped in dry-run mode).
    $token = $null
    if ($DryRun) {
        $token = "<dry-run-token>"
    }
    elseif ($TokenFile) {
        if (-not (Test-Path -LiteralPath $TokenFile -PathType Leaf)) {
            throw "Frontend token file not found: $TokenFile"
        }
        $token = (Get-Content -LiteralPath $TokenFile -Raw).Trim()
    }
    elseif ($env:CLEANROOM_FRONTEND_TOKEN) {
        $token = $env:CLEANROOM_FRONTEND_TOKEN.Trim()
    }
    elseif ($env:MANAGEDCLEANROOM_ACCESS_TOKEN) {
        $token = $env:MANAGEDCLEANROOM_ACCESS_TOKEN.Trim()
    }
    else {
        if (-not $TokenFile) {
            $TokenFile = Join-Path ([System.IO.Path]::GetTempPath()) "msal-idtoken-$Persona.txt"
        }
        if (-not (Test-Path $TokenFile)) {
            throw "Frontend token not found. Set `$env:CLEANROOM_FRONTEND_TOKEN or create '$TokenFile' (see README Step 1.5)."
        }
        $token = (Get-Content $TokenFile -Raw).Trim()
    }
    if ([string]::IsNullOrWhiteSpace($token)) {
        throw "Frontend token is empty. Supply a valid -TokenFile or frontend token environment variable."
    }

    return [pscustomobject]@{
        Persona    = $Persona
        Frontend   = $Frontend.TrimEnd('/')
        ApiVersion = $ApiVersion
        Token      = $token
        DryRun     = [bool]$DryRun
    }
}

function Invoke-Frontend {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [pscustomobject]$Context,

        [string]$Method = "GET",

        [string]$Path,

        [object]$Body
    )

    $headers = @{
        Authorization  = "Bearer $($Context.Token)"
        "Content-Type" = "application/json"
    }

    $url = "$($Context.Frontend)/collaborations"
    if ($Path) { $url += $(if ($Path.StartsWith("?")) { $Path } else { "/$Path" }) }
    if ($url -notmatch '\?') { $url += "?api-version=$($Context.ApiVersion)" }
    else { $url += "&api-version=$($Context.ApiVersion)" }

    $bodyJson = $null
    if ($Body) {
        $bodyJson = if ($Body -is [string]) { $Body } else { $Body | ConvertTo-Json -Depth 20 }
    }

    if ($Context.DryRun) {
        Write-Host "[DRY-RUN] $Method $url" -ForegroundColor Yellow
        if ($bodyJson) { Write-Host "[DRY-RUN] body: $bodyJson" -ForegroundColor DarkYellow }
        return $null
    }

    $params = @{
        Uri                  = $url
        Method               = $Method
        Headers              = $headers
    }
    if ($bodyJson) {
        $params.Body = $bodyJson
        $params.ContentType = "application/json"
    }

    return Invoke-RestMethod @params
}

function Resolve-CollaborationId {
    <#
    .SYNOPSIS
        Resolves a UUID by ID, unique name, or the only visible collaboration.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][pscustomobject]$Context,
        [string]$CollaborationId,
        [string]$CollaborationName
    )

    if ($CollaborationId) { return $CollaborationId }

    if ($Context.DryRun) {
        return "<collaboration-id>"
    }

    $collabs = (Invoke-Frontend -Context $Context -Path "" -Method GET).collaborations
    if (-not $collabs) { throw "No collaborations visible to persona '$($Context.Persona)'." }

    if ($CollaborationName) {
        $match = @($collabs | Where-Object { $_.collaborationName -eq $CollaborationName })
        if (-not $match) { throw "No collaboration named '$CollaborationName' visible to '$($Context.Persona)'." }
        if ($match.Count -ne 1) {
            throw "Multiple collaborations named '$CollaborationName'. Pass -CollaborationId."
        }
        return $match[0].collaborationId
    }

    if (@($collabs).Count -gt 1) {
        throw "Multiple collaborations visible. Pass -CollaborationName or -CollaborationId."
    }
    return @($collabs)[0].collaborationId
}
