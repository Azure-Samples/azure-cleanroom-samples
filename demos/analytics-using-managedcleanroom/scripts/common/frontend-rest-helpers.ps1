<#
.SYNOPSIS
    Common frontend REST API helper functions.

.DESCRIPTION
    Provides helper functions for making authenticated REST calls to the managed
    cleanroom frontend service. Replaces `az managedcleanroom frontend` CLI calls
    which are broken on Python 3.13 ('tuple' object has no attribute 'token').

    All functions accept a $FrontendContext hashtable containing:
      - baseUrl:    Frontend base URL (no trailing /collaborations)
      - apiVersion: API version query parameter (default: 2026-03-01-preview)

    Auth tokens are obtained via `az account get-access-token`.

    TLS Handling:
      - If the /report endpoint is working, call Get-FrontendCertificate first
        to get the CCR CA cert, then use it for validation.
      - If /report is broken (current state), use -SkipCertificateCheck as a
        temporary workaround.

.NOTES
    REST API endpoint mapping (verified from CLI source _operations.py):
    
    frontend analytics dataset show      -> GET  /collaborations/{id}/analytics/datasets/{docId}
    frontend analytics dataset publish   -> POST /collaborations/{id}/analytics/datasets/{docId}/publish
    frontend consent set                 -> PUT  /collaborations/{id}/consent/{docId}
    frontend analytics query show        -> GET  /collaborations/{id}/analytics/queries/{docId}
    frontend analytics query publish     -> POST /collaborations/{id}/analytics/queries/{docId}/publish
    frontend analytics query vote accept -> POST /collaborations/{id}/analytics/queries/{docId}/vote
    frontend analytics query run         -> POST /collaborations/{id}/analytics/queries/{docId}/run
    frontend analytics query runresult   -> GET  /collaborations/{id}/analytics/runs/{jobId}
    frontend analytics query runhistory  -> GET  /collaborations/{id}/analytics/queries/{docId}/runs
    frontend analytics auditevent list   -> GET  /collaborations/{id}/analytics/auditevents
    frontend oidc issuerinfo show        -> GET  /collaborations/{id}/oidc/issuerInfo
    frontend oidc keys show              -> GET  /collaborations/{id}/oidc/keys
    frontend oidc setIssuerUrl           -> POST /collaborations/{id}/oidc/setIssuerUrl
#>

# -- Context initialization -------------------------------------------------------

function New-FrontendContext {
    <#
    .SYNOPSIS
        Creates a frontend context for REST calls.
    .PARAMETER frontendEndpoint
        Frontend URL. May include /collaborations suffix (will be stripped).
    .PARAMETER apiVersion
        API version query parameter.
    #>
    param(
        [Parameter(Mandatory)]
        [string]$frontendEndpoint,
        [string]$apiVersion = "2026-03-01-preview"
    )

    $baseUrl = $frontendEndpoint.TrimEnd('/')
    if ($baseUrl.EndsWith('/collaborations')) {
        $baseUrl = $baseUrl.Substring(0, $baseUrl.Length - '/collaborations'.Length)
    }

    return @{
        baseUrl    = $baseUrl
        apiVersion = $apiVersion
    }
}

# -- Token management -------------------------------------------------------------

function Get-FrontendToken {
    <#
    .SYNOPSIS
        Gets a bearer token for frontend API calls.
    .DESCRIPTION
        Priority:
        1. $env:CLEANROOM_FRONTEND_TOKEN — pre-set token (e.g., MSAL IdToken)
        2. /tmp/msal-idtoken.txt — cached MSAL device-code IdToken
        3. az account get-access-token — ARM token fallback

        Use MSAL device-code flow to generate the IdToken:
          $token = Get-MsalToken -ClientId "8a3849c1-81c5-4d62-b83e-3bb2bb11251a" -TenantId "common" -Scopes "User.Read" -DeviceCode
          $token.IdToken | Out-File -FilePath "/tmp/msal-idtoken.txt" -NoNewline
    #>
    # Priority 1: Environment variable
    if ($env:CLEANROOM_FRONTEND_TOKEN) {
        return $env:CLEANROOM_FRONTEND_TOKEN
    }

    # Priority 2: Cached MSAL IdToken file
    $msalTokenFile = "/tmp/msal-idtoken.txt"
    if (Test-Path $msalTokenFile) {
        $cachedToken = (Get-Content $msalTokenFile -Raw).Trim()
        if ($cachedToken) {
            return $cachedToken
        }
    }

    # Priority 3: ARM access token fallback
    $token = az account get-access-token --query accessToken -o tsv
    if ($LASTEXITCODE -ne 0 -or -not $token) {
        throw "No frontend token available. Run MSAL device-code flow first:`n  `$t = Get-MsalToken -ClientId '8a3849c1-81c5-4d62-b83e-3bb2bb11251a' -TenantId 'common' -Scopes 'User.Read' -DeviceCode`n  `$t.IdToken | Out-File /tmp/msal-idtoken.txt -NoNewline"
    }
    return $token
}

# -- Core REST caller --------------------------------------------------------------

function Invoke-FrontendRest {
    <#
    .SYNOPSIS
        Makes an authenticated REST call to the frontend service.
    .PARAMETER Context
        Frontend context from New-FrontendContext.
    .PARAMETER Method
        HTTP method (GET, POST, PUT, DELETE).
    .PARAMETER Path
        Relative path after /collaborations (e.g., "{collabId}/analytics/datasets/{docId}").
    .PARAMETER Body
        Request body (will be converted to JSON if not already a string).
    .PARAMETER Description
        Human-readable description for logging.
    .OUTPUTS
        Response object (parsed from JSON).
    #>
    param(
        [Parameter(Mandatory)]
        [hashtable]$Context,

        [Parameter(Mandatory)]
        [string]$Method,

        [Parameter(Mandatory)]
        [string]$Path,

        [object]$Body,

        [string]$Description = ""
    )

    $token = Get-FrontendToken
    $url = "$($Context.baseUrl)/collaborations/$($Path.TrimStart('/'))"
    if ($url -notmatch '\?') {
        $url += "?api-version=$($Context.apiVersion)"
    } else {
        $url += "&api-version=$($Context.apiVersion)"
    }

    $headers = @{
        Authorization  = "Bearer $token"
        "Content-Type" = "application/json"
    }

    $logPrefix = if ($Description) { "$Description — " } else { "" }
    Write-Host "  ${logPrefix}$Method $url" -ForegroundColor DarkGray

    $params = @{
        Uri                  = $url
        Method               = $Method
        Headers              = $headers
        SkipCertificateCheck = $true   # Temporary: /report endpoint broken, can't get CA cert
        ErrorAction          = "Stop"
    }

    if ($Body) {
        if ($Body -is [string]) {
            $params.Body = $Body
        } else {
            $params.Body = $Body | ConvertTo-Json -Depth 20
        }
        $params.ContentType = "application/json"
    }

    try {
        $response = Invoke-RestMethod @params
        return $response
    }
    catch {
        $statusCode = $_.Exception.Response.StatusCode
        $errorBody = ""
        try {
            $stream = $_.Exception.Response.GetResponseStream()
            $reader = [System.IO.StreamReader]::new($stream)
            $errorBody = $reader.ReadToEnd()
        } catch {}
        
        Write-Host "  ERROR: $Method $url returned $statusCode" -ForegroundColor Red
        if ($errorBody) {
            Write-Host "  Response: $errorBody" -ForegroundColor Red
        }
        throw
    }
}

function Invoke-FrontendRestSafe {
    <#
    .SYNOPSIS
        Like Invoke-FrontendRest, but returns $null on failure instead of throwing.
    #>
    param(
        [Parameter(Mandatory)]
        [hashtable]$Context,
        [Parameter(Mandatory)]
        [string]$Method,
        [Parameter(Mandatory)]
        [string]$Path,
        [object]$Body,
        [string]$Description = ""
    )

    try {
        return Invoke-FrontendRest -Context $Context -Method $Method -Path $Path -Body $Body -Description $Description
    }
    catch {
        return $null
    }
}

# -- Dataset operations ------------------------------------------------------------

function Get-FrontendDataset {
    <#
    .SYNOPSIS
        Gets a dataset from the frontend.
        CLI: az managedcleanroom frontend analytics dataset show
        API: GET /collaborations/{id}/analytics/datasets/{docId}
    #>
    param(
        [Parameter(Mandatory)][hashtable]$Context,
        [Parameter(Mandatory)][string]$CollaborationId,
        [Parameter(Mandatory)][string]$DocumentId
    )
    Invoke-FrontendRestSafe -Context $Context -Method "GET" `
        -Path "$CollaborationId/analytics/datasets/$DocumentId" `
        -Description "Show dataset '$DocumentId'"
}

function Publish-FrontendDataset {
    <#
    .SYNOPSIS
        Publishes a dataset to the frontend.
        CLI: az managedcleanroom frontend analytics dataset publish
        API: POST /collaborations/{id}/analytics/datasets/{docId}/publish
    .PARAMETER Body
        Dataset specification body. Expected keys:
          name, datasetSchema, datasetAccessPolicy, store, identity
          (and optionally dek, kek for CPK encryption mode)
    #>
    param(
        [Parameter(Mandatory)][hashtable]$Context,
        [Parameter(Mandatory)][string]$CollaborationId,
        [Parameter(Mandatory)][string]$DocumentId,
        [Parameter(Mandatory)][object]$Body
    )
    Invoke-FrontendRest -Context $Context -Method "POST" `
        -Path "$CollaborationId/analytics/datasets/$DocumentId/publish" `
        -Body $Body `
        -Description "Publish dataset '$DocumentId'"
}

function Set-FrontendConsent {
    <#
    .SYNOPSIS
        Sets execution consent on a dataset or query.
        CLI: az managedcleanroom frontend consent set
        API: PUT /collaborations/{id}/consent/{docId}
    .PARAMETER Action
        Consent action (enable or disable).
    #>
    param(
        [Parameter(Mandatory)][hashtable]$Context,
        [Parameter(Mandatory)][string]$CollaborationId,
        [Parameter(Mandatory)][string]$DocumentId,
        [ValidateSet("enable", "disable")][string]$Action = "enable"
    )
    $body = @{ consentAction = $Action }
    Invoke-FrontendRest -Context $Context -Method "PUT" `
        -Path "$CollaborationId/consent/$DocumentId" `
        -Body $body `
        -Description "Set consent '$Action' on '$DocumentId'"
}

# -- Query operations --------------------------------------------------------------

function Get-FrontendQuery {
    <#
    .SYNOPSIS
        Gets a query from the frontend.
        CLI: az managedcleanroom frontend analytics query show
        API: GET /collaborations/{id}/analytics/queries/{docId}
    #>
    param(
        [Parameter(Mandatory)][hashtable]$Context,
        [Parameter(Mandatory)][string]$CollaborationId,
        [Parameter(Mandatory)][string]$DocumentId
    )
    Invoke-FrontendRestSafe -Context $Context -Method "GET" `
        -Path "$CollaborationId/analytics/queries/$DocumentId" `
        -Description "Show query '$DocumentId'"
}

function Publish-FrontendQuery {
    <#
    .SYNOPSIS
        Publishes a query to the frontend.
        CLI: az managedcleanroom frontend analytics query publish
        API: POST /collaborations/{id}/analytics/queries/{docId}/publish
    .PARAMETER Body
        Query specification body. Expected keys:
          inputDatasets (string: "ds1:view1,ds2:view2"),
          outputDataset (string: "ds:view"),
          queryData (array of {data, executionSequence, preConditions, postFilters})
    #>
    param(
        [Parameter(Mandatory)][hashtable]$Context,
        [Parameter(Mandatory)][string]$CollaborationId,
        [Parameter(Mandatory)][string]$DocumentId,
        [Parameter(Mandatory)][object]$Body
    )
    Invoke-FrontendRest -Context $Context -Method "POST" `
        -Path "$CollaborationId/analytics/queries/$DocumentId/publish" `
        -Body $Body `
        -Description "Publish query '$DocumentId'"
}

function Invoke-FrontendQueryVoteAccept {
    <#
    .SYNOPSIS
        Votes to accept a query.
        CLI: az managedcleanroom frontend analytics query vote accept
        API: POST /collaborations/{id}/analytics/queries/{docId}/vote
    #>
    param(
        [Parameter(Mandatory)][hashtable]$Context,
        [Parameter(Mandatory)][string]$CollaborationId,
        [Parameter(Mandatory)][string]$DocumentId
    )
    $body = @{ voteAction = "accept" }
    Invoke-FrontendRest -Context $Context -Method "POST" `
        -Path "$CollaborationId/analytics/queries/$DocumentId/vote" `
        -Body $body `
        -Description "Vote accept query '$DocumentId'"
}

function Invoke-FrontendQueryRun {
    <#
    .SYNOPSIS
        Runs an approved query.
        CLI: az managedcleanroom frontend analytics query run
        API: POST /collaborations/{id}/analytics/queries/{docId}/run
    .PARAMETER RunId
        Optional run ID. Auto-generated UUID if not provided.
    #>
    param(
        [Parameter(Mandatory)][hashtable]$Context,
        [Parameter(Mandatory)][string]$CollaborationId,
        [Parameter(Mandatory)][string]$DocumentId,
        [string]$RunId
    )
    $body = @{}
    if ($RunId) {
        $body.runId = $RunId
    } else {
        $body.runId = [guid]::NewGuid().ToString()
    }
    Invoke-FrontendRest -Context $Context -Method "POST" `
        -Path "$CollaborationId/analytics/queries/$DocumentId/run" `
        -Body $body `
        -Description "Run query '$DocumentId'"
}

function Get-FrontendQueryRunResult {
    <#
    .SYNOPSIS
        Gets a query run result by job ID.
        CLI: az managedcleanroom frontend analytics query runresult show
        API: GET /collaborations/{id}/analytics/runs/{jobId}
    #>
    param(
        [Parameter(Mandatory)][hashtable]$Context,
        [Parameter(Mandatory)][string]$CollaborationId,
        [Parameter(Mandatory)][string]$JobId
    )
    Invoke-FrontendRestSafe -Context $Context -Method "GET" `
        -Path "$CollaborationId/analytics/runs/$JobId" `
        -Description "Get run result for job '$JobId'"
}

function Get-FrontendQueryRunHistory {
    <#
    .SYNOPSIS
        Gets query run history.
        CLI: az managedcleanroom frontend analytics query runhistory list
        API: GET /collaborations/{id}/analytics/queries/{docId}/runs
    #>
    param(
        [Parameter(Mandatory)][hashtable]$Context,
        [Parameter(Mandatory)][string]$CollaborationId,
        [Parameter(Mandatory)][string]$DocumentId
    )
    Invoke-FrontendRest -Context $Context -Method "GET" `
        -Path "$CollaborationId/analytics/queries/$DocumentId/runs" `
        -Description "Get run history for query '$DocumentId'"
}

function Get-FrontendAuditEvents {
    <#
    .SYNOPSIS
        Gets audit events for a collaboration.
        CLI: az managedcleanroom frontend analytics auditevent list
        API: GET /collaborations/{id}/analytics/auditevents
    #>
    param(
        [Parameter(Mandatory)][hashtable]$Context,
        [Parameter(Mandatory)][string]$CollaborationId
    )
    Invoke-FrontendRest -Context $Context -Method "GET" `
        -Path "$CollaborationId/analytics/auditevents" `
        -Description "Get audit events"
}
