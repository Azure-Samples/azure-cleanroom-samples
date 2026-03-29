<#
.SYNOPSIS
    Unified frontend helper functions supporting both REST and CLI modes.

.DESCRIPTION
    Provides helper functions for interacting with the managed cleanroom frontend
    service. Supports two API modes:

    REST mode (default):
        Makes direct HTTP calls to the frontend REST API using Invoke-RestMethod.
        This is the original approach and works regardless of CLI extension state.

    CLI mode:
        Uses `az managedcleanroom frontend` CLI commands. Requires the
        managedcleanroom CLI extension (v1.0.0b5+) with bug fixes for:
        - Token wrapping (AccessToken namedtuple normalization)
        - SSL cert verification (AZURE_CLI_DISABLE_CONNECTION_VERIFICATION)
        - --schema-file parameter handling
        - runhistory list / runresult show endpoint routing

    All public functions have identical signatures regardless of mode. Scripts
    only need to:
    1. Dot-source this file instead of frontend-rest-helpers.ps1
    2. Pass -ApiMode "cli" to New-FrontendContext (or omit for REST default)

    Auth tokens are obtained via Get-FrontendToken (same priority chain as before):
        0. $TokenFile parameter (explicit persona token file)
        1. $env:CLEANROOM_FRONTEND_TOKEN (pre-set token)
        2. /tmp/msal-idtoken.txt (cached MSAL device-code IdToken)
        3. az account get-access-token (ARM token fallback)

    CLI mode auth uses MANAGEDCLEANROOM_ACCESS_TOKEN env var (priority 0 in the
    CLI extension) and AZURE_CLI_DISABLE_CONNECTION_VERIFICATION=1 for dogfood.

.NOTES
    REST API endpoint mapping (from CLI source _operations.py):

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

    CLI command mapping:

    az managedcleanroom frontend analytics dataset show -c $CID -d <name>
    az managedcleanroom frontend analytics dataset publish -c $CID -d <name> --storage-account-url ... --schema-file @file.json ...
    az managedcleanroom frontend analytics query show -c $CID -d <name>
    az managedcleanroom frontend analytics query publish -c $CID -d <name> --query-segment @seg.json ... --input-datasets "ds1:view1,ds2:view2" --output-dataset "ds:view"
    az managedcleanroom frontend analytics query vote -c $CID -d <name> --vote-action accept [--proposal-id <id>]
    az managedcleanroom frontend analytics query run -c $CID -d <name>
    az managedcleanroom frontend analytics query runresult show -c $CID --job-id <id>
    az managedcleanroom frontend analytics query runhistory list -c $CID -d <name>
    az managedcleanroom frontend analytics auditevent list -c $CID
    az managedcleanroom frontend consent set -c $CID -d <name> --consent-action enable
#>

# MSAL public client application ID for managed cleanroom frontend authentication
$script:MsalClientId = "8a3849c1-81c5-4d62-b83e-3bb2bb11251a"

# =============================================================================
# Context initialization
# =============================================================================

function New-FrontendContext {
    <#
    .SYNOPSIS
        Creates a frontend context for API calls.
    .PARAMETER frontendEndpoint
        Frontend URL. May include /collaborations suffix (will be stripped).
    .PARAMETER apiVersion
        API version query parameter (REST mode only).
    .PARAMETER ApiMode
        API mode: "rest" (default) or "cli".
    #>
    param(
        [Parameter(Mandatory)]
        [string]$frontendEndpoint,
        [string]$apiVersion = "2026-03-01-preview",
        [ValidateSet("rest", "cli")]
        [string]$ApiMode = "rest"
    )

    $baseUrl = $frontendEndpoint.TrimEnd('/')
    if ($baseUrl.EndsWith('/collaborations')) {
        $baseUrl = $baseUrl.Substring(0, $baseUrl.Length - '/collaborations'.Length)
    }

    return @{
        baseUrl    = $baseUrl
        apiVersion = $apiVersion
        apiMode    = $ApiMode
    }
}

# =============================================================================
# Token management
# =============================================================================

function Get-FrontendToken {
    <#
    .SYNOPSIS
        Gets a bearer token for frontend API calls.
    .DESCRIPTION
        Priority:
        0. $TokenFile parameter — explicit token file path (for persona switching)
        1. $env:CLEANROOM_FRONTEND_TOKEN — pre-set token (e.g., MSAL IdToken)
        2. /tmp/msal-idtoken.txt — cached MSAL device-code IdToken
        3. az account get-access-token — ARM token fallback
    .PARAMETER TokenFile
        Optional path to a file containing a pre-generated MSAL IdToken.
        Enables persona switching (e.g., /tmp/msal-idtoken-collaboratorA.txt).
    #>
    param(
        [string]$TokenFile
    )

    # Priority 0: Explicit token file parameter (persona switching)
    if ($TokenFile) {
        if (-not (Test-Path $TokenFile)) {
            throw "TokenFile '$TokenFile' not found."
        }
        $fileToken = (Get-Content $TokenFile -Raw).Trim()
        if ($fileToken) {
            Write-Host "  [Auth] Using token from: $TokenFile" -ForegroundColor DarkGray
            return $fileToken
        }
        throw "TokenFile '$TokenFile' is empty."
    }

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
        throw "No frontend token available. Run MSAL device-code flow first:`n  `$t = Get-MsalToken -ClientId '$script:MsalClientId' -TenantId 'common' -Scopes 'User.Read' -DeviceCode`n  `$t.IdToken | Out-File /tmp/msal-idtoken.txt -NoNewline"
    }
    return $token
}

# =============================================================================
# CLI auth setup
# =============================================================================

function Initialize-CliAuth {
    <#
    .SYNOPSIS
        Configures the CLI extension for frontend calls.
    .DESCRIPTION
        Sets MANAGEDCLEANROOM_ACCESS_TOKEN and AZURE_CLI_DISABLE_CONNECTION_VERIFICATION
        env vars, then calls `az managedcleanroom frontend configure --endpoint`.
    #>
    param(
        [Parameter(Mandatory)][hashtable]$Context,
        [string]$TokenFile
    )

    $token = Get-FrontendToken -TokenFile $TokenFile

    $env:MANAGEDCLEANROOM_ACCESS_TOKEN = $token
    $env:AZURE_CLI_DISABLE_CONNECTION_VERIFICATION = "1"

    # The SDK URL templates already include /collaborations/ prefix, so the
    # configured endpoint must be the bare base URL (no /collaborations suffix).
    $endpoint = $Context.baseUrl
    Write-Host "  [CLI] Configuring endpoint: $endpoint" -ForegroundColor DarkGray
    az managedcleanroom frontend configure --endpoint $endpoint 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to configure CLI frontend endpoint."
    }
}

# =============================================================================
# Core REST caller (used by REST mode)
# =============================================================================

function Invoke-FrontendRest {
    <#
    .SYNOPSIS
        Makes an authenticated REST call to the frontend service.
    .PARAMETER Context
        Frontend context from New-FrontendContext.
    .PARAMETER Method
        HTTP method (GET, POST, PUT, DELETE).
    .PARAMETER Path
        Relative path after /collaborations.
    .PARAMETER Body
        Request body (will be converted to JSON if not already a string).
    .PARAMETER Description
        Human-readable description for logging.
    .PARAMETER TokenFile
        Optional path to a token file (for persona switching).
    #>
    param(
        [Parameter(Mandatory)]
        [hashtable]$Context,

        [Parameter(Mandatory)]
        [string]$Method,

        [Parameter(Mandatory)]
        [string]$Path,

        [object]$Body,

        [string]$Description = "",

        [string]$TokenFile
    )

    $token = Get-FrontendToken -TokenFile $TokenFile
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
        [string]$Description = "",
        [string]$TokenFile
    )

    try {
        return Invoke-FrontendRest -Context $Context -Method $Method -Path $Path -Body $Body -Description $Description -TokenFile $TokenFile
    }
    catch {
        return $null
    }
}

# =============================================================================
# CLI helper: Run an az command, returning $null instead of throwing on failure.
# =============================================================================

function Invoke-AzCliSafe {
    param([string[]]$Arguments)
    $savedPref = $PSNativeCommandUseErrorActionPreference
    $PSNativeCommandUseErrorActionPreference = $false
    $result = & az @Arguments 2>$null
    $PSNativeCommandUseErrorActionPreference = $savedPref
    if ($LASTEXITCODE -eq 0 -and $result) {
        return ($result | ConvertFrom-Json)
    }
    return $null
}

function Invoke-AzCli {
    <#
    .SYNOPSIS
        Runs an az CLI command, parses JSON output, throws on failure.
    #>
    param(
        [string[]]$Arguments,
        [string]$Description = ""
    )
    $logPrefix = if ($Description) { "$Description — " } else { "" }
    Write-Host "  ${logPrefix}az $($Arguments -join ' ')" -ForegroundColor DarkGray

    $result = & az @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        $errText = $result -join "`n"
        Write-Host "  ERROR: az command failed: $errText" -ForegroundColor Red
        throw "CLI command failed: az $($Arguments -join ' ')`n$errText"
    }
    if ($result) {
        try {
            return ($result | ConvertFrom-Json)
        } catch {
            # Not JSON output — return raw
            return $result
        }
    }
    return $null
}

# =============================================================================
# Dataset operations
# =============================================================================

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
        [Parameter(Mandatory)][string]$DocumentId,
        [string]$TokenFile
    )

    if ($Context.apiMode -eq "cli") {
        Initialize-CliAuth -Context $Context -TokenFile $TokenFile
        return Invoke-AzCliSafe @(
            "managedcleanroom", "frontend", "analytics", "dataset", "show",
            "--collaboration-id", $CollaborationId,
            "--document-id", $DocumentId
        )
    }

    # REST mode
    Invoke-FrontendRestSafe -Context $Context -Method "GET" `
        -Path "$CollaborationId/analytics/datasets/$DocumentId" `
        -Description "Show dataset '$DocumentId'" `
        -TokenFile $TokenFile
}

function Publish-FrontendDataset {
    <#
    .SYNOPSIS
        Publishes a dataset to the frontend.
        CLI: az managedcleanroom frontend analytics dataset publish
        API: POST /collaborations/{id}/analytics/datasets/{docId}/publish
    .PARAMETER Body
        Dataset specification body (hashtable or JSON string).
        For REST mode: sent directly as request body.
        For CLI mode: body fields are mapped to CLI parameters.
    #>
    param(
        [Parameter(Mandatory)][hashtable]$Context,
        [Parameter(Mandatory)][string]$CollaborationId,
        [Parameter(Mandatory)][string]$DocumentId,
        [Parameter(Mandatory)][object]$Body,
        [string]$TokenFile
    )

    if ($Context.apiMode -eq "cli") {
        Initialize-CliAuth -Context $Context -TokenFile $TokenFile

        # Parse body to extract CLI parameters
        $bodyObj = $Body
        if ($Body -is [string]) {
            $bodyObj = $Body | ConvertFrom-Json -AsHashtable
        }

        # Write schema to temp file for --schema-file
        $tempSchemaFile = [System.IO.Path]::GetTempFileName() + ".json"
        $bodyObj.datasetSchema | ConvertTo-Json -Depth 10 | Out-File -FilePath $tempSchemaFile -Encoding utf8

        $args = @(
            "managedcleanroom", "frontend", "analytics", "dataset", "publish",
            "--collaboration-id", $CollaborationId,
            "--document-id", $DocumentId,
            "--storage-account-url", $bodyObj.store.storageAccountUrl,
            "--container-name", $bodyObj.store.containerName,
            "--storage-account-type", $bodyObj.store.storageAccountType,
            "--encryption-mode", $(if ($bodyObj.store.encryptionMode) { $bodyObj.store.encryptionMode } else { "SSE" }),
            "--schema-file", "@$tempSchemaFile",
            "--schema-format", $bodyObj.datasetSchema.format,
            "--access-mode", $bodyObj.datasetAccessPolicy.accessMode,
            "--identity-name", $bodyObj.identity.name,
            "--identity-client-id", $bodyObj.identity.clientId,
            "--identity-tenant-id", $bodyObj.identity.tenantId,
            "--identity-issuer-url", $(if ($bodyObj.identity.issuerUrl) { $bodyObj.identity.issuerUrl } else { "https://cgs/oidc" })
        )

        # Add allowed fields if present
        if ($bodyObj.datasetAccessPolicy.allowedFields) {
            $fields = $bodyObj.datasetAccessPolicy.allowedFields -join ","
            $args += @("--allowed-fields", $fields)
        }

        # Add CPK-specific DEK/KEK arguments if present
        if ($bodyObj.dek) {
            $args += @("--dek-keyvault-url", $bodyObj.dek.keyVaultUrl)
            $args += @("--dek-secret-id", $bodyObj.dek.secretId)
        }
        if ($bodyObj.kek) {
            $args += @("--kek-keyvault-url", $bodyObj.kek.keyVaultUrl)
            $args += @("--kek-secret-id", $bodyObj.kek.secretId)
            if ($bodyObj.kek.maaUrl) {
                $args += @("--kek-maa-url", $bodyObj.kek.maaUrl)
            }
        }

        try {
            return Invoke-AzCli -Arguments $args -Description "Publish dataset '$DocumentId'"
        } finally {
            if (Test-Path $tempSchemaFile) { Remove-Item $tempSchemaFile -Force -ErrorAction SilentlyContinue }
        }
    }

    # REST mode
    Invoke-FrontendRest -Context $Context -Method "POST" `
        -Path "$CollaborationId/analytics/datasets/$DocumentId/publish" `
        -Body $Body `
        -Description "Publish dataset '$DocumentId'" `
        -TokenFile $TokenFile
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
        [ValidateSet("enable", "disable")][string]$Action = "enable",
        [string]$TokenFile
    )

    if ($Context.apiMode -eq "cli") {
        Initialize-CliAuth -Context $Context -TokenFile $TokenFile
        return Invoke-AzCli -Arguments @(
            "managedcleanroom", "frontend", "consent", "set",
            "--collaboration-id", $CollaborationId,
            "--document-id", $DocumentId,
            "--consent-action", $Action
        ) -Description "Set consent '$Action' on '$DocumentId'"
    }

    # REST mode
    $body = @{ consentAction = $Action }
    Invoke-FrontendRest -Context $Context -Method "PUT" `
        -Path "$CollaborationId/consent/$DocumentId" `
        -Body $body `
        -Description "Set consent '$Action' on '$DocumentId'" `
        -TokenFile $TokenFile
}

# =============================================================================
# Query operations
# =============================================================================

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
        [Parameter(Mandatory)][string]$DocumentId,
        [string]$TokenFile
    )

    if ($Context.apiMode -eq "cli") {
        Initialize-CliAuth -Context $Context -TokenFile $TokenFile
        return Invoke-AzCliSafe @(
            "managedcleanroom", "frontend", "analytics", "query", "show",
            "--collaboration-id", $CollaborationId,
            "--document-id", $DocumentId
        )
    }

    # REST mode
    Invoke-FrontendRestSafe -Context $Context -Method "GET" `
        -Path "$CollaborationId/analytics/queries/$DocumentId" `
        -Description "Show query '$DocumentId'" `
        -TokenFile $TokenFile
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
        [Parameter(Mandatory)][object]$Body,
        [string]$TokenFile
    )

    if ($Context.apiMode -eq "cli") {
        Initialize-CliAuth -Context $Context -TokenFile $TokenFile

        # Parse body
        $bodyObj = $Body
        if ($Body -is [string]) {
            $bodyObj = $Body | ConvertFrom-Json -AsHashtable
        }

        # Write each query segment to a temp file for --query-segment
        $tempSegmentFiles = @()
        $args = @(
            "managedcleanroom", "frontend", "analytics", "query", "publish",
            "--collaboration-id", $CollaborationId,
            "--document-id", $DocumentId,
            "--input-datasets", $bodyObj.inputDatasets,
            "--output-dataset", $bodyObj.outputDataset
        )

        foreach ($segment in $bodyObj.queryData) {
            $tempFile = [System.IO.Path]::GetTempFileName() + ".json"
            $segment | ConvertTo-Json -Depth 10 | Out-File -FilePath $tempFile -Encoding utf8
            $args += @("--query-segment", "@$tempFile")
            $tempSegmentFiles += $tempFile
        }

        try {
            return Invoke-AzCli -Arguments $args -Description "Publish query '$DocumentId'"
        } finally {
            foreach ($f in $tempSegmentFiles) {
                if (Test-Path $f) { Remove-Item $f -Force -ErrorAction SilentlyContinue }
            }
        }
    }

    # REST mode
    Invoke-FrontendRest -Context $Context -Method "POST" `
        -Path "$CollaborationId/analytics/queries/$DocumentId/publish" `
        -Body $Body `
        -Description "Publish query '$DocumentId'" `
        -TokenFile $TokenFile
}

function Invoke-FrontendQueryVoteAccept {
    <#
    .SYNOPSIS
        Votes to accept a query.
        CLI: az managedcleanroom frontend analytics query vote
        API: POST /collaborations/{id}/analytics/queries/{docId}/vote
    #>
    param(
        [Parameter(Mandatory)][hashtable]$Context,
        [Parameter(Mandatory)][string]$CollaborationId,
        [Parameter(Mandatory)][string]$DocumentId,
        [Parameter(Mandatory)][string]$ProposalId,
        [string]$TokenFile
    )

    if ($Context.apiMode -eq "cli") {
        Initialize-CliAuth -Context $Context -TokenFile $TokenFile
        return Invoke-AzCli -Arguments @(
            "managedcleanroom", "frontend", "analytics", "query", "vote",
            "--collaboration-id", $CollaborationId,
            "--document-id", $DocumentId,
            "--vote-action", "accept",
            "--proposal-id", $ProposalId
        ) -Description "Vote accept query '$DocumentId'"
    }

    # REST mode
    $body = @{ voteAction = "accept"; proposalId = $ProposalId }
    Invoke-FrontendRest -Context $Context -Method "POST" `
        -Path "$CollaborationId/analytics/queries/$DocumentId/vote" `
        -Body $body `
        -Description "Vote accept query '$DocumentId'" `
        -TokenFile $TokenFile
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
        [string]$RunId,
        [string]$TokenFile
    )

    if ($Context.apiMode -eq "cli") {
        Initialize-CliAuth -Context $Context -TokenFile $TokenFile
        $args = @(
            "managedcleanroom", "frontend", "analytics", "query", "run",
            "--collaboration-id", $CollaborationId,
            "--document-id", $DocumentId
        )
        # CLI auto-generates a run ID if not provided
        if ($RunId) {
            $args += @("--run-id", $RunId)
        }
        return Invoke-AzCli -Arguments $args -Description "Run query '$DocumentId'"
    }

    # REST mode
    $body = @{}
    if ($RunId) {
        $body.runId = $RunId
    } else {
        $body.runId = [guid]::NewGuid().ToString()
    }
    Invoke-FrontendRest -Context $Context -Method "POST" `
        -Path "$CollaborationId/analytics/queries/$DocumentId/run" `
        -Body $body `
        -Description "Run query '$DocumentId'" `
        -TokenFile $TokenFile
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
        [Parameter(Mandatory)][string]$JobId,
        [string]$TokenFile
    )

    if ($Context.apiMode -eq "cli") {
        Initialize-CliAuth -Context $Context -TokenFile $TokenFile
        return Invoke-AzCliSafe @(
            "managedcleanroom", "frontend", "analytics", "query", "runresult", "show",
            "--collaboration-id", $CollaborationId,
            "--job-id", $JobId
        )
    }

    # REST mode
    Invoke-FrontendRestSafe -Context $Context -Method "GET" `
        -Path "$CollaborationId/analytics/runs/$JobId" `
        -Description "Get run result for job '$JobId'" `
        -TokenFile $TokenFile
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
        [Parameter(Mandatory)][string]$DocumentId,
        [string]$TokenFile
    )

    if ($Context.apiMode -eq "cli") {
        Initialize-CliAuth -Context $Context -TokenFile $TokenFile
        return Invoke-AzCli -Arguments @(
            "managedcleanroom", "frontend", "analytics", "query", "runhistory", "list",
            "--collaboration-id", $CollaborationId,
            "--document-id", $DocumentId
        ) -Description "Get run history for query '$DocumentId'"
    }

    # REST mode
    Invoke-FrontendRest -Context $Context -Method "GET" `
        -Path "$CollaborationId/analytics/queries/$DocumentId/runs" `
        -Description "Get run history for query '$DocumentId'" `
        -TokenFile $TokenFile
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
        [Parameter(Mandatory)][string]$CollaborationId,
        [string]$TokenFile
    )

    if ($Context.apiMode -eq "cli") {
        Initialize-CliAuth -Context $Context -TokenFile $TokenFile
        return Invoke-AzCli -Arguments @(
            "managedcleanroom", "frontend", "analytics", "auditevent", "list",
            "--collaboration-id", $CollaborationId
        ) -Description "Get audit events"
    }

    # REST mode
    Invoke-FrontendRest -Context $Context -Method "GET" `
        -Path "$CollaborationId/analytics/auditevents" `
        -Description "Get audit events" `
        -TokenFile $TokenFile
}
