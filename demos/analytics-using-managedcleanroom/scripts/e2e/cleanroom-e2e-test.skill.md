# Azure Managed Clean Room E2E Testing Skill (Local Auth)

## Description

This skill helps run and debug end-to-end (E2E) tests for the Azure Managed Clean Room analytics demo using **local user authentication** (no VMs, no managed identity boilerplate, no app registrations). Scripts 04-15 use `az login` credentials for ARM operations and MSAL device-code IdTokens for frontend REST calls.

## When to Use This Skill

Use this skill when you need to:

- Run or debug E2E tests for Azure Managed Clean Room analytics
- Understand the script execution flow and file dependency chain
- Troubleshoot frontend REST API errors (ContractNotFound, InvalidUserIdentifier, etc.)
- Set up MSAL IdToken authentication for MSA guest accounts
- Configure OIDC issuers for MSFT-hosted collaborations
- Publish datasets, queries, vote, run queries, or view results via the frontend
- Monitor Spark job lifecycle (status, run history, audit events)

## Architecture

### Auth Model

```
ARM operations (storage, KV, MI, RBAC):
  az login -> az account get-access-token -> standard ARM endpoints

Frontend operations (datasets, queries, OIDC):
  MSAL device-code flow -> IdToken -> Bearer token in REST calls
  Token cached at /tmp/msal-idtoken-{persona}.txt
```

**Two-persona auth from a single machine**: Use separate token files for each persona:
- `/tmp/msal-idtoken-notsaksham.txt` — `notsaksham@gmail.com` (Woodgrove)
- `/tmp/msal-idtoken-collaboratorA.txt` — `anantshankar17@outlook.com` (Northwind)

Scripts accept a `-TokenFile` parameter to select which persona's token to use for frontend calls.

**Why MSAL IdToken?** The frontend (`TokenUtilities.ExtractUserInfoFromToken`) reads claims: `preferred_username -> upn -> sub`. MSA guest accounts (e.g., `notsaksham@gmail.com`) lack `preferred_username` and `upn` in ARM access tokens, falling back to opaque `sub`. MSAL IdTokens include `preferred_username`, fixing this.

### Key Design Principles

1. **No VMs** -- Run scripts from a developer workstation
2. **No managed identity code** -- `setup-local-auth.ps1` replaces all MI boilerplate
3. **No cloud switching for scripts 04-15** -- Only scripts 01-02 (collaboration creation via `Private.CleanRoom` RP) need `PrivateCleanroomAzureCloud`
4. **Direct REST calls** -- The `az managedcleanroom frontend` CLI has a Python 3.13 bug (`'tuple' object has no attribute 'token'`); all frontend calls use `Invoke-RestMethod` via `frontend-helpers.ps1`
5. **No `createContract` call** -- Never call the createContract endpoint
6. **Single user, both personas** -- One `az login` session runs ARM ops; separate MSAL token files handle frontend auth per persona

### Frontend

- **Endpoint**: `https://dogfood.workload-frontendwestus.cleanroom.cloudapp.azure-test.net`
  - NO `/collaborations` suffix (`New-FrontendContext` strips it if present)
- **API version**: `2026-03-01-preview` (appended as `?api-version=` query param)
- **TLS**: `-SkipCertificateCheck` required (dogfood self-signed cert)
- **Auth**: `Authorization: Bearer {MSAL-IdToken}` header

### ARM

- **Endpoint**: `https://eastus2euap.management.azure.com/`
- **API version**: `2025-10-31-preview`
- **Subscription**: `fccb68eb-8ccf-49a6-a69a-7ea3c2867e9c` (AzureCleanRoom-NonProd)
- ARM operations require: `az account set --subscription fccb68eb-8ccf-49a6-a69a-7ea3c2867e9c`

### Direct Analytics Endpoint (bypass frontend)

The analytics workload has a direct API endpoint:

| Operation | Method | URL |
|---|---|---|
| Run query | POST | `https://analytics-{id}.westus.cloudapp.azure.com/queries/{queryId}/run` |
| Job status | GET | `https://analytics-{id}.westus.cloudapp.azure.com/status/{jobId}` |
| Run history | GET | `https://analytics-{id}.westus.cloudapp.azure.com/queries/{queryId}/runs` |

Auth: `x-ms-cleanroom-authorization: Bearer <MSAL IdToken>`

## Prerequisites

1. **Azure CLI** (2.50.0+) with `managedcleanroom` extension
2. **PowerShell Core 7.0+**
3. **`az login`** with Contributor + User Access Administrator on the target subscription
4. **MSAL IdTokens** cached at `/tmp/msal-idtoken-{persona}.txt` (see Token Setup below)
5. **Collaboration** already created (scripts 01-02 are separate; handled by collaboration owner)

### MSAL IdToken Setup (Per Persona)

```powershell
Install-Module MSAL.PS -Scope CurrentUser -Force

# Login as persona 1 (e.g., notsaksham@gmail.com)
$token = Get-MsalToken -ClientId "8a3849c1-81c5-4d62-b83e-3bb2bb11251a" `
  -TenantId "common" -Scopes "User.Read" -DeviceCode
$token.IdToken | Out-File -FilePath "/tmp/msal-idtoken-notsaksham.txt" -NoNewline

# Login as persona 2 (e.g., anantshankar17@outlook.com)
$token = Get-MsalToken -ClientId "8a3849c1-81c5-4d62-b83e-3bb2bb11251a" `
  -TenantId "common" -Scopes "User.Read" -DeviceCode
$token.IdToken | Out-File -FilePath "/tmp/msal-idtoken-collaboratorA.txt" -NoNewline
```

The token is resolved by `Get-FrontendToken` with this priority chain:
1. `-TokenFile` parameter (explicit per-persona token file)
2. `$env:CLEANROOM_FRONTEND_TOKEN` (environment variable override)
3. `/tmp/msal-idtoken.txt` (cached MSAL IdToken -- fallback)
4. `az account get-access-token` (ARM fallback -- fails for MSA guests)

## Working Directory

All paths are relative to:
```
demos/analytics-using-managedcleanroom/
```

## Script Files

### Common Helpers

| File | Purpose |
|---|---|
| `scripts/common/setup-local-auth.ps1` | Replaces MI boilerplate. Verifies `az login`, sets `UsePrivateCleanRoomNamespace`, ensures AzureCloud. Dot-sourced by every numbered script. |
| `scripts/common/frontend-helpers.ps1` | All frontend API wrappers with MSAL IdToken priority chain, `-SkipCertificateCheck`, `api-version`. |
| `scripts/common/prepare-resources.ps1` | Provisions resource group, storage account, Key Vault (premium), managed identity. Assigns RBAC. Outputs `names.generated.ps1` and `resources.generated.json`. |
| `scripts/common/setup-oidc-issuer.ps1` | OIDC setup: fetch JWKS from frontend, create discovery doc, upload to storage static website, register issuer URL via `setIssuerUrl`. |
| `scripts/common/setup-access.ps1` | RBAC assignments on storage/KV for managed identity + federated credential creation for OIDC token exchange. |

### E2E Scripts (04-15)

| Script | Persona | Purpose |
|---|---|---|
| `04-prepare-resources.ps1` | Both | Create RG, SA, KV, MI, RBAC |
| `05-prepare-data-sse.ps1` | Both | Upload CSVs to blob storage (SSE mode) |
| `06-setup-identity.ps1` | Both | Create OIDC issuer, save identity metadata |
| `07-grant-access.ps1` | Both | Federated credential + RBAC for cleanroom workload |
| `08-publish-dataset-sse.ps1` | Both | Publish dataset specs to frontend (northwind: input; woodgrove: input + output) |
| `09-publish-query.ps1` | Woodgrove | Publish Spark SQL query (3 segments) |
| `10-vote-query.ps1` | Both | Vote accept on published query |
| `11-run-query.ps1` | Woodgrove | Submit query run, poll for completion |
| `12-view-results.ps1` | Any | Display run history + audit events |
| `13-run-status.ps1` | Any | Check/poll specific job status by job ID |
| `14-run-history.ps1` | Any | Standalone run history with stats, 404 handling |
| `15-audit-events.ps1` | Any | Standalone audit events with Spark lifecycle tracking |

## Execution Flow

### Script Dependency Chain

```
04-prepare-resources.ps1 (per persona)
 |  PRODUCES: generated/$rg/names.generated.ps1
 |  PRODUCES: generated/$rg/resources.generated.json
 |
 +-> 05-prepare-data-sse.ps1 (per persona)
 |    CONSUMES: generated/$rg/names.generated.ps1
 |    PRODUCES: generated/datastores/$persona-datastore-metadata.json
 |
 +-> 06-setup-identity.ps1 (per persona)
 |    CONSUMES: generated/$rg/names.generated.ps1
 |    PRODUCES: generated/$rg/issuer-url.txt
 |    PRODUCES: generated/$rg/identity-metadata.json
 |    PRODUCES: generated/$rg/jwks.json, openid-configuration.json
 |
 +-> 07-grant-access.ps1 (per persona)
 |    CONSUMES: generated/$rg/issuer-url.txt
 |    CONSUMES: generated/$rg/names.generated.ps1
 |
 +-> 08-publish-dataset-sse.ps1 (per persona)
 |    CONSUMES: generated/$rg/names.generated.ps1
 |    CONSUMES: generated/datastores/$persona-datastore-metadata.json
 |    CONSUMES: generated/$rg/identity-metadata.json
 |
 +-> 09-publish-query.ps1 (woodgrove only)
 |    CONSUMES: query segment .txt files
 |
  +-> 10-vote-query.ps1 (both personas)
  +-> 11-run-query.ps1 (woodgrove only)
  +-> 12-view-results.ps1 (any persona)
  +-> 13-run-status.ps1 (any persona, requires jobId from 11)
  +-> 14-run-history.ps1 (any persona)
  +-> 15-audit-events.ps1 (any persona)
```

### Running (Single User, Both Personas)

```powershell
# Login and set subscription
az login  # as notsaksham@gmail.com
az account set --subscription "dd6ae7e0-4013-486b-9aef-c51cf8eb840a"

$collabId = "<collaboration-frontend-guid>"
$frontend = "https://dogfood.workload-frontendwestus.cleanroom.cloudapp.azure-test.net"

# --- Phase 1-4: Per persona (northwind, then woodgrove) ---
$persona = "northwind"
$rg = "cr-e2e-northwind-rg"

./scripts/04-prepare-resources.ps1 -resourceGroup $rg -persona $persona
./scripts/05-prepare-data-sse.ps1 -resourceGroup $rg -persona $persona `
    -dataDir "../demos/datasource/northwind"
./scripts/06-setup-identity.ps1 -resourceGroup $rg -persona $persona `
    -collaborationId $collabId -frontendEndpoint $frontend
./scripts/07-grant-access.ps1 -resourceGroup $rg -collaborationId $collabId `
    -userId $persona
./scripts/08-publish-dataset-sse.ps1 -collaborationId $collabId `
    -resourceGroup $rg -persona $persona -frontendEndpoint $frontend

# Repeat 04-08 for woodgrove with -resourceGroup "cr-e2e-woodgrove-rg" -persona woodgrove

# --- Phase 5: Publish query (woodgrove) ---
./scripts/09-publish-query.ps1 -collaborationId $collabId -queryName "query1" `
    -queryDir "../demos/query/woodgrove/query1" `
    -publisherInputDataset "northwind-input-csv" `
    -consumerInputDataset "woodgrove-input-csv" `
    -outputDataset "woodgrove-output-csv" `
    -frontendEndpoint $frontend

# --- Phase 6-8: Vote, Run, View ---
./scripts/10-vote-query.ps1 -collaborationId $collabId -queryName "query1" `
    -frontendEndpoint $frontend -persona northwind
./scripts/10-vote-query.ps1 -collaborationId $collabId -queryName "query1" `
    -frontendEndpoint $frontend -persona woodgrove
./scripts/11-run-query.ps1 -collaborationId $collabId -queryName "query1" `
    -frontendEndpoint $frontend
./scripts/12-view-results.ps1 -collaborationId $collabId -queryName "query1" `
    -frontendEndpoint $frontend
```

## Frontend REST API Reference

All calls require:
- `?api-version=2026-03-01-preview` query parameter
- `-SkipCertificateCheck` (dogfood self-signed cert)
- `Authorization: Bearer {MSAL-IdToken}` header

### Endpoints

| Operation | Method | Path |
|---|---|---|
| collaboration list | GET | `/collaborations` |
| collaboration show | GET | `/collaborations/{id}` |
| collaboration report | GET | `/collaborations/{id}/report` |
| analytics show | GET | `/collaborations/{id}/analytics` |
| cleanroom policy | GET | `/collaborations/{id}/analytics/cleanroompolicy` |
| dataset list | GET | `/collaborations/{id}/analytics/datasets` |
| dataset show | GET | `/collaborations/{id}/analytics/datasets/{docId}` |
| dataset publish | POST | `/collaborations/{id}/analytics/datasets/{docId}/publish` |
| dataset queries | GET | `/collaborations/{id}/analytics/datasets/{docId}/queries` |
| consent check | GET | `/collaborations/{id}/consent/{docId}` |
| consent set | PUT | `/collaborations/{id}/consent/{docId}` |
| query list | GET | `/collaborations/{id}/analytics/queries` |
| query show | GET | `/collaborations/{id}/analytics/queries/{docId}` |
| query publish | POST | `/collaborations/{id}/analytics/queries/{docId}/publish` |
| query vote | POST | `/collaborations/{id}/analytics/queries/{docId}/vote` |
| query run | POST | `/collaborations/{id}/analytics/queries/{docId}/run` |
| query runresult | GET | `/collaborations/{id}/analytics/runs/{jobId}` |
| query runhistory | GET | `/collaborations/{id}/analytics/queries/{docId}/runs` |
| audit events | GET | `/collaborations/{id}/analytics/auditevents` |
| oidc issuerinfo | GET | `/collaborations/{id}/oidc/issuerInfo` |
| oidc keys | GET | `/collaborations/{id}/oidc/keys` |
| oidc setIssuerUrl | POST | `/collaborations/{id}/oidc/setIssuerUrl` |
| invitation list | GET | `/collaborations/{id}/invitations` |
| invitation accept | POST | `/collaborations/{id}/invitations/{invId}/accept` |
| secret set | PUT | `/collaborations/{id}/analytics/secrets/{name}` |

### Request Body Structures

**Dataset Publish** (`POST .../datasets/{docId}/publish`):

Returns **204 No Content** on success (no response body).

```json
{
  "name": "<document_id>",
  "datasetSchema": { "format": "csv", "fields": [{"fieldName": "...", "fieldType": "..."}] },
  "datasetAccessPolicy": { "accessMode": "read|write", "allowedFields": ["..."] },
  "store": {
    "storageAccountUrl": "https://....blob.core.windows.net/",
    "containerName": "...",
    "storageAccountType": "Azure_BlobStorage",
    "encryptionMode": "SSE"
  },
  "identity": {
    "name": "...", "clientId": "...", "tenantId": "...",
    "issuerUrl": "https://cleanroomoidc.z22.web.core.windows.net/{collaborationId}"
  }
}
```

> **CRITICAL**: The `issuerUrl` in `identity` must be the **public OIDC URL** (e.g., `https://cleanroomoidc.z22.web.core.windows.net/{collaborationId}`), NOT `"https://cgs/oidc"`. Using `"https://cgs/oidc"` causes AAD token exchange failure: `AADSTS700211: No matching federated identity record found for presented assertion issuer`. This was a critical bug discovered during E2E testing — see `E2E-TEST-FINDINGS.md` for details.

**Query Publish** (`POST .../queries/{docId}/publish`):
```json
{
  "inputDatasets": "ds1:view1,ds2:view2",
  "outputDataset": "ds:view",
  "queryData": [
    { "data": "<SQL>", "executionSequence": 1, "preConditions": "", "postFilters": "" }
  ]
}
```

**Consent** (`PUT .../consent/{docId}`): `{ "consentAction": "enable|disable" }` — returns 204
**Vote** (`POST .../queries/{docId}/vote`): `{ "voteAction": "accept|reject", "proposalId": "..." }` — returns 204
**Query Run** (`POST .../queries/{docId}/run`): `{ "runId": "<uuid>" }` — returns 200
**setIssuerUrl** (`POST .../oidc/setIssuerUrl`): `{ "url": "<issuer-url>" }` — returns 200

### Response Structures

**Query Run Response** (`POST .../queries/{docId}/run`):
```json
{"status": "success", "id": "cl-spark-{uuid}", "dryRun": null}
```
Note: The field is `id`, NOT `jobId`. Scripts should check both for compatibility.

**Run History** (`GET .../queries/{docId}/runs`):
Returns 404 `NotFound` when no runs have reached terminal state. Success response:
```json
{
  "queryId": "...",
  "latestRun": {
    "runId": "...", "startTime": "...", "endTime": "...",
    "isSuccessful": true, "error": null,
    "stats": { "rowsRead": 13872, "rowsWritten": 248 },
    "durationSeconds": 968
  },
  "runs": [ ... ],
  "summary": {
    "totalRuns": 1, "successfulRuns": 1, "failedRuns": 0,
    "totalRuntimeSeconds": 968, "avgDurationSeconds": 968.0,
    "totalRowsRead": 13872, "totalRowsWritten": 248
  }
}
```

**Audit Events** (`GET .../auditevents`):
```json
{
  "value": [
    {
      "scope": "...", "id": "SparkApplicationSubmitted",
      "timestamp": 1743098765000, "timestampIso": "",
      "data": { "source": "analytics-agent", "message": "..." }
    }
  ],
  "nextLink": null
}
```

### Spark Job States

Known states: `SUBMITTED`, `RUNNING`, `COMPLETED`, `FAILED`, `SUBMISSION_FAILED`, `PENDING_RERUN`, `INVALIDATING`, `SUCCEEDING`, `FAILING`, `SUSPENDING`, `SUSPENDED`, `RESUMING`, `UNKNOWN`

Typical lifecycle: `SUBMITTED → PENDING_RERUN → SUBMITTED (retry) → RUNNING → COMPLETED`

### Output Path Pattern

Query output is written to:
```
{container}/Analytics/{date}/{runId}/part-00000-{uuid}.csv
```

## frontend-helpers.ps1 Functions

| Function | Description |
|---|---|
| `New-FrontendContext -Endpoint $url -CollaborationId $id` | Creates `$Context` hashtable (baseUrl, apiVersion). Strips `/collaborations` suffix if present. |
| `Get-FrontendToken` | Resolves token via priority chain (env var -> /tmp/msal-idtoken.txt -> az CLI) |
| `Invoke-FrontendRest -Context $ctx -Method GET -Path "/..." [-Body $obj]` | Core REST caller with Bearer auth, JSON body, error logging |
| `Invoke-FrontendRestSafe ...` | Non-throwing wrapper (returns `$null` on error) |
| `Get-FrontendDataset -Context $ctx -DocId $id` | Show dataset |
| `Publish-FrontendDataset -Context $ctx -DocId $id -Body $obj` | Publish dataset |
| `Set-FrontendConsent -Context $ctx -DocId $id -Action enable` | Enable/disable execution consent |
| `Get-FrontendQuery -Context $ctx -DocId $id` | Show query |
| `Publish-FrontendQuery -Context $ctx -DocId $id -Body $obj` | Publish query |
| `Invoke-FrontendQueryVoteAccept -Context $ctx -DocId $id` | Vote accept |
| `Invoke-FrontendQueryRun -Context $ctx -DocId $id` | Run query (auto-generates runId UUID) |
| `Get-FrontendQueryRunResult -Context $ctx -JobId $id` | Get run result |
| `Get-FrontendQueryRunHistory -Context $ctx -DocId $id` | Get run history |
| `Get-FrontendAuditEvents -Context $ctx` | Get audit events |

## OIDC Setup for MSFT-Hosted Collaborations

When a collaboration is hosted in the **MSFT internal tenant** (`72f988bf-...`), the OIDC issuer URL must point to a **whitelisted** storage account. Arbitrary storage accounts are rejected.

### Whitelisted SA

| Property | Value |
|---|---|
| SA Name | `cleanroomoidc` |
| Resource Group | `azcleanroom-ctest-rg` |
| Subscription | `AzureCleanRoom-NonProd` (`fccb68eb-8ccf-49a6-a69a-7ea3c2867e9c`) |
| Static Website URL | `https://cleanroomoidc.z22.web.core.windows.net` |

### OIDC Upload

Uploading OIDC docs requires logging in as `sakshamgarg@microsoft.com` with the `fccb68eb-...` subscription:

```powershell
# Switch to MSFT account
az login  # as sakshamgarg@microsoft.com
az account set --subscription "fccb68eb-8ccf-49a6-a69a-7ea3c2867e9c"

# Upload openid-configuration
az storage blob upload --account-name cleanroomoidc --container-name '$web' `
    --name "$collaborationId/.well-known/openid-configuration" `
    --file ./openid-configuration.json --content-type "application/json" `
    --overwrite --auth-mode login

# Upload JWKS
az storage blob upload --account-name cleanroomoidc --container-name '$web' `
    --name "$collaborationId/openid/v1/jwks" `
    --file ./jwks.json --content-type "application/json" `
    --overwrite --auth-mode login
```

### Issuer Info Structure

After `setIssuerUrl`, the issuer info for MSA users looks like:
```json
{
  "enabled": true,
  "issuerUrl": null,
  "tenantData": {
    "tenantId": "9188040d-6c67-4c5b-b112-36a304b66dad",
    "issuerUrl": "https://cleanroomoidc.z22.web.core.windows.net/{collaborationId}"
  }
}
```

The top-level `issuerUrl` remains `null` when set by an MSA user. The per-tenant `tenantData.issuerUrl` is populated correctly. The top-level field may only be set by the collaboration owner.

## Data Schema

**Input CSV fields**: `date:date, time:string, author:string, mentions:string`
**Output CSV fields**: `author:string, Number_Of_Mentions:long, Restricted_Sum:number`

### Query SQL (3 segments, in `demos/query/woodgrove/query1/`)

- **Segment 1** (seq 1): `CREATE OR REPLACE TEMP VIEW publisher_view AS SELECT * FROM publisher_data`
- **Segment 2** (seq 1): `CREATE OR REPLACE TEMP VIEW consumer_view AS SELECT * FROM consumer_data`
- **Segment 3** (seq 2): Joins both views, filters for `MikeDoesBigData` mentions, groups by author

## Troubleshooting

### ContractNotFound

**Symptom**: All frontend analytics operations (publish dataset, query, vote, run) return:
```json
{"error":{"code":"ContractNotFound","message":"A contract with the specified id was not found."}}
```

**Root cause**: Misleading error. The actual issue is a DNS resolution failure for the CCF backend. The CGS client cannot reach the CCF network endpoint. Check with the collaboration owner whether the CCF infrastructure (ACI or AKS) is running and the DNS mapping is correct.

**Diagnostic**: `GET /collaborations/{id}/analytics/cleanroompolicy` returns 200 if the workload is enabled. If this succeeds but publish/query operations fail, it's a CGS connectivity issue.

### InvalidUserIdentifier (MSA Accounts)

**Symptom**: Frontend returns `InvalidUserIdentifier` when using an ARM access token from an MSA guest account.

**Fix**: Use MSAL IdToken (see Token Setup section above).

### Python 3.13 CLI Bug

**Symptom**: `az managedcleanroom frontend` commands fail with `'tuple' object has no attribute 'token'`.

**Fix**: Use direct REST calls via `frontend-helpers.ps1` instead of the CLI.

### RBAC Propagation Delays

After `setup-access.ps1` assigns roles, there's a 90-second built-in wait plus a retry loop (10 x 15s) for verification. If federated credential operations still fail, wait longer and retry.

## Important Rules

1. **Never call `createContract`** -- this endpoint is not used in the local-auth flow
2. **Never switch cloud for scripts 04-15** -- only scripts 01-02 need `PrivateCleanroomAzureCloud`
3. **Never modify MI (managed identity) setup** -- MI is not needed for local auth
4. **Always use `-SkipCertificateCheck`** on all frontend REST calls
5. **Always include `?api-version=2026-03-01-preview`** on all frontend REST calls
6. **Idempotency**: All scripts are idempotent -- they check existence before creating resources/datasets/queries
7. **`issuerUrl` MUST be the public OIDC URL** (e.g., `https://cleanroomoidc.z22.web.core.windows.net/{collaborationId}`), NEVER `"https://cgs/oidc"` -- see Critical Findings in `E2E-TEST-FINDINGS.md`
8. **`contractId` is case-sensitive**: Use `"Analytics"` (capital A), not `"analytics"` -- the Spark agent uses capital A
9. **`securityPolicyOption` is NOT a valid input to `enableWorkload`** -- only `workloadType` is accepted
10. **Both parties must vote** on the query -- both datasets are required for execution
11. **Dataset publish returns 204 No Content** -- there is no response body on success
12. **Use `-TokenFile` parameter** for two-persona auth from a single machine
