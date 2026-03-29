# E2E Test Findings: Azure Managed CleanRoom Analytics (Local Auth)

**Date**: March 24-27, 2026
**Environment**: Dogfood
**Auth**: Local user via `az login` + MSAL device-code IdToken for frontend
**Frontend**: `https://dogfood.workload-frontendwestus.cleanroom.cloudapp.azure-test.net`
**Collaborations tested**:
- `ashank9` (ID: `bfd9e441-792d-4b29-912a-5f2ebf43a9f3`) — blocked by CCF DNS issue (March 24-25)
- `saksham-e2e-2` (ID: `a0d5a85d-289f-4373-ada6-94aac059854e`) — **full E2E success** (March 26-27)

---

## Executive Summary

**Full end-to-end success achieved on `saksham-e2e-2` collaboration.** All scripts (04-15) pass. A Spark analytics query was successfully executed, reading 13,872 rows from two datasets and writing 248 aggregated rows to the output container. Total runtime was ~16 minutes including Spark pod startup.

The initial `ashank9` collaboration was blocked by a server-side CCF DNS issue (resolved as collaboration-specific, not systemic). A new collaboration `saksham-e2e-2` was created to bypass this, and the full flow was completed end-to-end.

All scripts use local `az login` credentials (no VMs, no managed identity boilerplate). Frontend calls use direct REST via `frontend-helpers.ps1` with MSAL IdTokens for MSA guest account compatibility. Two-persona auth uses separate MSAL token files (`-TokenFile` parameter).

### Script Status (as of `saksham-e2e-2`)

| Script | Status | Notes |
|--------|--------|-------|
| 04-prepare-resources | PASS | Both personas, resources created in `dd6ae7e0-...` subscription |
| 05-prepare-data-sse | PASS | Both personas, CSVs uploaded |
| 06-setup-identity | PASS | OIDC issuer on whitelisted `cleanroomoidc` SA |
| 07-grant-access | PASS | Federated credentials with capital `Analytics-` subject |
| 08-publish-dataset-sse | PASS | 3 datasets published with correct `issuerUrl` (public OIDC URL) |
| 09-publish-query | PASS | Query published with 3 SQL segments |
| 10-vote-query | PASS | Both personas voted accept |
| 11-run-query | PASS | Query COMPLETED — 13,872 rows in, 248 rows out, 968s |
| 12-view-results | PASS | Run history + audit events retrieved |
| 13-run-status | PASS | Job status polling with state tracking |
| 14-run-history | PASS | Run history with stats and summary |
| 15-audit-events | PASS | Audit events with Spark lifecycle tracking |

---

## Resolved Blocker: CCF Backend Unreachable (ashank9 only)

> **Status: RESOLVED** — This was specific to the `ashank9` collaboration. The `saksham-e2e-2` collaboration works end-to-end without this issue.

### Symptoms (ashank9)

All frontend analytics operations returned:
```json
{"error":{"code":"ContractNotFound","message":"A contract with the specified id was not found."}}
```

### Root Cause (from frontend service logs)

The `ContractNotFound` error was **misleading**. The actual error was a DNS resolution failure. The frontend's CGS (Clean Room Governance Service) client tried to connect to the CCF (Confidential Consortium Framework) network and failed:

```
System.Net.Http.HttpRequestException: Name or service not known
  (lb-nw-ccf-ashank1-zdvnlbqphizpk.westus.azurecontainer.io:443)
```

The CGS retried 3 times with backoff, then failed at `ContractsController.ListContracts`. The frontend wrapped this as `ContractNotFound`.

### Key Observations

1. The CGS was trying to reach `ashank1` ACI endpoint, but collaboration `ashank9` was provisioned on **AKS** (`ashank9-aks-dns-gbz8df34.hcp.westus.azmk8s.io`). This suggested a stale/misconfigured CGS endpoint mapping.

2. The `GET /collaborations/{id}/analytics/cleanroompolicy` endpoint returned 200 with `{"proposalIds":[],"claims":{"claims":{}}}`, confirming the analytics workload was enabled. Only CGS-dependent operations failed.

3. The collaboration itself was healthy -- `GET /collaborations/{id}` returned `{"userStatus":"Active"}`.

### Resolution

This was a collaboration-specific server-side infrastructure issue. Creating a new collaboration (`saksham-e2e-2`) resolved the problem entirely. The `ashank9` collaboration was not repaired.

---

## Architecture: Local Auth Approach

### Design Principles

1. **No VMs needed** -- Run scripts directly from a developer workstation
2. **No managed identity code** -- Use `az login` credentials for ARM, MSAL IdToken for frontend
3. **No cloud switching for scripts 04-12** -- Only scripts 01-02 (collaboration creation via `Private.CleanRoom` RP) need `PrivateCleanroomAzureCloud`. Scripts 04-12 use standard ARM + direct frontend REST.
4. **Direct REST calls** -- The `az managedcleanroom frontend` CLI has a Python 3.13 bug; use `Invoke-RestMethod` instead

### Authentication Flow

```
ARM operations (storage, KV, MI, RBAC):
  az login -> az account get-access-token -> standard ARM endpoints

Frontend operations (datasets, queries, OIDC):
  MSAL device-code flow -> IdToken -> Bearer token in REST calls
  Token cached at /tmp/msal-idtoken.txt
```

### Token Resolution Priority (Get-FrontendToken)

1. `$env:CLEANROOM_FRONTEND_TOKEN` (environment variable override)
2. `/tmp/msal-idtoken.txt` (cached MSAL IdToken -- preferred for MSA accounts)
3. `az account get-access-token` (ARM fallback -- fails for MSA guests)

### Why MSAL IdToken Instead of ARM Access Token

The frontend service (`TokenUtilities.ExtractUserInfoFromToken` in C#) extracts user identity from JWT claims with this fallback chain:
```
preferred_username -> upn -> sub
```

For MSA-backed guest accounts (e.g., `notsaksham@gmail.com`), ARM access tokens lack `preferred_username` and `upn`. The code falls back to `sub`, which is a pairwise pseudonymous identifier (opaque base64 string), causing `InvalidUserIdentifier` errors.

MSAL IdTokens contain `preferred_username`, resolving the issue.

**To get an MSAL IdToken:**
```powershell
Install-Module MSAL.PS -Scope CurrentUser -Force
$token = Get-MsalToken -ClientId "8a3849c1-81c5-4d62-b83e-3bb2bb11251a" `
  -TenantId "common" -Scopes "User.Read" -DeviceCode
$token.IdToken | Out-File -FilePath "/tmp/msal-idtoken.txt" -NoNewline
```

---

## OIDC Issuer Setup for MSFT-Hosted Collaborations

### The Whitelisted Storage Account Requirement

When a collaboration is hosted in the **MSFT internal tenant** (`72f988bf-86f1-41af-91ab-2d7cd011db47`), the OIDC issuer URL must point to a **whitelisted** storage account for federated identity credential federation to work. Arbitrary storage accounts will be rejected.

### Pre-Provisioned Whitelisted SA

| Property | Value |
|---|---|
| SA Name | `cleanroomoidc` |
| Resource Group | `azcleanroom-ctest-rg` |
| Subscription | `AzureCleanRoom-NonProd` (`fccb68eb-8ccf-49a6-a69a-7ea3c2867e9c`) |
| Static Website URL | `https://cleanroomoidc.z22.web.core.windows.net` |

### OIDC Setup Steps

1. **Fetch JWKS from frontend**:
   ```
   GET /collaborations/{id}/oidc/keys?api-version=2026-03-01-preview
   ```

2. **Create OIDC discovery document** (`openid-configuration.json`):
   ```json
   {
     "issuer": "https://cleanroomoidc.z22.web.core.windows.net/{collaborationId}",
     "jwks_uri": "https://cleanroomoidc.z22.web.core.windows.net/{collaborationId}/openid/v1/jwks",
     "response_types_supported": ["id_token"],
     "subject_types_supported": ["public"],
     "id_token_signing_alg_values_supported": ["RS256"]
   }
   ```

3. **Upload to `$web` container** (requires `sakshamgarg@microsoft.com` login with `fccb68eb-...` subscription):
   ```bash
   az storage blob upload --account-name cleanroomoidc --container-name '$web' \
     --name "{collaborationId}/.well-known/openid-configuration" \
     --file ./openid-configuration.json --content-type "application/json" \
     --overwrite --auth-mode login

   az storage blob upload --account-name cleanroomoidc --container-name '$web' \
     --name "{collaborationId}/openid/v1/jwks" \
     --file ./jwks.json --content-type "application/json" \
     --overwrite --auth-mode login
   ```

4. **Verify public accessibility**:
   ```bash
   curl https://cleanroomoidc.z22.web.core.windows.net/{collaborationId}/.well-known/openid-configuration
   curl https://cleanroomoidc.z22.web.core.windows.net/{collaborationId}/openid/v1/jwks
   ```

5. **Register issuer URL with frontend** (as the collaboration member):
   ```
   POST /collaborations/{id}/oidc/setIssuerUrl?api-version=2026-03-01-preview
   Body: {"url": "https://cleanroomoidc.z22.web.core.windows.net/{collaborationId}"}
   ```

### OIDC Issuer Info Structure

After `setIssuerUrl`, the issuer info looks like:
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

Note: The top-level `issuerUrl` remains `null` when set by an MSA user. The per-tenant `tenantData.issuerUrl` is populated correctly. The top-level field may only be set by the collaboration owner.

---

## Collaboration Details

### saksham-e2e-2 (ACTIVE — Full E2E Success)

| Property | Value |
|---|---|
| Collaboration ID (frontend) | `a0d5a85d-289f-4373-ada6-94aac059854e` |
| Collaboration Name | `saksham-e2e-2` |
| ARM ID | `/subscriptions/fccb68eb-8ccf-49a6-a69a-7ea3c2867e9c/resourceGroups/ashank-collab/providers/Private.CleanRoom/Collaborations/saksham-e2e-2` |
| Analytics Endpoint | `https://analytics-yandcukdrpifc.westus.cloudapp.azure.com` |
| OIDC Issuer | `https://cleanroomoidc.z22.web.core.windows.net/a0d5a85d-289f-4373-ada6-94aac059854e` |
| ARM Endpoint | `https://eastus2euap.management.azure.com/` |
| ARM API Version | `2025-10-31-preview` |
| Frontend API Version | `2026-03-01-preview` |

#### Personas (Two-Persona Auth)

Both personas operate from a single machine using separate MSAL token files:

| Persona | Email | MSAL OID | Token File |
|---------|-------|----------|------------|
| notsaksham (Woodgrove) | `notsaksham@gmail.com` | `00000000-0000-0000-4e19-b6285189ceda` | `/tmp/msal-idtoken-notsaksham.txt` |
| collaboratorA (Northwind) | `anantshankar17@outlook.com` | `00000000-0000-0000-cf3f-4b93e1b60fc1` | `/tmp/msal-idtoken-collaboratorA.txt` |

ARM operations: `sakshamgarg@microsoft.com` (oid: `ccc3e123-cc1e-48d2-a429-8b3aa7d04d63`)

#### Published Datasets (with correct issuerUrl)

| Dataset | Version | State | Proposer | Mode | allowedFields |
|---------|---------|-------|----------|------|---------------|
| `northwind-input-csv2` | 2.161 | Accepted | collaboratorA | read | date, author, mentions |
| `woodgrove-input-csv2` | 2.167 | Accepted | notsaksham | read | date, author, mentions |
| `woodgrove-output-csv2` | 2.173 | Accepted | notsaksham | write | author, Number_Of_Mentions |

#### Published Query

| Property | Value |
|---|---|
| Query Name | `query2` |
| Version | 2.181 |
| State | Accepted |
| Proposal ID | `3cbc6ae333b544b4ade06fa2d4bfa664` |
| Proposer | notsaksham |
| Input Datasets | `northwind-input-csv2:publisher_data`, `woodgrove-input-csv2:consumer_data` |
| Output Dataset | `woodgrove-output-csv2:output` |

#### Successful Run

| Property | Value |
|---|---|
| Job ID | `cl-spark-9e985d9c-b03a-4484-9c79-ca428ee73a2c` |
| State | COMPLETED |
| Duration | 968s (~16 min including pod startup) |
| Rows Read | 13,872 (4,435 northwind + 9,437 woodgrove) |
| Rows Written | 248 |
| Output Path | `woodgrove-output/Analytics/2026-03-27/9e985d9c-b03a-4484-9c79-ca428ee73a2c/part-00000-*.csv` |
| Output Size | 3,290 bytes |
| Output Columns | `author`, `Number_Of_Mentions` (the `Restricted_Sum` column was filtered by `allowedFields` access policy) |

### ashank9 (BLOCKED — CCF DNS Issue)

| Property | Value |
|---|---|
| Collaboration ID (frontend) | `bfd9e441-792d-4b29-912a-5f2ebf43a9f3` |
| Collaboration Name | `ashank9` |
| ARM ID | `/subscriptions/fccb68eb-.../resourceGroups/ashank-collab/providers/Private.CleanRoom/Collaborations/ashank9` |
| Consortium | `ccf-ashank9` |
| Cluster Endpoint (AKS) | `ashank9-aks-dns-gbz8df34.hcp.westus.azmk8s.io` |
| Provisioning State | `Succeeded` |
| Owner | `ashank@microsoft.com` (tenant `72f988bf-...`) |
| Member | `notsaksham@gmail.com` (added via `AddCollaboratorWorkflow`) |

### Timeline (from RP logs)

- **10:34:41** — `EnableWorkloadWorkflow` submitted for `ashank9` (workload type: analytics)
- **10:55:22** — `AddCollaboratorWorkflow` submitted, adding `notsaksham@gmail.com`

---

## Created Azure Resources

All resources in subscription `dd6ae7e0-4013-486b-9aef-c51cf8eb840a` (Visual Studio Enterprise), logged in as `notsaksham@gmail.com`.

### Northwind (`cr-e2e-northwind-rg`)

| Resource | Name |
|---|---|
| Storage Account | `sadb24cc3401aceec9565f70` |
| Key Vault | `kv-db24cc3401aceec9565f7` |
| Managed Identity | `id-db24cc3401aceec9565f7` (clientId: `72c05481-166b-4a90-aac8-c79c401e8ccf`) |
| OIDC Storage | `oidcdb24cc3401aceec9565f` (non-whitelisted, replaced by `cleanroomoidc`) |
| Blob Container | `northwind-input` (4 date-partitioned CSVs) |
| Federated Credential | `analytics-northwind-federation` |

### Woodgrove (`cr-e2e-woodgrove-rg`)

| Resource | Name |
|---|---|
| Storage Account | `sa0b138fede55199698dbb89` |
| Key Vault | `kv-0b138fede55199698dbb8` |
| Managed Identity | `id-0b138fede55199698dbb8` (clientId: `a12d64d4-b5bd-4138-95c8-8c47f7a67425`) |
| OIDC Storage | `oidc0b138fede55199698dbb` (non-whitelisted, replaced by `cleanroomoidc`) |
| Blob Containers | `woodgrove-input` (4 CSVs), `woodgrove-output` (empty) |
| Federated Credential | `analytics-woodgrove-federation` |

---

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
    "issuerUrl": "https://cgs/oidc"
  }
}
```

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

**Consent** (`PUT .../consent/{docId}`): `{ "consentAction": "enable|disable" }`
**Vote** (`POST .../queries/{docId}/vote`): `{ "voteAction": "accept|reject" }`
**Query Run** (`POST .../queries/{docId}/run`): `{ "runId": "<uuid>" }`
**setIssuerUrl** (`POST .../oidc/setIssuerUrl`): `{ "url": "<issuer-url>" }`

---

## Critical Findings (from saksham-e2e-2 Testing)

### 1. CRITICAL BUG: Dataset `issuerUrl` Must Be Public OIDC URL

**Script**: `08-publish-dataset-sse.ps1` (line 122)
**Bug**: Hardcoded `issuerUrl = "https://cgs/oidc"` — this is an internal CGS hostname that AAD cannot validate.
**Fix applied**: Changed to `$script:oidcIssuerUrl` which reads from `generated/{rg}/issuer-url.txt`.

The correct value is the **public OIDC URL**, e.g.:
```
https://cleanroomoidc.z22.web.core.windows.net/{collaborationId}
```

**Root cause**: The identity sidecar in Spark pods uses this `issuerUrl` as the issuer claim when requesting tokens from AAD. AAD attempts to fetch `/.well-known/openid-configuration` from the issuer URL. When `issuerUrl` is `https://cgs/oidc`, AAD rejects with:
```
AADSTS700211: No matching federated identity record found for presented assertion issuer 'https://cgs/oidc'
```

**Workaround used**: Since you cannot modify a published dataset's `issuerUrl` (proposals are immutable once accepted), we created new datasets (`*-csv2`) with the correct issuerUrl rather than trying to modify accepted proposals.

### 2. Federated Credential Subject Case Sensitivity (`contractId`)

**Script**: `07-grant-access.ps1`
**Bug**: Default `$contractId` was `"analytics"` (lowercase), but the Spark agent uses `"Analytics"` (capital A) as the contract ID.

The federated credential subject is `{contractId}-{oid}`, so:
- **Wrong**: `analytics-00000000-0000-0000-cf3f-4b93e1b60fc1`
- **Correct**: `Analytics-00000000-0000-0000-cf3f-4b93e1b60fc1`

**Fix applied**: Changed default `$contractId` from `"analytics"` to `"Analytics"` in `07-grant-access.ps1`.

### 3. Run API Returns `id` Field, Not `jobId`

**Script**: `11-run-query.ps1`
**Bug**: Script looked only for `$response.jobId`, but the direct REST run API returns `id` field.

Response shape:
```json
{"status":"success","id":"cl-spark-{uuid}","dryRun":null,...}
```

**Fix applied**: Script now checks both `$response.id` and `$response.jobId`.

### 4. Run History API Returns 404 When No Terminal Runs Exist

The `GET /collaborations/{id}/analytics/queries/{docId}/runs` endpoint returns:
```json
{"error":{"code":"NotFound","message":"No run history found for query ID: ..."}}
```
when no runs have reached a terminal state (COMPLETED or FAILED). Scripts should handle this 404 as a non-error condition.

**Successful response shape**:
```json
{
  "queryId": "...",
  "latestRun": {
    "runId": "...",
    "startTime": "2026-03-27T...",
    "endTime": "2026-03-27T...",
    "isSuccessful": true,
    "error": null,
    "stats": { "rowsRead": 13872, "rowsWritten": 248 },
    "durationSeconds": 968
  },
  "runs": [ ... ],
  "summary": {
    "totalRuns": 1,
    "successfulRuns": 1,
    "failedRuns": 0,
    "totalRuntimeSeconds": 968,
    "avgDurationSeconds": 968.0,
    "totalRowsRead": 13872,
    "totalRowsWritten": 248
  }
}
```

### 5. Audit Events Response Structure

The `GET /collaborations/{id}/analytics/auditevents` endpoint returns events wrapped in a `value` array:
```json
{
  "value": [
    {
      "scope": "...",
      "id": "SparkApplicationSubmitted",
      "timestamp": 1743098765000,
      "timestampIso": "",
      "data": {
        "source": "analytics-agent",
        "message": "..."
      }
    }
  ],
  "nextLink": null
}
```

Note: `timestampIso` is an empty string (not populated). Use `timestamp` (epoch milliseconds) instead.

### 6. Spark Job Lifecycle and States

**State transitions observed**:
```
SUBMITTED → PENDING_RERUN → SUBMITTED (retry) → RUNNING → COMPLETED
```

**Audit event sequence (complete lifecycle)**:
1. `SparkApplicationSubmitted`
2. `SparkApplicationPendingRerun`
3. `SparkDriverRunning`
4. `SparkExecutorPending`
5. `SparkExecutorRunning`
6. `DATASET_LOAD_STARTED`
7. `DATASET_LOAD_COMPLETED`
8. `QUERY_SEGMENT_EXECUTION_STARTED`
9. `QUERY_SEGMENT_EXECUTION_COMPLETED`
10. `DATASET_WRITE_STARTED`
11. `DATASET_WRITE_COMPLETED`
12. `QUERY_EXECUTION_COMPLETED`
13. `QUERY_STATISTICS`
14. `SparkDriverCompleted`
15. `SparkExecutorCompleted`

### 7. Direct Analytics Endpoint (Bypass Frontend)

The analytics workload has a direct API endpoint that bypasses the frontend:

| Operation | Method | URL |
|---|---|---|
| Run query | POST | `https://analytics-yandcukdrpifc.westus.cloudapp.azure.com/queries/{queryId}/run` |
| Job status | GET | `https://analytics-yandcukdrpifc.westus.cloudapp.azure.com/status/{jobId}` |
| Run history | GET | `https://analytics-yandcukdrpifc.westus.cloudapp.azure.com/queries/{queryId}/runs` |

Auth header: `x-ms-cleanroom-authorization: Bearer <MSAL IdToken>`

### 8. Output Path Pattern

Query output is written to:
```
{container}/Analytics/{date}/{runId}/part-00000-{uuid}.csv
```

Example:
```
woodgrove-output/Analytics/2026-03-27/9e985d9c-b03a-4484-9c79-ca428ee73a2c/part-00000-...csv
```

### 9. Two-Persona Authentication

When running both personas from a single machine, use the `-TokenFile` parameter on frontend-calling scripts:

```powershell
# As collaboratorA (Northwind)
./scripts/10-vote-query.ps1 -collaborationId $collabId -queryName "query2" `
    -frontendEndpoint $frontend -TokenFile "/tmp/msal-idtoken-collaboratorA.txt"

# As notsaksham (Woodgrove)
./scripts/10-vote-query.ps1 -collaborationId $collabId -queryName "query2" `
    -frontendEndpoint $frontend -TokenFile "/tmp/msal-idtoken-notsaksham.txt"
```

Token files are generated via MSAL device-code flow (one login per persona):
```powershell
$token = Get-MsalToken -ClientId "8a3849c1-81c5-4d62-b83e-3bb2bb11251a" `
    -TenantId "common" -Scopes "User.Read" -DeviceCode
$token.IdToken | Out-File -FilePath "/tmp/msal-idtoken-collaboratorA.txt" -NoNewline
```

---

## Known Bugs

### 1. Python 3.13 CLI Compatibility

The `az managedcleanroom frontend` CLI extension fails on Python 3.13:
```
AttributeError: 'tuple' object has no attribute 'token'
```

`Profile.get_raw_token()` returns a tuple in Python 3.13, but `BearerTokenCredentialPolicy` expects an `AccessToken` object. **Workaround**: Direct REST calls via `frontend-helpers.ps1`.

### 2. InvalidUserIdentifier for MSA Guest Accounts

The frontend reads `sub` claim as fallback, which is an opaque pairwise ID for MSA accounts. **Workaround**: Use MSAL device-code flow IdTokens which contain `preferred_username`.

### 3. Misleading ContractNotFound Error

When the CCF backend is unreachable (DNS failure, container down), the frontend returns `ContractNotFound` instead of a connection error. The actual error is only visible in service logs.

---

## Script Execution Guide

### Prerequisites

1. `az login` with subscription having Contributor + User Access Administrator
2. MSAL IdToken cached at `/tmp/msal-idtoken.txt` (see MSAL section above)
3. `az extension add --name managedcleanroom` (for cloud registration)

### Running (Single User, Both Personas)

```powershell
az login  # as notsaksham@gmail.com
az account set --subscription "dd6ae7e0-4013-486b-9aef-c51cf8eb840a"

$collabId = "bfd9e441-792d-4b29-912a-5f2ebf43a9f3"
$frontend = "https://dogfood.workload-frontendwestus.cleanroom.cloudapp.azure-test.net"

# --- Per persona (northwind, then woodgrove) ---
./scripts/04-prepare-resources.ps1 -resourceGroup "cr-e2e-northwind-rg" -persona northwind
./scripts/05-prepare-data-sse.ps1 -resourceGroup "cr-e2e-northwind-rg" -persona northwind -dataDir "../demos/datasource/northwind"
./scripts/06-setup-identity.ps1 -resourceGroup "cr-e2e-northwind-rg" -persona northwind -collaborationId $collabId -frontendEndpoint $frontend
./scripts/07-grant-access.ps1 -resourceGroup "cr-e2e-northwind-rg" -collaborationId $collabId -userId northwind
./scripts/08-publish-dataset-sse.ps1 -collaborationId $collabId -resourceGroup "cr-e2e-northwind-rg" -persona northwind -frontendEndpoint $frontend

# Repeat 04-08 for woodgrove with -resourceGroup "cr-e2e-woodgrove-rg" -persona woodgrove

# --- Query (woodgrove publishes) ---
./scripts/09-publish-query.ps1 -collaborationId $collabId -queryName "query1" \
  -queryDir "../demos/query/woodgrove/query1" \
  -publisherInputDataset "northwind-input-csv" -consumerInputDataset "woodgrove-input-csv" \
  -outputDataset "woodgrove-output-csv" -frontendEndpoint $frontend

# --- Vote, Run, View ---
./scripts/10-vote-query.ps1 -collaborationId $collabId -queryName "query1" -frontendEndpoint $frontend -persona northwind
./scripts/11-run-query.ps1 -collaborationId $collabId -queryName "query1" -frontendEndpoint $frontend
./scripts/12-view-results.ps1 -collaborationId $collabId -queryName "query1" -frontendEndpoint $frontend
```

### File Dependency Chain

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
 |    PRODUCES: generated/$rg/jwks.json
 |    PRODUCES: generated/$rg/openid-configuration.json
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
```

---

## Data Schema

**Input CSV**: `date:date, time:string, author:string, mentions:string`
**Output CSV**: `author:string, Number_Of_Mentions:long, Restricted_Sum:number`

### Query SQL (3 segments)

- Segment 1 (seq 1): `CREATE OR REPLACE TEMP VIEW publisher_view AS SELECT * FROM publisher_data`
- Segment 2 (seq 1): `CREATE OR REPLACE TEMP VIEW consumer_view AS SELECT * FROM consumer_data`
- Segment 3 (seq 2): `SELECT author, COUNT(*) AS Number_Of_Mentions, SUM(mentions) AS Restricted_Sum FROM (SELECT * FROM publisher_view UNION ALL SELECT * FROM consumer_view) AS combine_data WHERE mentions LIKE '%MikeDoesBigData%' GROUP BY author ORDER BY Number_Of_Mentions DESC`

---

## Files Modified

### Common Helpers (New/Rewritten)

| File | Purpose |
|---|---|
| `scripts/common/setup-local-auth.ps1` | Replaces MI boilerplate. Verifies `az login`, sets `UsePrivateCleanRoomNamespace`, ensures AzureCloud. |
| `scripts/common/frontend-helpers.ps1` | All frontend API wrappers with MSAL IdToken priority chain, `-SkipCertificateCheck`, `api-version`. |
| `scripts/common/setup-oidc-issuer.ps1` | OIDC setup via direct REST (fetch keys, create discovery doc, upload, register issuer). |
| `scripts/common/prepare-resources.ps1` | Clean user vs SP detection (no IMDS). |
| `scripts/common/setup-access.ps1` | RBAC + federated credential setup. |

### E2E Scripts (All dot-source `setup-local-auth.ps1`)

| Script | Change |
|---|---|
| 04-prepare-resources.ps1 | MI boilerplate -> `setup-local-auth.ps1` |
| 05-prepare-data-sse.ps1 | MI boilerplate -> `setup-local-auth.ps1` |
| 06-setup-identity.ps1 | MI boilerplate -> `setup-local-auth.ps1` |
| 07-grant-access.ps1 | MI boilerplate -> `setup-local-auth.ps1`; **FIX: `$contractId` default `"analytics"` → `"Analytics"`** |
| 08-publish-dataset-sse.ps1 | Rewritten: CLI -> REST via `frontend-helpers.ps1`; **FIX: `issuerUrl` reads from `issuer-url.txt` instead of hardcoded `"https://cgs/oidc"`** |
| 09-publish-query.ps1 | Rewritten: CLI -> REST via `frontend-helpers.ps1` |
| 10-vote-query.ps1 | Rewritten: CLI -> REST via `frontend-helpers.ps1` |
| 11-run-query.ps1 | Rewritten: CLI -> REST via `frontend-helpers.ps1`; **FIX: checks both `id` and `jobId` fields in run response** |
| 12-view-results.ps1 | Rewritten: CLI -> REST via `frontend-helpers.ps1` |
| 13-run-status.ps1 | **NEW**: Check/poll specific job status by job ID |
| 14-run-history.ps1 | **NEW**: Standalone run history with 404 handling, stats parsing, summary display |
| 15-audit-events.ps1 | **NEW**: Standalone audit events with `value[]` wrapper parsing, timestamp formatting |
