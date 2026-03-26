# E2E Test Findings: Azure Managed CleanRoom Analytics (Local Auth)

**Date**: March 24-25, 2026
**Environment**: Dogfood
**Auth**: Local user via `az login` + MSAL device-code IdToken for frontend
**Frontend**: `https://dogfood.workload-frontendwestus.cleanroom.cloudapp.azure-test.net`
**Collaboration**: `ashank9` (ID: `bfd9e441-792d-4b29-912a-5f2ebf43a9f3`)

---

## Executive Summary

Scripts 04-07 pass successfully using local user auth. Scripts 08-12 are **blocked by a server-side CCF infrastructure issue** -- the CGS client inside the frontend cannot resolve the CCF network DNS name, returning `ContractNotFound` on all analytics operations.

All scripts have been rewritten to use local `az login` credentials (no VMs, no managed identity boilerplate). Frontend calls use direct REST via `frontend-rest-helpers.ps1` with MSAL IdTokens for MSA guest account compatibility.

| Script | Status | Notes |
|--------|--------|-------|
| 04-prepare-resources | PASS | Both personas, resources created in `dd6ae7e0-...` subscription |
| 05-prepare-data-sse | PASS | Both personas, CSVs uploaded |
| 06-setup-identity | PASS | OIDC issuers created, identity metadata saved |
| 07-grant-access | PASS | Federated credentials + RBAC configured |
| 08-publish-dataset-sse | BLOCKED | `ContractNotFound` (CCF backend DNS failure) |
| 09-publish-query | BLOCKED | Same |
| 10-vote-query | BLOCKED | Same |
| 11-run-query | BLOCKED | Same |
| 12-view-results | BLOCKED | Same |

---

## Current Blocker: CCF Backend Unreachable

### Symptoms

All frontend analytics operations return:
```json
{"error":{"code":"ContractNotFound","message":"A contract with the specified id was not found."}}
```

### Root Cause (from frontend service logs)

The `ContractNotFound` error is **misleading**. The actual error is a DNS resolution failure. The frontend's CGS (Clean Room Governance Service) client tries to connect to the CCF (Confidential Consortium Framework) network and fails:

```
System.Net.Http.HttpRequestException: Name or service not known
  (lb-nw-ccf-ashank1-zdvnlbqphizpk.westus.azurecontainer.io:443)
```

The CGS retries 3 times with backoff, then fails at `ContractsController.ListContracts`. The frontend wraps this as `ContractNotFound`.

### Key Observations

1. The CGS is trying to reach `ashank1` ACI endpoint, but collaboration `ashank9` was provisioned on **AKS** (`ashank9-aks-dns-gbz8df34.hcp.westus.azmk8s.io`). This suggests a stale/misconfigured CGS endpoint mapping.

2. The `GET /collaborations/{id}/analytics/cleanroompolicy` endpoint returns 200 with `{"proposalIds":[],"claims":{"claims":{}}}`, confirming the analytics workload IS enabled. Only CGS-dependent operations fail.

3. The collaboration itself is healthy -- `GET /collaborations/{id}` returns `{"userStatus":"Active"}`.

### Resolution

This is a server-side infrastructure issue. The collaboration owner (`ashank@microsoft.com`) needs to investigate why the CGS is mapped to a stale `ashank1` ACI endpoint instead of the `ashank9` AKS cluster.

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

### ashank9

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

## Known Bugs

### 1. Python 3.13 CLI Compatibility

The `az managedcleanroom frontend` CLI extension fails on Python 3.13:
```
AttributeError: 'tuple' object has no attribute 'token'
```

`Profile.get_raw_token()` returns a tuple in Python 3.13, but `BearerTokenCredentialPolicy` expects an `AccessToken` object. **Workaround**: Direct REST calls via `frontend-rest-helpers.ps1`.

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
| `scripts/common/frontend-rest-helpers.ps1` | All frontend API wrappers with MSAL IdToken priority chain, `-SkipCertificateCheck`, `api-version`. |
| `scripts/common/setup-oidc-issuer.ps1` | OIDC setup via direct REST (fetch keys, create discovery doc, upload, register issuer). |
| `scripts/common/prepare-resources.ps1` | Clean user vs SP detection (no IMDS). |
| `scripts/common/setup-access.ps1` | RBAC + federated credential setup. |

### E2E Scripts (All dot-source `setup-local-auth.ps1`)

| Script | Change |
|---|---|
| 04-prepare-resources.ps1 | MI boilerplate -> `setup-local-auth.ps1` |
| 05-prepare-data-sse.ps1 | MI boilerplate -> `setup-local-auth.ps1` |
| 06-setup-identity.ps1 | MI boilerplate -> `setup-local-auth.ps1` |
| 07-grant-access.ps1 | MI boilerplate -> `setup-local-auth.ps1` |
| 08-publish-dataset-sse.ps1 | Rewritten: CLI -> REST via `frontend-rest-helpers.ps1` |
| 09-publish-query.ps1 | Rewritten: CLI -> REST via `frontend-rest-helpers.ps1` |
| 10-vote-query.ps1 | Rewritten: CLI -> REST via `frontend-rest-helpers.ps1` |
| 11-run-query.ps1 | Rewritten: CLI -> REST via `frontend-rest-helpers.ps1` |
| 12-view-results.ps1 | Rewritten: CLI -> REST via `frontend-rest-helpers.ps1` |
