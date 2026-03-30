# E2E Runbook: Azure Managed CleanRoom Analytics (SSE, Local Auth)

A complete step-by-step guide to running the analytics workflow end-to-end from a single developer machine. This runbook was validated against the `saksham-e2e-2` collaboration on March 27, 2026.

---

## Prerequisites

| Requirement | Details |
|---|---|
| Azure CLI | 2.75.0+ with `managedcleanroom` extension |
| PowerShell | 7.x+ |
| MSAL.PS module | `Install-Module MSAL.PS -Scope CurrentUser -Force` |
| Azure subscriptions | Personal subscription for storage/MI; MSFT subscription for collaboration & OIDC SA |
| Accounts | 2 Microsoft accounts (personal or work) — one per persona |

### Working Directory

All commands assume you are in:
```
demos/analytics-using-managedcleanroom/
```

---

## Phase 0: Authentication Setup

### 0.1 Login to Azure (ARM operations)

```powershell
az login --tenant "f880c6ca-fa2f-45ed-a89b-197f2e696868"     # Your tenant
az account set --subscription "dd6ae7e0-4013-486b-9aef-c51cf8eb840a"  # Your subscription
```

### 0.2 Generate MSAL Tokens (Frontend operations)

Generate a separate token file for each persona:

```powershell
# Persona 1 (e.g., notsaksham@gmail.com → Woodgrove)
$token = Get-MsalToken -ClientId "8a3849c1-81c5-4d62-b83e-3bb2bb11251a" `
    -TenantId "common" -Scopes "User.Read" -DeviceCode
$token.IdToken | Out-File -FilePath "/tmp/msal-idtoken-notsaksham.txt" -NoNewline

# Persona 2 (e.g., anantshankar17@outlook.com → Northwind)
$token = Get-MsalToken -ClientId "8a3849c1-81c5-4d62-b83e-3bb2bb11251a" `
    -TenantId "common" -Scopes "User.Read" -DeviceCode
$token.IdToken | Out-File -FilePath "/tmp/msal-idtoken-collaboratorA.txt" -NoNewline
```

**Verify**: Each `.txt` file should contain a JWT (three base64 segments separated by dots).

### 0.3 Set Variables

```powershell
$frontend = "https://dogfood.workload-frontendwestus.cleanroom.cloudapp.azure-test.net"
$armEndpoint = "https://eastus2euap.management.azure.com/"
$armApiVersion = "2025-10-31-preview"
$frontendApiVersion = "2026-03-01-preview"
$msftSubscription = "fccb68eb-8ccf-49a6-a69a-7ea3c2867e9c"

# Resource groups (will be created)
$northwindRg = "cr-e2e-northwind-rg"
$woodgroveRg = "cr-e2e-woodgrove-rg"
```

---

## Phase 1: Collaboration Setup (ARM)

> This phase requires access to the MSFT subscription (`fccb68eb-...`).

### 1.1 Create Collaboration

```powershell
az account set --subscription $msftSubscription

# Create collaboration via ARM REST
$collabName = "my-e2e-test"
$collabRg = "ashank-collab"  # or your own RG in the MSFT subscription

az rest --method PUT `
    --url "$armEndpoint/subscriptions/$msftSubscription/resourceGroups/$collabRg/providers/Private.CleanRoom/Collaborations/$collabName`?api-version=$armApiVersion" `
    --resource "https://management.azure.com/" `
    --body '{
        "location": "westus",
        "properties": {
            "consortiumType": "ConfidentialAKS",
            "userIdentity": {
                "tenantId": "<owner-tenant-id>",
                "objectId": "<owner-object-id>",
                "accountType": "MicrosoftAccount"
            }
        }
    }'
```

**Expected**: 201 Created or 200 OK with `provisioningState: "Succeeded"` (may need to poll for long-running operation).

### 1.2 Enable Analytics Workload

```powershell
az rest --method POST `
    --url "$armEndpoint/subscriptions/$msftSubscription/resourceGroups/$collabRg/providers/Private.CleanRoom/Collaborations/$collabName/enableWorkload`?api-version=$armApiVersion" `
    --resource "https://management.azure.com/" `
    --body '{"workloadType": "analytics"}'
```

> **IMPORTANT**: Only pass `workloadType`. Do NOT pass `securityPolicyOption` — it is not a valid parameter.

**Expected**: 202 Accepted (long-running operation). Poll the `Location` header URL until `provisioningState: "Succeeded"`.

### 1.3 Add Collaborators

```powershell
az rest --method POST `
    --url "$armEndpoint/subscriptions/$msftSubscription/resourceGroups/$collabRg/providers/Private.CleanRoom/Collaborations/$collabName/addCollaborator`?api-version=$armApiVersion" `
    --resource "https://management.azure.com/" `
    --body '{"email": "persona1@example.com"}'

az rest --method POST `
    --url "$armEndpoint/subscriptions/$msftSubscription/resourceGroups/$collabRg/providers/Private.CleanRoom/Collaborations/$collabName/addCollaborator`?api-version=$armApiVersion" `
    --resource "https://management.azure.com/" `
    --body '{"email": "persona2@example.com"}'
```

**Expected**: 202 Accepted for each.

### 1.4 Get Collaboration Frontend ID

> **NOTE**: The ARM response `properties.collaborationId` field is `null` (known bug).
> You must retrieve the frontend UUID via the frontend API instead.

```powershell
$collabShow = az rest --method GET `
    --url "$armEndpoint/subscriptions/$msftSubscription/resourceGroups/$collabRg/providers/Private.CleanRoom/Collaborations/$collabName`?api-version=$armApiVersion" `
    --resource "https://management.azure.com/" | ConvertFrom-Json

$frontendEndpoint = $collabShow.properties.workloads[0].endpoint
```

Get the frontend UUID (required for all `--collaboration-id` parameters):
```powershell
# Generate an MSAL token first (see Phase 0.2), then:
$env:MANAGEDCLEANROOM_ACCESS_TOKEN = Get-Content "/tmp/msal-idtoken-notsaksham.txt" -Raw
$env:AZURE_CLI_DISABLE_CONNECTION_VERIFICATION = "1"
$collabs = az managedcleanroom frontend collaboration list -o json | ConvertFrom-Json
$collabId = ($collabs | Where-Object { $_.name -eq $collabName }).id
```

### 1.5 Accept Invitations (Both Personas)

Each persona must accept their invitation via the frontend:

```powershell
./scripts/02-accept-invitation.ps1 `
    -collaborationId "/subscriptions/$msftSubscription/resourceGroups/$collabRg/providers/Private.CleanRoom/Collaborations/$collabName" `
    -frontendEndpoint $frontend
```

Run once per persona (logging in as each email when prompted for device code).

---

## Phase 2: Resource Provisioning (Scripts 04-07)

> Switch back to your personal subscription for these steps.

```powershell
az login --tenant "f880c6ca-fa2f-45ed-a89b-197f2e696868"
az account set --subscription "dd6ae7e0-4013-486b-9aef-c51cf8eb840a"
```

### 2.1 Prepare Resources (Both Personas)

```powershell
# Northwind
./scripts/04-prepare-resources.ps1 -resourceGroup $northwindRg -persona northwind

# Woodgrove
./scripts/04-prepare-resources.ps1 -resourceGroup $woodgroveRg -persona woodgrove
```

**Verify**: `scripts/generated/{rg}/names.generated.ps1` and `resources.generated.json` exist.

### 2.2 Upload Data (Both Personas)

> **NOTE**: The datasource directories are **not checked into the repo**. The
> `05-prepare-data-sse.ps1` script calls `common/get-input-data.ps1` which downloads
> Twitter CSV data from the `Azure-Samples/Synapse` GitHub repo at runtime. Create the
> `-dataDir` directories before running, or let the script create them automatically.

```powershell
# Northwind
./scripts/05-prepare-data-sse.ps1 -resourceGroup $northwindRg -persona northwind `
    -dataDir "./generated/datasource/northwind"

# Woodgrove
./scripts/05-prepare-data-sse.ps1 -resourceGroup $woodgroveRg -persona woodgrove `
    -dataDir "./generated/datasource/woodgrove"
```

**Verify**: `scripts/generated/datastores/{persona}-datastore-metadata.json` exists.

### 2.3 Setup OIDC Identity (Both Personas)

```powershell
# Northwind
./scripts/06-setup-identity.ps1 -resourceGroup $northwindRg -persona northwind `
    -collaborationId $collabId -frontendEndpoint $frontend

# Woodgrove
./scripts/06-setup-identity.ps1 -resourceGroup $woodgroveRg -persona woodgrove `
    -collaborationId $collabId -frontendEndpoint $frontend
```

**Verify**:
- `scripts/generated/{rg}/issuer-url.txt` contains a URL like `https://cleanroomoidc.z22.web.core.windows.net/{collabId}`
- `scripts/generated/{rg}/identity-metadata.json` contains `clientId` and `tenantId`

> **IMPORTANT for MSFT-hosted collaborations**: The OIDC issuer URL must point to a **whitelisted** storage account (`cleanroomoidc` in `azcleanroom-ctest-rg` on the MSFT subscription). Arbitrary storage accounts will be rejected by AAD.

### 2.4 Grant Access (Both Personas)

> **CRITICAL**: `-userId` must be the **JWT `oid` claim** from each persona's MSAL IdToken,
> NOT the persona name. Using persona names (e.g., `northwind`) creates a federated credential
> with subject `Analytics-northwind` which silently fails at runtime during token exchange.
>
> Extract the `oid` from the token:
> ```powershell
> $tokenB64 = (Get-Content "/tmp/msal-idtoken-collaboratorA.txt" -Raw).Split('.')[1]
> $padded = $tokenB64 + ('=' * (4 - $tokenB64.Length % 4) % 4)
> $claims = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($padded)) | ConvertFrom-Json
> $northwindOid = $claims.oid   # e.g., "00000000-0000-0000-4e19-b6285189ceda"
> ```

```powershell
# Northwind (use JWT oid from collaboratorA's token)
./scripts/07-grant-access.ps1 -resourceGroup $northwindRg `
    -collaborationId $collabId -contractId "Analytics" -userId $northwindOid

# Woodgrove (use JWT oid from notsaksham's token)
./scripts/07-grant-access.ps1 -resourceGroup $woodgroveRg `
    -collaborationId $collabId -contractId "Analytics" -userId $woodgroveOid
```

> **CRITICAL**: `contractId` must be `"Analytics"` (capital A). The Spark agent uses `Analytics-{oid}` as the federated credential subject. Lowercase `"analytics"` causes silent token exchange failures.

**Verify**: `az identity federated-credential list --resource-group {rg} --identity-name {mi-name}` shows credentials with:
- `issuer`: matches `issuer-url.txt`
- `subject`: `Analytics-{jwt-oid}` (e.g., `Analytics-00000000-0000-0000-4e19-b6285189ceda`)
- `audiences`: `["api://AzureADTokenExchange"]`

---

## Phase 3: Publish & Execute (Scripts 08-11)

### 3.1 Publish Datasets (Both Personas)

```powershell
# Northwind (as collaboratorA)
./scripts/08-publish-dataset-sse.ps1 -collaborationId $collabId `
    -resourceGroup $northwindRg -persona northwind `
    -frontendEndpoint $frontend `
    -TokenFile "/tmp/msal-idtoken-collaboratorA.txt"

# Woodgrove (as notsaksham)
./scripts/08-publish-dataset-sse.ps1 -collaborationId $collabId `
    -resourceGroup $woodgroveRg -persona woodgrove `
    -frontendEndpoint $frontend `
    -TokenFile "/tmp/msal-idtoken-notsaksham.txt"
```

**Expected**: HTTP 204 No Content for each dataset publish. The script publishes:
- Northwind: 1 input dataset (read)
- Woodgrove: 1 input dataset (read) + 1 output dataset (write)

> **CRITICAL**: The `issuerUrl` in the dataset body must be the public OIDC URL, NOT `"https://cgs/oidc"`. The script reads from `generated/{rg}/issuer-url.txt` automatically.

**Verify**: `GET /collaborations/{collabId}/analytics/datasets/{docId}` shows `state: "Accepted"`.

### 3.2 Publish Query (Woodgrove)

```powershell
./scripts/09-publish-query.ps1 -collaborationId $collabId `
    -queryName "query1" `
    -queryDir "../demos/query/woodgrove/query1" `
    -publisherInputDataset "northwind-input-csv" `
    -consumerInputDataset "woodgrove-input-csv" `
    -outputDataset "woodgrove-output-csv" `
    -frontendEndpoint $frontend `
    -TokenFile "/tmp/msal-idtoken-notsaksham.txt"
```

**Expected**: HTTP 204 for publish.

**Verify**: `GET /collaborations/{collabId}/analytics/queries/query1` shows the query with `state: "Proposed"`.

### 3.3 Vote on Query (Both Personas)

```powershell
# Northwind votes
./scripts/10-vote-query.ps1 -collaborationId $collabId -queryName "query1" `
    -frontendEndpoint $frontend `
    -TokenFile "/tmp/msal-idtoken-collaboratorA.txt"

# Woodgrove votes
./scripts/10-vote-query.ps1 -collaborationId $collabId -queryName "query1" `
    -frontendEndpoint $frontend `
    -TokenFile "/tmp/msal-idtoken-notsaksham.txt"
```

**Expected**: HTTP 204 for each vote.

**Verify**: After both votes, `GET .../queries/query1` shows `state: "Accepted"`.

### 3.4 Run Query (Woodgrove)

```powershell
./scripts/11-run-query.ps1 -collaborationId $collabId -queryName "query1" `
    -frontendEndpoint $frontend `
    -TokenFile "/tmp/msal-idtoken-notsaksham.txt"
```

**Expected**: HTTP 200 with response:
```json
{"status": "success", "id": "cl-spark-{uuid}", "dryRun": null}
```

The script will poll for completion. Typical runtime: **10-20 minutes** (includes Spark pod startup).

**Job state transitions**: `SUBMITTED → PENDING_RERUN → SUBMITTED → RUNNING → COMPLETED`

---

## Phase 4: Monitor & Validate (Scripts 13-15)

### 4.1 Check Run Status

```powershell
./scripts/13-run-status.ps1 -collaborationId $collabId `
    -jobId "cl-spark-{uuid}" `
    -frontendEndpoint $frontend `
    -TokenFile "/tmp/msal-idtoken-notsaksham.txt"
```

**Expected states**: `SUBMITTED` → `RUNNING` → `COMPLETED`

### 4.2 View Run History

```powershell
./scripts/14-run-history.ps1 -collaborationId $collabId `
    -queryName "query1" `
    -frontendEndpoint $frontend `
    -TokenFile "/tmp/msal-idtoken-notsaksham.txt"
```

**Expected response**:
```json
{
  "queryId": "query1",
  "latestRun": {
    "runId": "...",
    "isSuccessful": true,
    "stats": { "rowsRead": 13872, "rowsWritten": 248 },
    "durationSeconds": 968
  },
  "summary": {
    "totalRuns": 1,
    "successfulRuns": 1,
    "totalRowsRead": 13872,
    "totalRowsWritten": 248
  }
}
```

> **Note**: Returns HTTP 404 if no runs have reached terminal state. This is expected during polling.

### 4.3 View Audit Events

```powershell
./scripts/15-audit-events.ps1 -collaborationId $collabId `
    -frontendEndpoint $frontend `
    -TokenFile "/tmp/msal-idtoken-notsaksham.txt"
```

**Expected**: Spark lifecycle events in order:
1. `SparkApplicationSubmitted`
2. `SparkDriverRunning`
3. `SparkExecutorRunning`
4. `DATASET_LOAD_STARTED` / `DATASET_LOAD_COMPLETED`
5. `QUERY_SEGMENT_EXECUTION_STARTED` / `QUERY_SEGMENT_EXECUTION_COMPLETED`
6. `DATASET_WRITE_STARTED` / `DATASET_WRITE_COMPLETED`
7. `QUERY_EXECUTION_COMPLETED`
8. `QUERY_STATISTICS`
9. `SparkDriverCompleted` / `SparkExecutorCompleted`

### 4.4 Download Output CSV

```powershell
# Switch back to personal subscription
az login --tenant "f880c6ca-fa2f-45ed-a89b-197f2e696868"
az account set --subscription "dd6ae7e0-4013-486b-9aef-c51cf8eb840a"

# List output blobs
az storage blob list --account-name $woodgroveSA `
    --container-name woodgrove-output `
    --prefix "Analytics/" `
    --auth-mode login -o table

# Download the CSV
az storage blob download --account-name $woodgroveSA `
    --container-name woodgrove-output `
    --name "Analytics/{date}/{runId}/part-00000-{uuid}.csv" `
    --file ./output.csv --auth-mode login
```

**Expected output**: CSV with columns `author`, `Number_Of_Mentions`. The `Restricted_Sum` column is filtered by the `allowedFields` access policy on the output dataset.

---

## Appendix A: Troubleshooting

### AADSTS700211: No matching federated identity record

**Cause**: The `issuerUrl` in the published dataset is wrong (likely `"https://cgs/oidc"` instead of the public OIDC URL).

**Fix**: You cannot modify a published dataset. Create new datasets with the correct `issuerUrl` (read from `generated/{rg}/issuer-url.txt`). The script `08-publish-dataset-sse.ps1` now does this automatically.

### ContractNotFound on all analytics operations

**Cause**: The CCF backend DNS is unreachable (stale ACI endpoint mapping). This is collaboration-specific, not systemic.

**Fix**: Create a new collaboration. The `ashank9` collaboration had this issue; `saksham-e2e-2` did not.

### InvalidUserIdentifier

**Cause**: Using ARM access token from an MSA guest account (lacks `preferred_username` claim).

**Fix**: Use MSAL IdTokens (see Phase 0.2).

### Python 3.13 CLI Bug

**Symptom**: `'tuple' object has no attribute 'token'` on `az managedcleanroom frontend` commands.

**Fix**: Use direct REST calls via `frontend-helpers.ps1` (all scripts already do this).

### PENDING_RERUN State

**Cause**: The Spark pod may enter `PENDING_RERUN` during initial setup. This is normal — the system will retry automatically.

**Action**: Continue polling. The job will transition back to `SUBMITTED` and then `RUNNING`.

---

## Appendix B: API Response Reference

### Dataset Publish
- **Method**: `POST /collaborations/{id}/analytics/datasets/{docId}/publish`
- **Success**: 204 No Content (no body)

### Dataset Show
- **Method**: `GET /collaborations/{id}/analytics/datasets/{docId}`
- **Response**: JSON with `name`, `state`, `version`, `data.datasetSchema`, `data.datasetAccessPolicy`, `data.datasetAccessPoint`

### Query Run
- **Method**: `POST /collaborations/{id}/analytics/queries/{docId}/run`
- **Body**: `{ "runId": "<uuid>" }`
- **Response**: `{ "status": "success", "id": "cl-spark-{uuid}", "dryRun": null }`

### Run Result
- **Method**: `GET /collaborations/{id}/analytics/runs/{jobId}`
- **Response**: `{ "status": { "applicationState": { "state": "COMPLETED|RUNNING|..." } } }`

### Run History
- **Method**: `GET /collaborations/{id}/analytics/queries/{docId}/runs`
- **404 Response**: `{ "error": { "code": "NotFound", "message": "No run history found..." } }` (normal when no terminal runs)
- **200 Response**: `{ "queryId", "latestRun", "runs[]", "summary" }`

### Audit Events
- **Method**: `GET /collaborations/{id}/analytics/auditevents`
- **Response**: `{ "value": [{ "scope", "id", "timestamp", "data": { "source", "message" } }], "nextLink": null }`

### Consent
- **Method**: `PUT /collaborations/{id}/consent/{docId}`
- **Body**: `{ "consentAction": "enable" }`
- **Success**: 204 No Content

### Vote
- **Method**: `POST /collaborations/{id}/analytics/queries/{docId}/vote`
- **Body**: `{ "voteAction": "accept", "proposalId": "..." }`
- **Success**: 204 No Content
