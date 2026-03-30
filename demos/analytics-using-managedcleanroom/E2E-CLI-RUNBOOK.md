# E2E CLI Runbook: Azure Managed CleanRoom Analytics (SSE + CPK)

A complete step-by-step guide to running an analytics collaboration end-to-end using
`az managedcleanroom frontend` CLI commands. Covers both **SSE** (Server-Side Encryption)
and **CPK** (Customer-Provided Key) data encryption modes.

**Validated runs**:
- **SSE**: `e2e-test-collab`, March 29, 2026. Query succeeded: 13,872 rows read, 697 rows written. Single user (`notsaksham@gmail.com`).
- **CPK**: `e2e-test-collab`, March 30, 2026. Query succeeded: job `cl-spark-25380698-88db-43d3-b9e9-47683a3a8fc8`, 13,872 rows read, 248 rows written. **Two real users** (`notsaksham@gmail.com` as Northwind, `anantshankar17@outlook.com` as Woodgrove). Full fresh run from Step 04 through Step 12, all scripts validated.

> **Audience**: Developer running the full E2E flow from a single machine.
> All frontend operations use the `az managedcleanroom` CLI extension.
> ARM operations (resource provisioning) use standard `az` CLI commands.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Architecture Overview](#architecture-overview)
- [Phase 0: Authentication Setup](#phase-0-authentication-setup)
- [Phase 1: Collaboration Setup (ARM)](#phase-1-collaboration-setup-arm)
- [Phase 2: Resource Provisioning](#phase-2-resource-provisioning)
- [Phase 3: OIDC Identity Setup](#phase-3-oidc-identity-setup)
- [Phase 4: Access & Federated Credentials](#phase-4-access--federated-credentials)
- [Phase 5: Publish Datasets](#phase-5-publish-datasets)
- [Phase 6: Publish & Approve Query](#phase-6-publish--approve-query)
- [Phase 7: Execute Query](#phase-7-execute-query)
- [Phase 8: Monitor & Validate](#phase-8-monitor--validate)
- [CPK Encryption Mode](#cpk-encryption-mode)
- [Appendix A: Federated Credential Subject Reference](#appendix-a-federated-credential-subject-reference)
- [Appendix B: Common Errors & Troubleshooting](#appendix-b-common-errors--troubleshooting)
- [Appendix C: CLI Command Reference](#appendix-c-cli-command-reference)
- [Appendix D: Known Bugs & Workarounds](#appendix-d-known-bugs--workarounds)

---

## Prerequisites

| Requirement | Details |
|---|---|
| Azure CLI | 2.75.0+ |
| `managedcleanroom` extension | Installed via `az extension add --name managedcleanroom` (v1.0.0b5+ with bug fixes) |
| PowerShell | 7.x+ (for running provisioning scripts) |
| MSAL.PS module | `Install-Module MSAL.PS -Scope CurrentUser -Force` |
| azcopy | v10+ (required for CPK mode — `azcopy copy --cpk-by-value`) |
| Python 3 + `cryptography` | Required for CPK mode — `pip install cryptography` |
| Azure subscriptions | One subscription for collaboration ARM resource; one for storage/MI resources |
| Microsoft accounts | One or more Microsoft accounts (MSA or AAD) — one per persona |
| OIDC storage account | A **whitelisted** storage account with static website enabled (e.g., `cleanroomoidc` in the MSFT subscription) |

### Inputs You Need Before Starting

| Input | Example | Where It Comes From |
|---|---|---|
| MSFT subscription ID | `fccb68eb-8ccf-49a6-a69a-7ea3c2867e9c` | Azure portal |
| Personal subscription ID | `dd6ae7e0-4013-486b-9aef-c51cf8eb840a` | Azure portal |
| Personal tenant ID | `f880c6ca-fa2f-45ed-a89b-197f2e696868` | Azure portal |
| Collaborator email(s) | `notsaksham@gmail.com` | Microsoft account |
| OIDC SA name | `cleanroomoidc` | Pre-provisioned, whitelisted |
| Data source directories | `./generated/datasource/northwind/`, `./generated/datasource/woodgrove/` | Downloaded at runtime by `05-prepare-data-*.ps1` (from `Azure-Samples/Synapse` GitHub repo) |
| Query segment files | `demos/query/woodgrove/query1/segment{1,2,3}.txt` | This repo |

### Working Directory

All commands assume you are in:
```
demos/analytics-using-managedcleanroom/
```

---

## Architecture Overview

```
┌─────────────────────────────┐     ┌─────────────────────────────────────────┐
│  Your Machine               │     │  Azure (MSFT Subscription)              │
│                             │     │                                         │
│  az login (ARM ops)         │────▶│  Private.CleanRoom/Collaborations       │
│  MSAL token (frontend ops)  │     │    └─ Analytics Workload (Spark)        │
│                             │     │    └─ Frontend API (CCF-backed)         │
│  az managedcleanroom        │────▶│    └─ OIDC SA (cleanroomoidc)           │
│    frontend ...             │     └─────────────────────────────────────────┘
│                             │
│  az identity/storage/role   │     ┌─────────────────────────────────────────┐
│    (standard ARM)           │────▶│  Azure (Personal Subscription)          │
│                             │     │    └─ Storage Accounts (input/output)   │
└─────────────────────────────┘     │    └─ Managed Identities               │
                                    │    └─ Key Vaults (CPK only)            │
                                    └─────────────────────────────────────────┘
```

**Data flow (SSE)**: The Spark workload runs inside a confidential AKS cluster. It uses
OIDC federated credentials to authenticate as the collaborator's managed identity,
reads input data from each collaborator's storage, executes the SQL query, and writes
results to the designated output container.

**Data flow (CPK)**: Same as SSE, but with an additional key management layer. At runtime,
the cleanroom releases each dataset's KEK via Secure Key Release (SKR), unwraps the DEK,
and passes it to Azure Storage via CPK headers. Azure Storage transparently decrypts data
on read and encrypts on write.

---

## Phase 0: Authentication Setup

There are two separate auth contexts:

| Context | Mechanism | Used For |
|---|---|---|
| ARM operations | `az login` (standard Azure CLI) | Resource provisioning, RBAC, federated credentials |
| Frontend operations | MSAL IdToken via env var | All `az managedcleanroom frontend` commands |

### 0.1 Login to Azure (ARM operations)

```bash
az login --tenant "<your-tenant-id>"
az account set --subscription "<your-personal-subscription-id>"
```

### 0.2 Generate MSAL IdToken (Frontend operations)

The frontend requires an MSAL IdToken (NOT an ARM access token). Generate one per persona:

```powershell
$token = Get-MsalToken -ClientId "8a3849c1-81c5-4d62-b83e-3bb2bb11251a" `
    -TenantId "common" -Scopes "User.Read" -DeviceCode
$token.IdToken | Out-File -FilePath "/tmp/msal-idtoken-persona1.txt" -NoNewline
```

This will print a device code URL. Open it in a browser and sign in with the persona's Microsoft account.

**Validation**:
```bash
# Should print a JWT (three dot-separated base64 segments)
cat /tmp/msal-idtoken-persona1.txt | head -c 50
```

**Token lifetime**: ~24 hours. Check expiry:
```bash
cat /tmp/msal-idtoken-persona1.txt | cut -d. -f2 | base64 -d 2>/dev/null | python3 -m json.tool | grep exp
```

> **IMPORTANT**: Note the `oid` claim from the token — you will need it for federated credentials in Phase 4.
> ```bash
> cat /tmp/msal-idtoken-persona1.txt | cut -d. -f2 | base64 -d 2>/dev/null | python3 -m json.tool | grep oid
> ```
> For MSA (personal Microsoft accounts), this is typically `00000000-0000-0000-XXXX-XXXXXXXXXXXX`.

### 0.3 Configure the CLI Extension

```bash
# Set the frontend endpoint (do this once)
az managedcleanroom frontend configure \
  --endpoint "https://dogfood.workload-frontendwestus.cleanroom.cloudapp.azure-test.net"
```

### 0.4 Frontend Auth Pattern

Every `az managedcleanroom frontend` command must be prefixed with these env vars:

```bash
MANAGEDCLEANROOM_ACCESS_TOKEN=$(cat /tmp/msal-idtoken-persona1.txt) \
AZURE_CLI_DISABLE_CONNECTION_VERIFICATION=1 \
az managedcleanroom frontend <subcommand> ...
```

| Env Var | Why |
|---|---|
| `MANAGEDCLEANROOM_ACCESS_TOKEN` | Passes the MSAL IdToken to the CLI extension (priority 0 in auth chain) |
| `AZURE_CLI_DISABLE_CONNECTION_VERIFICATION=1` | Disables SSL verification (dogfood server cert CN=`ccr-proxy` doesn't match hostname) |

> **NOTE**: Even with `MANAGEDCLEANROOM_ACCESS_TOKEN` set, a valid `az login` session is still
> required because the CLI extension calls `get_subscription_id(cmd.cli_ctx)` before checking
> the env var.

---

## Phase 1: Collaboration Setup (ARM)

> **Subscription**: MSFT subscription (or whichever subscription hosts the collaboration ARM resource).

### 1.1 Create Collaboration

```bash
az account set --subscription "<msft-subscription-id>"

COLLAB_NAME="my-e2e-collab"
COLLAB_RG="my-collab-rg"
ARM_ENDPOINT="https://eastus2euap.management.azure.com"
ARM_API="2025-10-31-preview"

az rest --method PUT \
  --url "$ARM_ENDPOINT/subscriptions/<msft-sub>/resourceGroups/$COLLAB_RG/providers/Private.CleanRoom/Collaborations/$COLLAB_NAME?api-version=$ARM_API" \
  --resource "https://management.azure.com/" \
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

**Expected output**: 201 Created or 200 OK with `provisioningState: "Succeeded"`.

**Wait time**: May return 202 Accepted. Poll the `Location` header until `provisioningState: "Succeeded"` (typically 2-5 minutes).

### 1.2 Enable Analytics Workload

```bash
az rest --method POST \
  --url "$ARM_ENDPOINT/subscriptions/<msft-sub>/resourceGroups/$COLLAB_RG/providers/Private.CleanRoom/Collaborations/$COLLAB_NAME/enableWorkload?api-version=$ARM_API" \
  --resource "https://management.azure.com/" \
  --body '{"workloadType": "analytics"}'
```

> **CRITICAL**: Only pass `workloadType`. Do NOT pass `securityPolicyOption` — it is not a valid parameter and will cause errors.

**Expected output**: 202 Accepted (long-running operation).

**Wait time**: Poll the `Location` header URL until complete. Typically **5-15 minutes**.

### 1.3 Add Collaborators

```bash
az rest --method POST \
  --url "$ARM_ENDPOINT/subscriptions/<msft-sub>/resourceGroups/$COLLAB_RG/providers/Private.CleanRoom/Collaborations/$COLLAB_NAME/addCollaborator?api-version=$ARM_API" \
  --resource "https://management.azure.com/" \
  --body '{"email": "collaborator@example.com"}'
```

Repeat for each collaborator. **Expected output**: 202 Accepted.

### 1.4 Get Collaboration Frontend UUID

```bash
COLLAB_SHOW=$(az rest --method GET \
  --url "$ARM_ENDPOINT/subscriptions/<msft-sub>/resourceGroups/$COLLAB_RG/providers/Private.CleanRoom/Collaborations/$COLLAB_NAME?api-version=$ARM_API" \
  --resource "https://management.azure.com/")

echo $COLLAB_SHOW | python3 -m json.tool
```

Save the frontend endpoint from `properties.workloads[0].endpoint`.

> **NOTE**: The `properties.collaborationId` field from the ARM response may be `null`.
> The frontend UUID must be obtained by listing collaborations via the frontend API:
> ```bash
> MANAGEDCLEANROOM_ACCESS_TOKEN=$(cat /tmp/msal-idtoken-persona1.txt) \
> AZURE_CLI_DISABLE_CONNECTION_VERIFICATION=1 \
> az managedcleanroom frontend collaboration list -o json
> ```
> Look for your collaboration name and note its `id` field (a UUID like `e9c5d770-7475-4add-948f-100966dcdaef`).

Save the UUID:
```bash
COLLAB_ID="<frontend-uuid>"
echo $COLLAB_ID > scripts/generated/collaboration-uuid.txt
```

> **CRITICAL**: ALL `--collaboration-id` parameters in frontend CLI commands take this
> **frontend UUID**, NOT the ARM resource ID. The ARM resource ID contains slashes that
> get URL-encoded by the SDK serializer (`%2F`), causing 404 errors.

### 1.5 Accept Invitations

Each collaborator must accept their invitation:

```bash
MANAGEDCLEANROOM_ACCESS_TOKEN=$(cat /tmp/msal-idtoken-persona1.txt) \
AZURE_CLI_DISABLE_CONNECTION_VERIFICATION=1 \
az managedcleanroom frontend invitation accept \
  --collaboration-id "$COLLAB_ID"
```

**Expected output**: JSON with collaboration details.

**Validation**: Check collaboration status:
```bash
MANAGEDCLEANROOM_ACCESS_TOKEN=$(cat /tmp/msal-idtoken-persona1.txt) \
AZURE_CLI_DISABLE_CONNECTION_VERIFICATION=1 \
az managedcleanroom frontend show \
  --collaboration-id "$COLLAB_ID" -o json
```
Should show `"status": "Finalized"` once all invitations are accepted.

---

## Phase 2: Resource Provisioning

> **Subscription**: Switch to your personal subscription.

```bash
az login --tenant "<your-tenant-id>"
az account set --subscription "<your-personal-subscription-id>"
```

### 2.1 Prepare Resources (Script)

Creates storage account, key vault (Premium SKU for CPK), and managed identity for each persona.

```powershell
# Northwind
./scripts/04-prepare-resources.ps1 -resourceGroup "cr-e2e-northwind-rg" -persona northwind

# Woodgrove
./scripts/04-prepare-resources.ps1 -resourceGroup "cr-e2e-woodgrove-rg" -persona woodgrove
```

**Expected output**: Resources created in Azure.

**Validation**: Check generated files exist:
```bash
cat generated/cr-e2e-northwind-rg/names.generated.ps1
# Should show: $STORAGE_ACCOUNT_NAME="sa...", $KEYVAULT_NAME="kv-...", $MANAGED_IDENTITY_NAME="id-..."

cat generated/cr-e2e-northwind-rg/resources.generated.json
# Should show JSON with resource details
```

### 2.2 Upload Data (Script)

Uploads CSV data to the storage containers.

```powershell
# Northwind
./scripts/05-prepare-data-sse.ps1 -resourceGroup "cr-e2e-northwind-rg" -persona northwind

# Woodgrove
./scripts/05-prepare-data-sse.ps1 -resourceGroup "cr-e2e-woodgrove-rg" -persona woodgrove
```

**Expected output**: Blobs uploaded.

**Validation**: Check datastore metadata:
```bash
cat generated/datastores/northwind-datastore-metadata.json | python3 -m json.tool
# Should show input dataset metadata (name, schema, storeUrl, containerName)

cat generated/datastores/woodgrove-datastore-metadata.json | python3 -m json.tool
# Should show input AND output dataset metadata
```

---

## Phase 3: OIDC Identity Setup

> **Subscription**: Personal subscription (for MI lookup), but OIDC document upload
> requires access to the whitelisted storage account.

### 3.1 Setup OIDC Issuer

This generates OIDC discovery documents and uploads them to the whitelisted storage account.

```powershell
# Northwind
./scripts/06-setup-identity.ps1 \
  -resourceGroup "cr-e2e-northwind-rg" \
  -persona northwind \
  -collaborationId "$COLLAB_ID" \
  -frontendEndpoint "https://dogfood.workload-frontendwestus.cleanroom.cloudapp.azure-test.net" \
  -OidcStorageAccount "cleanroomoidc" \
  -ApiMode "cli"

# Woodgrove
./scripts/06-setup-identity.ps1 \
  -resourceGroup "cr-e2e-woodgrove-rg" \
  -persona woodgrove \
  -collaborationId "$COLLAB_ID" \
  -frontendEndpoint "https://dogfood.workload-frontendwestus.cleanroom.cloudapp.azure-test.net" \
  -OidcStorageAccount "cleanroomoidc" \
  -ApiMode "cli"
```

> **Cross-Tenant Issue**: If the whitelisted storage account (`cleanroomoidc`) is in a
> different tenant than your `az login` session, the `az storage blob upload --auth-mode login`
> command will fail with "Issuer validation failed". In this case you need to:
> 1. Open a separate terminal
> 2. `az login --tenant <msft-tenant-id>` (browser-based login, NOT device code)
> 3. Run the OIDC upload from that terminal
> 4. Return to your original terminal for remaining steps

**Validation**:
```bash
# Check issuer URL was saved
cat generated/cr-e2e-northwind-rg/issuer-url.txt
# Should show: https://cleanroomoidc.z22.web.core.windows.net/<collab-uuid>

# Check identity metadata was saved
cat generated/cr-e2e-northwind-rg/identity-metadata.json | python3 -m json.tool
# Should show: clientId, tenantId, tokenIssuerUrl

# Verify OIDC documents are publicly accessible
curl -s "$(cat generated/cr-e2e-northwind-rg/issuer-url.txt)/.well-known/openid-configuration" | python3 -m json.tool
```

---

## Phase 4: Access & Federated Credentials

> **Subscription**: Personal subscription.

This step assigns RBAC roles to the managed identity and creates federated credentials
so the Spark workload can authenticate via OIDC token exchange.

### 4.1 Understanding the Federated Credential Subject

The federated credential `subject` must match what the Spark workload requests during
token exchange. The format is:

```
{contractId}-{ownerId}
```

Where:
- `contractId` = `"Analytics"` (capital A, case-sensitive)
- `ownerId` = the collaborator's object ID **as stored in the CCF governance ledger**

> **CRITICAL: The `ownerId` is the `oid` claim from the MSAL IdToken (JWT), NOT the
> Azure AD Graph API object ID.** These are different values for MSA (personal Microsoft)
> accounts:
>
> | Source | Value | Use? |
> |---|---|---|
> | `az ad signed-in-user show --query id` | Graph API object ID (e.g., `71c20d0b-...`) | **NO** |
> | JWT `oid` claim from MSAL IdToken | MSA passthrough OID (e.g., `00000000-0000-0000-4e19-...`) | **YES** |
> | Dataset `proposerId` field from frontend | Same as JWT `oid` | **YES** (confirmation) |
>
> The frontend stores the JWT `oid` as the owner ID when datasets are published. The
> Spark workload uses `dataset.OwnerId` (from `QueriesController.cs` line 355) to
> construct the token exchange subject.

Extract your `oid` from the token:
```bash
cat /tmp/msal-idtoken-persona1.txt | cut -d. -f2 | \
  python3 -c "import sys,base64,json; b=sys.stdin.read(); print(json.loads(base64.urlsafe_b64decode(b+'=='))['oid'])"
```

### 4.2 Grant Access (Script)

```powershell
# Get the oid from the JWT (see above)
$OID = "00000000-0000-0000-4e19-b6285189ceda"  # Replace with your JWT oid

# Northwind
./scripts/07-grant-access.ps1 \
  -resourceGroup "cr-e2e-northwind-rg" \
  -collaborationId "$COLLAB_ID" \
  -contractId "Analytics" \
  -userId "$OID"

# Woodgrove
./scripts/07-grant-access.ps1 \
  -resourceGroup "cr-e2e-woodgrove-rg" \
  -collaborationId "$COLLAB_ID" \
  -contractId "Analytics" \
  -userId "$OID"
```

> If you have different collaborators per persona (different Microsoft accounts), use
> each collaborator's JWT `oid` for their respective persona.

> **CPK Note**: For CPK mode, also run with `-setupKeyVault` to grant the managed identity
> `Key Vault Crypto User` and `Key Vault Secrets User` roles on the Key Vault:
> ```powershell
> ./scripts/07-grant-access.ps1 \
>   -resourceGroup "cr-e2e-northwind-rg" \
>   -collaborationId "$COLLAB_ID" \
>   -contractId "Analytics" \
>   -userId "$OID" \
>   -setupKeyVault
> ```

**Wait time**: RBAC role assignments take **60-120 seconds** to propagate. The script
waits 90 seconds and then retries verification up to 10 times.

**Validation**:
```bash
# Check federated credentials
az identity federated-credential list \
  --identity-name "<managed-identity-name>" \
  --resource-group "cr-e2e-northwind-rg" \
  -o table

# Should show:
# Name                         Issuer                                                          Subject
# ---------------------------  ------                                                          -------
# Analytics-<oid>-federation   https://cleanroomoidc.z22.web.core.windows.net/<collab-uuid>    Analytics-<oid>
```

**Verify the subject is correct**:
- Subject format: `Analytics-<jwt-oid>`
- NOT `Analytics-northwind` or `Analytics-<graph-api-oid>`

---

## Phase 5: Publish Datasets

> **Auth**: MSAL IdToken for the publishing persona.

### 5.1 Publish Northwind Input Dataset

```bash
MANAGEDCLEANROOM_ACCESS_TOKEN=$(cat /tmp/msal-idtoken-persona1.txt) \
AZURE_CLI_DISABLE_CONNECTION_VERIFICATION=1 \
az managedcleanroom frontend analytics dataset publish \
  --collaboration-id "$COLLAB_ID" \
  --document-id "northwind-input-csv" \
  --storage-account-url "https://<northwind-sa>.blob.core.windows.net" \
  --container-name "northwind-input" \
  --storage-account-type "blobstorage" \
  --encryption-mode "SSE" \
  --schema-file @/tmp/northwind-schema.json \
  --schema-format "csv" \
  --access-mode "read" \
  --allowed-fields "date,author,mentions" \
  --identity-name "northwind-identity" \
  --identity-client-id "<northwind-mi-client-id>" \
  --identity-tenant-id "<your-tenant-id>" \
  --identity-issuer-url "https://cleanroomoidc.z22.web.core.windows.net/<collab-uuid>"
```

> **Alternatively**, use the script which reads all metadata from generated files:
> ```powershell
> ./scripts/08-publish-dataset-sse.ps1 \
>   -collaborationId "$COLLAB_ID" \
>   -resourceGroup "cr-e2e-northwind-rg" \
>   -persona northwind \
>   -frontendEndpoint "https://dogfood.workload-frontendwestus.cleanroom.cloudapp.azure-test.net" \
>   -TokenFile "/tmp/msal-idtoken-persona1.txt" \
>   -ApiMode "cli"
> ```

**Expected output**: No output body (204 No Content on success). The CLI may show warnings — this is normal.

> **CRITICAL**: The `--identity-issuer-url` must be the **public OIDC URL**
> (e.g., `https://cleanroomoidc.z22.web.core.windows.net/<uuid>`), never `"https://cgs/oidc"`.

### 5.2 Publish Woodgrove Input + Output Datasets

Same pattern as above, substituting woodgrove values. Woodgrove publishes both:
- `woodgrove-input-csv` (read access)
- `woodgrove-output-csv` (write access)

### 5.3 Enable Execution Consent on Each Dataset

```bash
# For each dataset
MANAGEDCLEANROOM_ACCESS_TOKEN=$(cat /tmp/msal-idtoken-persona1.txt) \
AZURE_CLI_DISABLE_CONNECTION_VERIFICATION=1 \
az managedcleanroom frontend consent set \
  --collaboration-id "$COLLAB_ID" \
  --document-id "northwind-input-csv" \
  --consent-action "enable"
```

Repeat for `woodgrove-input-csv` and `woodgrove-output-csv`.

**Expected output**: No output body (204 No Content).

### 5.4 Validation

```bash
MANAGEDCLEANROOM_ACCESS_TOKEN=$(cat /tmp/msal-idtoken-persona1.txt) \
AZURE_CLI_DISABLE_CONNECTION_VERIFICATION=1 \
az managedcleanroom frontend analytics dataset show \
  --collaboration-id "$COLLAB_ID" \
  --document-id "northwind-input-csv" \
  -o json
```

**Expected**: `"state": "Accepted"`, with correct `identity.issuerUrl`, `store.containerName`, etc.

---

## Phase 6: Publish & Approve Query

### 6.1 Publish the Query

```bash
MANAGEDCLEANROOM_ACCESS_TOKEN=$(cat /tmp/msal-idtoken-persona1.txt) \
AZURE_CLI_DISABLE_CONNECTION_VERIFICATION=1 \
az managedcleanroom frontend analytics query publish \
  --collaboration-id "$COLLAB_ID" \
  --document-id "query1" \
  --input-datasets "northwind-input-csv:publisher_data,woodgrove-input-csv:consumer_data" \
  --output-dataset "woodgrove-output-csv:output" \
  --query-segment @/tmp/seg1.json \
  --query-segment @/tmp/seg2.json \
  --query-segment @/tmp/seg3.json
```

Where each segment JSON file contains:
```json
{
  "data": "<SQL statement>",
  "executionSequence": 1,
  "preConditions": "",
  "postFilters": ""
}
```

The demo query has 3 segments:
- Segment 1 (seq=1): `CREATE OR REPLACE TEMP VIEW publisher_view AS SELECT * FROM publisher_data`
- Segment 2 (seq=1): `CREATE OR REPLACE TEMP VIEW consumer_view AS SELECT * FROM consumer_data`
- Segment 3 (seq=2): `SELECT author, COUNT(*) AS Number_Of_Mentions, SUM(mentions) AS Restricted_Sum FROM (SELECT * FROM publisher_view UNION ALL SELECT * FROM consumer_view) AS combine_data WHERE mentions LIKE '%MikeDoesBigData%' GROUP BY author ORDER BY Number_Of_Mentions DESC`

Segments with the same `executionSequence` run in parallel; higher sequences run after lower ones complete.

> **Alternatively**, use the script:
> ```powershell
> ./scripts/09-publish-query.ps1 \
>   -collaborationId "$COLLAB_ID" \
>   -queryName "query1" \
>   -queryDir "demos/query/woodgrove/query1" \
>   -publisherInputDataset "northwind-input-csv" \
>   -consumerInputDataset "woodgrove-input-csv" \
>   -outputDataset "woodgrove-output-csv" \
>   -frontendEndpoint "https://dogfood.workload-frontendwestus.cleanroom.cloudapp.azure-test.net" \
>   -TokenFile "/tmp/msal-idtoken-persona1.txt" \
>   -ApiMode "cli"
> ```

**Expected output**: 204 No Content.

### 6.2 Vote to Accept the Query

First, get the proposal ID:
```bash
MANAGEDCLEANROOM_ACCESS_TOKEN=$(cat /tmp/msal-idtoken-persona1.txt) \
AZURE_CLI_DISABLE_CONNECTION_VERIFICATION=1 \
az managedcleanroom frontend analytics query show \
  --collaboration-id "$COLLAB_ID" \
  --document-id "query1" \
  -o json
```

Note the `proposalId` from the response.

Then vote:
```bash
MANAGEDCLEANROOM_ACCESS_TOKEN=$(cat /tmp/msal-idtoken-persona1.txt) \
AZURE_CLI_DISABLE_CONNECTION_VERIFICATION=1 \
az managedcleanroom frontend analytics query vote \
  --collaboration-id "$COLLAB_ID" \
  --document-id "query1" \
  --vote-action "accept" \
  --proposal-id "<proposal-id>"
```

**Expected output**: 204 No Content.

> **NOTE**: All collaborators must vote. If you are running with a single persona (same
> account owns all datasets), only one vote is needed.

### 6.3 Enable Execution Consent on the Query

```bash
MANAGEDCLEANROOM_ACCESS_TOKEN=$(cat /tmp/msal-idtoken-persona1.txt) \
AZURE_CLI_DISABLE_CONNECTION_VERIFICATION=1 \
az managedcleanroom frontend consent set \
  --collaboration-id "$COLLAB_ID" \
  --document-id "query1" \
  --consent-action "enable"
```

**Expected output**: 204 No Content.

> **IMPORTANT: Consent ordering with multiple voters.** Consent can only be enabled on a
> query that is in `Accepted` state. If the query is still `Proposed` (waiting for votes),
> `consent set --consent-action enable` will fail with `UserDocumentNotAccepted`. In a
> multi-party collaboration:
> 1. The first voter's consent enable will **fail** because their vote alone doesn't move
>    the query to `Accepted` state
> 2. The second (final) voter's vote moves the query to `Accepted`
> 3. The second voter can now enable consent successfully
> 4. The **first voter must re-run** their consent enable after the query reaches `Accepted`
>
> **Workaround**: Enable consent for all parties **after** all votes are in.

### 6.4 Validation

```bash
MANAGEDCLEANROOM_ACCESS_TOKEN=$(cat /tmp/msal-idtoken-persona1.txt) \
AZURE_CLI_DISABLE_CONNECTION_VERIFICATION=1 \
az managedcleanroom frontend analytics query show \
  --collaboration-id "$COLLAB_ID" \
  --document-id "query1" \
  -o json
```

**Expected**: `"state": "Accepted"` with `proposalId` populated.

---

## Phase 7: Execute Query

### 7.1 Submit the Query Run

```bash
MANAGEDCLEANROOM_ACCESS_TOKEN=$(cat /tmp/msal-idtoken-persona1.txt) \
AZURE_CLI_DISABLE_CONNECTION_VERIFICATION=1 \
az managedcleanroom frontend analytics query run \
  --collaboration-id "$COLLAB_ID" \
  --document-id "query1" \
  -o json
```

**Expected output**:
```json
{
  "status": "success",
  "id": "cl-spark-<uuid>",
  "dryRun": null,
  "jobIdField": null,
  "optimizationUsed": null,
  "reasoning": null,
  "skuSettings": null
}
```

Save the `id` value — this is the **job ID** used for monitoring.

```bash
JOB_ID="cl-spark-<uuid>"
```

> **NOTE**: `"status": "success"` means the run was **accepted for scheduling**, not that
> it completed. The Spark job takes 10-20 minutes to complete.

---

## Phase 8: Monitor & Validate

### 8.1 Check Run Status (Real-Time)

Use `runresult show` to see the live Spark application state:

```bash
MANAGEDCLEANROOM_ACCESS_TOKEN=$(cat /tmp/msal-idtoken-persona1.txt) \
AZURE_CLI_DISABLE_CONNECTION_VERIFICATION=1 \
az managedcleanroom frontend analytics query runresult show \
  --collaboration-id "$COLLAB_ID" \
  --job-id "$JOB_ID" \
  -o json
```

**State transitions** (typical timeline):

| Time | State | Events |
|---|---|---|
| +0 min | `SUBMITTED` | `SparkApplicationSubmitted` |
| +5-8 min | `RUNNING` | `SparkDriverRunning`, `SparkExecutorPending` |
| +8-12 min | `RUNNING` | `SparkExecutorRunning`, `DATASET_LOAD_STARTED` |
| +10-15 min | `RUNNING` | `DATASET_LOAD_COMPLETED`, `QUERY_SEGMENT_EXECUTION_*` |
| +15-20 min | `COMPLETED` | `DATASET_WRITE_COMPLETED`, `QUERY_EXECUTION_COMPLETED`, `SparkDriverCompleted` |

**Wait time**: **10-20 minutes** total. Poll every 30-60 seconds.

> **NOTE**: The `runresult show` endpoint shows live state from the Spark operator.
> The `runhistory list` endpoint only shows runs that have reached a terminal state
> (COMPLETED or FAILED). So during execution, use `runresult show`.

### 8.2 Check Run History (Terminal Runs)

After the run completes:

```bash
MANAGEDCLEANROOM_ACCESS_TOKEN=$(cat /tmp/msal-idtoken-persona1.txt) \
AZURE_CLI_DISABLE_CONNECTION_VERIFICATION=1 \
az managedcleanroom frontend analytics query runhistory list \
  --collaboration-id "$COLLAB_ID" \
  --document-id "query1" \
  -o json
```

**Expected output** (successful run):
```json
{
  "queryId": "query1",
  "latestRun": {
    "runId": "cl-spark-<uuid>",
    "isSuccessful": true,
    "startTime": "2026-03-29T19:46:50Z",
    "endTime": "2026-03-29T20:04:09Z",
    "durationSeconds": 1039,
    "stats": { "rowsRead": 13872, "rowsWritten": 697 },
    "error": null
  },
  "summary": {
    "totalRuns": 1,
    "successfulRuns": 1,
    "failedRuns": 0,
    "totalRowsRead": 13872,
    "totalRowsWritten": 697
  }
}
```

> **QUIRK: `--document-id` may need the `contractId` value, not the custom query name.**
> During CPK validation, `runhistory list --document-id Analytics-CPK-v4` returned
> "No run history found", while audit events showed the run completed with `"id": "Analytics"`.
> The backend may index runs by the internal `contractId` (`Analytics`) rather than the
> user-chosen query document ID. If `runhistory list` returns no results for a completed
> run, try using `--document-id Analytics` (the contract ID) instead.

### 8.3 View Audit Events

```bash
MANAGEDCLEANROOM_ACCESS_TOKEN=$(cat /tmp/msal-idtoken-persona1.txt) \
AZURE_CLI_DISABLE_CONNECTION_VERIFICATION=1 \
az managedcleanroom frontend analytics auditevent list \
  --collaboration-id "$COLLAB_ID" \
  -o json
```

**Expected output**: Array of audit events including `QUERY_COMPLETED_3003`.

### 8.4 Download Output Data

```bash
# List output blobs
az storage blob list \
  --account-name "<woodgrove-storage-account>" \
  --container-name "woodgrove-output" \
  --prefix "Analytics/" \
  --auth-mode login -o table

# Download the output CSV
az storage blob download \
  --account-name "<woodgrove-storage-account>" \
  --container-name "woodgrove-output" \
  --name "Analytics/<date>/<run-uuid>/part-00000-<uuid>.csv" \
  --file ./output.csv --auth-mode login

# View the results
cat ./output.csv
```

Output path pattern: `Analytics/{date}/{runId}/part-00000-{uuid}.csv`

---

## CPK Encryption Mode

This section documents the CPK (Customer-Provided Key) encryption flow as an alternative
to SSE. In CPK mode, each collaborator controls the encryption keys for their data. The
cleanroom can only access data after releasing the keys via Secure Key Release (SKR) in
a confidential compute environment that matches the collaboration's attestation policy.

### CPK vs SSE: Key Differences

| Aspect | SSE | CPK |
|---|---|---|
| Encryption | Azure-managed keys (transparent) | Customer-provided keys per dataset |
| Key Vault | Not required | Required (Premium SKU with HSM) |
| Upload tool | `az storage blob upload-batch` | `azcopy copy --cpk-by-value` |
| Output download | `az storage blob download` | `azcopy copy --cpk-by-value` (need DEK) |
| Key management | None | Per-dataset DEK + KEK with SKR policy |
| Data control | Trust Azure Storage encryption | Collaborator retains key control |

### CPK Concepts

- **DEK (Data Encryption Key)**: 32-byte random key used by Azure Storage for server-side
  encryption/decryption. Passed via CPK HTTP headers at upload/download time.
- **KEK (Key Encryption Key)**: RSA-2048 key stored in Azure Key Vault (HSM-backed,
  exportable). Has an SKR release policy tied to the collaboration's attestation hash.
- **Wrapped DEK**: The DEK encrypted (wrapped) with the KEK's public key using RSA-OAEP-SHA256.
  Stored as a Key Vault secret. The cleanroom unwraps it at runtime after releasing the KEK via SKR.
- **SKR (Secure Key Release)**: Mechanism that releases the KEK private key only to code
  running in a verified confidential compute environment matching the attestation policy.

### CPK Architecture

```
Upload time (your machine):
  plaintext CSV ──▶ azcopy --cpk-by-value ──▶ Azure Storage (encrypted at rest with DEK)
                         ▲
                    CPK_ENCRYPTION_KEY env var (base64 of DEK)

Key setup:
  DEK (32 bytes) ──▶ RSA-OAEP-SHA256 wrap with KEK pub ──▶ KV Secret (wrapped DEK)
  KEK (RSA-2048) ──▶ az keyvault key import (with SKR policy) ──▶ KV Key (exportable, HSM)

Runtime (cleanroom):
  SKR release ──▶ KEK private key ──▶ unwrap DEK ──▶ CPK header to Azure Storage ──▶ plaintext
```

### CRITICAL: CPK is Server-Side Encryption, NOT Client-Side

A common mistake is to manually encrypt files (e.g., AES-CBC) before uploading. This is
**wrong** and results in double-encrypted data that the cleanroom cannot read.

CPK mode means:
1. You upload **plaintext** files using `azcopy copy --cpk-by-value`
2. Azure Storage encrypts them **server-side** using the DEK you provide via HTTP headers
3. At read time, the caller provides the same DEK via CPK headers and Azure Storage decrypts transparently

The cleanroom's blobfuse layer handles the CPK headers automatically after releasing the KEK
and unwrapping the DEK.

### CPK Phase A: Prepare Data (Step 05)

This phase generates the DEK and uploads data using azcopy with CPK encryption.

> **NOTE**: The datasource directories are **not checked into the repo**. The data is
> downloaded at runtime by `05-prepare-data-cpk.ps1` (which calls `common/get-input-data.ps1`
> to fetch Twitter CSV data from the `Azure-Samples/Synapse` GitHub repo). The `-dataDir`
> path will be created automatically.

```powershell
# Northwind
./scripts/05-prepare-data-cpk.ps1 \
  -resourceGroup "cr-e2e-northwind-rg" \
  -persona northwind \
  -dataDir "./generated/datasource/northwind/input/csv" \
  -datasetSuffix "-cpk-v4"

# Woodgrove
./scripts/05-prepare-data-cpk.ps1 \
  -resourceGroup "cr-e2e-woodgrove-rg" \
  -persona woodgrove \
  -dataDir "./generated/datasource/woodgrove/input/csv" \
  -datasetSuffix "-cpk-v4"
```

**What this does**:
1. Generates a 32-byte random DEK, saved to `generated/datastores/keys/<dataset>-dek.bin`
2. Creates blob containers (`<persona>-input`, `<persona>-output` for woodgrove)
3. Uploads data files using `azcopy copy --cpk-by-value` with env vars:
   - `CPK_ENCRYPTION_KEY` = base64(DEK)
   - `CPK_ENCRYPTION_KEY_SHA256` = base64(SHA256(DEK))
4. Saves datastore metadata to `generated/datastores/<persona>-datastore-metadata.json`

**Per-dataset naming convention**:
- Input dataset: `<persona>-input-csv<suffix>` (e.g., `northwind-input-csv-cpk-v4`)
- Output dataset: `woodgrove-output-csv<suffix>` (e.g., `woodgrove-output-csv-cpk-v4`)
- KEK: `<dataset-name>-kek` (e.g., `northwind-input-csv-cpk-v4-kek`)
- Wrapped DEK secret: `wrapped-<dataset-name>-dek-<kek-name>`

**Validation**:
```bash
# Check DEK files were generated
ls -la generated/datastores/keys/

# Check metadata
cat generated/datastores/northwind-datastore-metadata.json | python3 -m json.tool

# Verify blobs are in storage (they should be CPK-encrypted)
az storage blob list \
  --account-name "<sa-name>" \
  --container-name "northwind-input" \
  --auth-mode login -o table
```

> **NOTE**: You cannot download CPK-encrypted blobs with `az storage blob download` — you
> must use `azcopy copy --cpk-by-value` with the correct DEK, or you'll get a 409 error.

### CPK Phase B: Publish Datasets + Create Keys (Step 08)

This phase publishes datasets, then creates per-dataset KEKs and wraps DEKs. This is
the **publish-first flow**: the SKR release policy is fetched from the just-published
dataset, ensuring the KEK is bound to the correct collaboration attestation policy.

```powershell
# Northwind (using northwind's token)
./scripts/08-publish-dataset-cpk.ps1 \
  -collaborationId "$COLLAB_ID" \
  -resourceGroup "cr-e2e-northwind-rg" \
  -persona northwind \
  -frontendEndpoint "https://dogfood.workload-frontendwestus.cleanroom.cloudapp.azure-test.net" \
  -TokenFile "/tmp/msal-idtoken-notsaksham.txt" \
  -ApiMode "cli"

# Woodgrove (using woodgrove's token)
./scripts/08-publish-dataset-cpk.ps1 \
  -collaborationId "$COLLAB_ID" \
  -resourceGroup "cr-e2e-woodgrove-rg" \
  -persona woodgrove \
  -frontendEndpoint "https://dogfood.workload-frontendwestus.cleanroom.cloudapp.azure-test.net" \
  -TokenFile "/tmp/msal-idtoken-anant.txt" \
  -ApiMode "cli"
```

**What this does** (3 sub-phases):
1. **B-1: Publish datasets** — Publishes dataset metadata with DEK/KEK references to the collaboration
2. **B-2: Create KEKs + wrap DEKs** — For each dataset:
   - Fetches SKR release policy from the just-published dataset (`skr-policy` endpoint)
   - Generates RSA-2048 key locally using Python `cryptography` library
   - Imports the key to Key Vault via `az keyvault key import` with `--exportable true --protection hsm --policy <skr-policy>`
   - Wraps the DEK with the KEK's public key using RSA-OAEP-SHA256 (client-side, NOT `az keyvault key encrypt`)
   - Stores the wrapped DEK as a Key Vault secret
3. **B-3: Enable execution consent** — Enables consent on all published datasets

**Why publish-first?** The SKR release policy contains the collaboration's `ccePolicyHash`
(attestation hash). This is only available from a published dataset via the frontend's
`skr-policy` endpoint. By publishing first, we don't need to hardcode or pre-fetch this hash.

**KEK creation details** (matches `az cleanroom` CLI implementation):
- Key is generated **locally** as RSA-2048 using Python `cryptography` lib
- Imported to Key Vault via `az keyvault key import` (NOT `az keyvault key create`)
- Ops: `encrypt wrapKey` (the cleanroom uses `unwrapKey` after SKR release)
- `--exportable true --protection hsm --immutable false`
- SKR release policy JSON attached via `--policy` parameter

**DEK wrapping details** (matches `az cleanroom` CLI implementation):
- Done **client-side** using Python `cryptography` lib
- Algorithm: RSA-OAEP with SHA-256 (both MGF1 and hash)
- NOT done via `az keyvault key encrypt` (server-side wrapping doesn't work for CPK flow)

**Validation**:
```bash
# Check KEK was created with correct properties
az keyvault key show --vault-name "<kv-name>" --name "<kek-name>" \
  --query '{exportable:attributes.exportable,keyType:key.kty,ops:key.keyOps}' -o json

# Check wrapped DEK secret exists
az keyvault secret show --vault-name "<kv-name>" --name "wrapped-<dataset>-dek-<kek>" \
  --query '{name:name,id:id}' -o json

# Check dataset was published
MANAGEDCLEANROOM_ACCESS_TOKEN=$(cat /tmp/msal-idtoken.txt) \
AZURE_CLI_DISABLE_CONNECTION_VERIFICATION=1 \
az managedcleanroom frontend analytics dataset show \
  --collaboration-id "$COLLAB_ID" \
  --document-id "<dataset-name>" -o json
```

### CPK Phase C: Query + Run (Steps 09-11)

The query publish, vote, and run steps are **identical** to SSE. Just use the CPK dataset
names in the query:

```powershell
# Publish query referencing CPK datasets
./scripts/09-publish-query.ps1 \
  -collaborationId "$COLLAB_ID" \
  -queryName "Analytics-CPK-v4" \
  -queryDir "demos/query/woodgrove/query1" \
  -publisherInputDataset "northwind-input-csv-cpk-v4" \
  -consumerInputDataset "woodgrove-input-csv-cpk-v4" \
  -outputDataset "woodgrove-output-csv-cpk-v4" \
  -frontendEndpoint "https://dogfood.workload-frontendwestus.cleanroom.cloudapp.azure-test.net" \
  -TokenFile "/tmp/msal-idtoken-anant.txt" \
  -ApiMode "cli"

# Vote (both collaborators)
./scripts/10-vote-query.ps1 ...

# Run
./scripts/11-run-query.ps1 ...
```

### CPK Phase D: Download Output

For CPK datasets, output is also CPK-encrypted. You need the output DEK to download:

```bash
# Using azcopy directly
export CPK_ENCRYPTION_KEY=$(python3 -c "import base64; print(base64.b64encode(open('generated/datastores/keys/woodgrove-output-csv-cpk-v4-dek.bin','rb').read()).decode())")
export CPK_ENCRYPTION_KEY_SHA256=$(python3 -c "import base64,hashlib; dek=open('generated/datastores/keys/woodgrove-output-csv-cpk-v4-dek.bin','rb').read(); print(base64.b64encode(hashlib.sha256(dek).digest()).decode())")
export AZCOPY_AUTO_LOGIN_TYPE="AZCLI"
export AZCOPY_TENANT_ID="<your-tenant-id>"

azcopy copy \
  "https://<woodgrove-sa>.blob.core.windows.net/woodgrove-output/*" \
  ./output/ \
  --recursive --cpk-by-value \
  --include-pattern "*.csv;*.crc;*_SUCCESS*"
```

> **NOTE**: The `--include-pattern` filter is required for storage accounts with Hierarchical
> Namespace (HNS) enabled. HNS creates directory marker blobs (zero-byte blobs with
> `hdi_isfolder=true` metadata) that cause azcopy to fail when it tries to download them
> as files into directories that already exist locally. The pattern filter skips these
> directory blobs and downloads only actual data files.

Or use the script with `-DownloadCpkOutput`:
```powershell
./scripts/12-view-results.ps1 \
  -collaborationId "$COLLAB_ID" \
  -queryName "Analytics-CPK-v4" \
  -frontendEndpoint "https://dogfood.workload-frontendwestus.cleanroom.cloudapp.azure-test.net" \
  -TokenFile "/tmp/msal-idtoken-anant.txt" \
  -ApiMode "cli" \
  -DownloadCpkOutput \
  -OutputDekFile "generated/datastores/keys/woodgrove-output-csv-cpk-v4-dek.bin" \
  -OutputStorageAccount "<woodgrove-sa-name>"
```

### CPK Troubleshooting

#### DATASET_LOAD_FAILED / java.io.IOException: Input/output error

**Cause**: Data was manually encrypted (e.g., AES-CBC) before upload instead of using
`azcopy --cpk-by-value`. The cleanroom's blobfuse layer can't read double-encrypted files.

**Fix**: Re-upload data using `azcopy copy --cpk-by-value` with a fresh DEK, then update
the wrapped DEK secret in Key Vault.

#### 409 BlobUsesCustomerSpecifiedEncryption on blob download

**Cause**: Trying to download CPK-encrypted blobs without providing the CPK headers.

**Fix**: Use `azcopy copy --cpk-by-value` with the correct DEK env vars to download.

#### KEK SKR release fails at runtime

**Symptoms**: Spark driver fails with SKR-related errors in audit events.

**Possible causes**:
1. KEK not exportable (`--exportable true` missing)
2. KEK not HSM-protected (`--protection hsm` missing)
3. SKR release policy doesn't match the collaboration's attestation policy
4. Wrong MAA URL in the dataset's `kek.maaUrl` field

**Fix**: Verify KEK properties:
```bash
az keyvault key show --vault-name "<kv>" --name "<kek>" \
  --query '{exportable:attributes.exportable,keyType:key.kty,releasePolicy:releasePolicy}' -o json
```

#### Wrong wrapped DEK value

**Symptoms**: SKR succeeds but data read fails (blobfuse can release KEK and unwrap DEK,
but the unwrapped DEK doesn't match what was used for upload).

**Fix**: Re-generate DEK, re-upload data with azcopy, re-wrap with KEK, and update the KV secret.

---

## Appendix A: Federated Credential Subject Reference

This is the **most common source of query failures**. Getting the subject wrong causes
a silent token exchange failure — the Spark driver exits with code 1 and 0 rows read.

### Subject Format

```
{contractId}-{ownerId}
```

### How the Subject is Constructed (Server-Side)

From `QueriesController.cs` line 355:
```csharp
string subject = string.Join("-", inputJob.ContractId, dataset.OwnerId);
```

- `ContractId` = The contract/analytics ID. For this workflow, always `"Analytics"` (capital A).
- `OwnerId` = The collaborator's object ID **as stored in the CCF governance ledger**.

### How OwnerId Gets Stored

When a dataset is published via the frontend API, the frontend extracts the `oid` claim
from the caller's JWT (MSAL IdToken) and stores it as `proposerId`/`ownerId` in the CCF ledger.

### MSA vs AAD Object IDs

For AAD (work/school) accounts, the JWT `oid` and Graph API object ID are typically the same.

For MSA (personal Microsoft) accounts, they differ:

| Source | Command | Typical Format |
|---|---|---|
| JWT `oid` (correct) | `python3 -c "..." < token.txt` | `00000000-0000-0000-XXXX-XXXXXXXXXXXX` |
| Graph API (wrong) | `az ad signed-in-user show --query id` | Standard GUID |

**Always use the JWT `oid`** for the federated credential subject.

### Verification

After publishing a dataset, verify what the frontend stored:
```bash
MANAGEDCLEANROOM_ACCESS_TOKEN=$(cat /tmp/msal-idtoken-persona1.txt) \
AZURE_CLI_DISABLE_CONNECTION_VERIFICATION=1 \
az managedcleanroom frontend analytics dataset show \
  --collaboration-id "$COLLAB_ID" \
  --document-id "northwind-input-csv" \
  -o json | python3 -c "import sys,json; print(json.load(sys.stdin).get('proposerId','NOT FOUND'))"
```

The `proposerId` value should match your JWT `oid`. The federated credential subject
should be `Analytics-<that-proposerId>`.

### Example

| Collaborator | JWT oid | Correct Subject |
|---|---|---|
| `notsaksham@gmail.com` | `00000000-0000-0000-4e19-b6285189ceda` | `Analytics-00000000-0000-0000-4e19-b6285189ceda` |

### Fixing Wrong Subjects

If you created federated credentials with the wrong subject (e.g., using persona names
like `Analytics-northwind`):

```bash
# Delete old
az identity federated-credential delete \
  --name "Analytics-northwind-federation" \
  --identity-name "<mi-name>" \
  --resource-group "cr-e2e-northwind-rg" \
  --yes

# Create new with correct subject
az identity federated-credential create \
  --name "Analytics-<oid>-federation" \
  --identity-name "<mi-name>" \
  --resource-group "cr-e2e-northwind-rg" \
  --issuer "https://cleanroomoidc.z22.web.core.windows.net/<collab-uuid>" \
  --subject "Analytics-<jwt-oid>" \
  --audiences "api://AzureADTokenExchange"
```

**Wait time**: Federated credential changes can take **1-5 minutes** to propagate.

---

## Appendix B: Common Errors & Troubleshooting

### SPARK_JOB_FAILED: driver container failed with ExitCode: 1

**Symptoms**: Query run shows `SPARK_JOB_FAILED`, 0 rows read, 0 rows written.

**Most likely cause**: Federated credential subject mismatch. The Spark driver can't
exchange the OIDC token for an Azure AD token to access storage.

**Fix**: See [Appendix A](#appendix-a-federated-credential-subject-reference). Delete
and recreate federated credentials with the correct subject (`Analytics-<jwt-oid>`).

**Diagnosis**: Compare the federated credential subject with the dataset's `proposerId`:
```bash
# What the federated credential expects
az identity federated-credential list \
  --identity-name "<mi-name>" \
  --resource-group "<rg>" \
  --query "[].subject" -o tsv

# What the frontend stored as the owner
# (check proposerId from dataset show output)
```

---

### AADSTS700211: No matching federated identity record

**Cause**: The `issuerUrl` in the published dataset is wrong (commonly `"https://cgs/oidc"`
instead of the public OIDC URL).

**Fix**: You cannot modify a published dataset. Republish with the correct issuer URL.
The correct URL is in `generated/<rg>/issuer-url.txt`.

---

### SSL certificate verify failed / hostname mismatch

**Cause**: Dogfood frontend server cert has CN=`ccr-proxy`, doesn't match the actual hostname.

**Fix**: Always set `AZURE_CLI_DISABLE_CONNECTION_VERIFICATION=1`. The `REQUESTS_CA_BUNDLE`
env var does NOT work because the issue is hostname mismatch, not an untrusted CA.

---

### 404 Not Found on frontend commands

**Cause**: Using the ARM resource ID instead of the frontend UUID for `--collaboration-id`.

**Fix**: Use the frontend UUID (e.g., `e9c5d770-7475-4add-948f-100966dcdaef`). Get it from:
```bash
MANAGEDCLEANROOM_ACCESS_TOKEN=... AZURE_CLI_DISABLE_CONNECTION_VERIFICATION=1 \
az managedcleanroom frontend collaboration list -o json
```

---

### "Issuer validation failed" on OIDC upload

**Cause**: Cross-tenant auth failure. Your `az login` is in tenant A, but the OIDC
storage account is in tenant B.

**Fix**: Login to the storage account's tenant in a separate terminal:
```bash
az login --tenant "<storage-account-tenant-id>"
```
Then run the OIDC upload from that terminal.

---

### ContractNotFound on all analytics operations

**Cause**: Stale CCF/ACI endpoint mapping. This is collaboration-specific.

**Fix**: Create a new collaboration.

---

### Python 3.13 `'tuple' object has no attribute 'token'`

**Cause**: Bug in the CLI extension's token handling on Python 3.13.

**Fix**: Upgrade to the patched `managedcleanroom` extension (v1.0.0b5+) which normalizes
the `AccessToken` namedtuple.

---

### "Already voted" / "Conflict" on query vote

**Cause**: Idempotent — the vote was already submitted.

**Action**: Safe to ignore. Verify query state shows `"state": "Accepted"`.

---

### Run submitted but not appearing in runhistory

**Cause**: The `runhistory list` endpoint only shows **terminal** runs (COMPLETED/FAILED).
In-progress runs only appear via `runresult show`.

**Fix**: Use `runresult show --job-id <id>` to check live status during execution.

---

### PENDING_RERUN state

**Cause**: Normal Spark scheduling behavior. The Spark operator may restart the application
during initial setup.

**Action**: Keep polling. It will transition to `SUBMITTED` → `RUNNING` → `COMPLETED`.

---

## Appendix C: CLI Command Reference

All commands require the env var prefix:
```bash
MANAGEDCLEANROOM_ACCESS_TOKEN=$(cat /tmp/msal-idtoken.txt) \
AZURE_CLI_DISABLE_CONNECTION_VERIFICATION=1 \
```

### Frontend Configuration

| Command | Description |
|---|---|
| `az managedcleanroom frontend configure --endpoint <url>` | Set frontend endpoint |
| `az managedcleanroom frontend login` | MSAL device code login (alternative to env var) |
| `az managedcleanroom frontend show -c <id>` | Show collaboration details |

### Dataset Operations

| Command | Description |
|---|---|
| `az managedcleanroom frontend analytics dataset show -c <id> -d <name>` | Show dataset |
| `az managedcleanroom frontend analytics dataset publish -c <id> -d <name> ...` | Publish dataset |

### Query Operations

| Command | Description |
|---|---|
| `az managedcleanroom frontend analytics query show -c <id> -d <name>` | Show query |
| `az managedcleanroom frontend analytics query publish -c <id> -d <name> ...` | Publish query |
| `az managedcleanroom frontend analytics query vote -c <id> -d <name> --vote-action accept --proposal-id <pid>` | Vote on query |
| `az managedcleanroom frontend analytics query run -c <id> -d <name>` | Run query |
| `az managedcleanroom frontend analytics query runresult show -c <id> --job-id <jid>` | Live run status |
| `az managedcleanroom frontend analytics query runhistory list -c <id> -d <name>` | Terminal run history |

### Consent Operations

| Command | Description |
|---|---|
| `az managedcleanroom frontend consent set -c <id> -d <name> --consent-action enable` | Enable consent |

### Audit

| Command | Description |
|---|---|
| `az managedcleanroom frontend analytics auditevent list -c <id>` | List audit events |

### Invitation

| Command | Description |
|---|---|
| `az managedcleanroom frontend invitation accept -c <id>` | Accept invitation |
| `az managedcleanroom frontend collaboration list` | List collaborations |

---

## Appendix D: Known Bugs & Workarounds

### 1. ~~`Invoke-AzCli` in `frontend-helpers.ps1` throws on stderr warnings~~ (FIXED)

**Bug**: When PowerShell scripts set `$PSNativeCommandUseErrorActionPreference = $true`
(which most numbered scripts do at line ~47), any output to stderr from `az` commands
causes a `NativeCommandExitException` — even when the command succeeds.

The `az` CLI writes warnings to stderr (e.g., "SSL verification disabled", "Using token
from env var"). When `$PSNativeCommandUseErrorActionPreference = $true` is inherited by
the `Invoke-AzCli` function in `frontend-helpers.ps1`, PowerShell treats these warnings
as errors.

**Impact**: Frontend CLI commands that return 204 No Content (dataset publish, consent set,
query vote) appear to fail because:
1. The CLI writes warnings to stderr
2. PowerShell throws `NativeCommandExitException`
3. The script exits even though the operation succeeded on the server

**Fix applied**: `Invoke-AzCli` now saves and restores `$PSNativeCommandUseErrorActionPreference`
in a `try/finally` block (matching `Invoke-AzCliSafe`). It also filters `ErrorRecord`
objects from the `2>&1` capture, passing only string output to the JSON parser.

### 2. ARM response `collaborationId` field is null

**Bug**: `az rest --method GET .../Collaborations/<name>` returns `properties.collaborationId: null`.

**Workaround**: Get the frontend UUID via `az managedcleanroom frontend collaboration list`.

### 3. `--consortium-type` and `--user-identity` not valid CLI parameters

**Bug**: Scripts that use `az managedcleanroom collaboration create` with `--consortium-type`
and `--user-identity` fail because these aren't valid parameters on the CLI command.

**Workaround**: Use `az rest` for collaboration creation (as documented in Phase 1.1).

### 4. `08-publish-dataset-cpk.ps1` double `ConvertFrom-Json` on `Invoke-AzCli` result

**Bug**: `Invoke-AzCli` (in `frontend-helpers.ps1`) already parses the JSON response and
returns a PSObject. Line 288 of `08-publish-dataset-cpk.ps1` was calling
`$skrPolicyRaw | ConvertFrom-Json` on the already-parsed result, causing a parse error
(you can't `ConvertFrom-Json` on a PSObject).

**Impact**: SKR policy fetch failed during KEK creation, blocking the entire CPK publish flow.

**Fix applied**: Removed the redundant `ConvertFrom-Json` call — the `Invoke-AzCli` result
is used directly as a PSObject.

### 5. `07-grant-access.ps1` missing `-setupKeyVault` switch for CPK mode

**Bug**: The `07-grant-access.ps1` script previously hardcoded `-setupKeyVault:$false` when
calling `setup-access.ps1`. For CPK mode, the cleanroom needs Key Vault access (Crypto
Officer + Secrets User roles on the managed identity) to unwrap KEKs and read wrapped DEKs.

**Impact**: CPK query execution failed because the managed identity lacked KV permissions.

**Fix applied**: Added `-setupKeyVault` as a passthrough switch parameter to `07-grant-access.ps1`.
When specified, the script passes `-setupKeyVault` to `setup-access.ps1`, which assigns
`Key Vault Crypto User` and `Key Vault Secrets User` RBAC roles.

### 6. `12-view-results.ps1` azcopy fails on HNS directory blobs

**Bug**: Storage accounts with Hierarchical Namespace (HNS) enabled create directory marker
blobs (zero-byte blobs with `hdi_isfolder=true`). When `azcopy copy "container/*" local
--recursive` runs, it tries to download these directory marker blobs as files, but the
local filesystem already has directories at those paths, causing a conflict error.

**Impact**: CPK output download failed with azcopy errors on HNS-enabled storage accounts.

**Fix applied**: Added `--include-pattern "*.csv;*.crc;*_SUCCESS*"` to the azcopy download
command, filtering to actual data files and skipping directory marker blobs.
