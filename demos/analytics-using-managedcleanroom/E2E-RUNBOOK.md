# E2E Runbook: Azure Managed CleanRoom Analytics

Complete step-by-step guide for running an analytics collaboration end-to-end.
Supports **SSE** and **CPK** encryption modes, with **CLI** or **REST API** operation modes.

---

## Table of Contents

- [Configuration](#configuration)
- [Step 01: Prerequisites & Authentication](#step-01-prerequisites--authentication) `[ALL]`
- [Step 02: Create Collaboration](#step-02-create-collaboration) `[OWNER]`
- [Step 03: Add Collaborators & Accept Invitations](#step-03-add-collaborators--accept-invitations) `[OWNER → WOODGROVE]`
- [Step 04: Resource Provisioning](#step-04-resource-provisioning) `[WOODGROVE]`
- [Step 05: OIDC Identity & Federated Credentials](#step-05-oidc-identity--federated-credentials) `[WOODGROVE]`
- [Step 06: Publish Datasets](#step-06-publish-datasets) `[WOODGROVE]`
- [Step 07: CPK Key Management](#step-07-cpk-key-management) `[WOODGROVE]` _(CPK only)_
- [Step 08: Publish Query](#step-08-publish-query) `[WOODGROVE]`
- [Step 09: Approve Query](#step-09-approve-query) `[WOODGROVE]`
- [Step 10: Execute Query](#step-10-execute-query) `[WOODGROVE]`
- [Step 11: Monitor Query](#step-11-monitor-query) `[ANY]`
- [Step 12: Results & Audit](#step-12-results--audit) `[WOODGROVE]`
- [Appendix A: Federated Credential Subject Reference](#appendix-a-federated-credential-subject-reference)
- [Appendix B: Troubleshooting](#appendix-b-troubleshooting)
- [Appendix C: CPK Deep Dive](#appendix-c-cpk-deep-dive)
- [Appendix D: CLI Command Reference](#appendix-d-cli-command-reference)
- [Appendix E: API Response Reference](#appendix-e-api-response-reference)
- [Appendix F: Known Bugs & Workarounds](#appendix-f-known-bugs--workarounds)

---

## Configuration

### Mode Flags

Set once at the start. All scripts respect these flags via parameters.

```powershell
# Operation mode: how frontend commands are executed
#   "cli"  → az managedcleanroom frontend CLI commands
#   "rest" → Direct REST API calls via Invoke-RestMethod in PowerShell scripts
$ApiMode = "cli"

# Encryption mode: how data is protected at rest
#   "SSE" → Server-Side Encryption (Azure-managed keys, simpler)
#   "CPK" → Customer-Provided Key (you control the encryption keys)
$EncryptionMode = "SSE"
```

| Mode | Frontend operations | ARM operations | When to use |
|---|---|---|---|
| `cli` | `az managedcleanroom frontend` commands | `az rest` | Default. Requires `managedcleanroom` CLI extension. |
| `rest` | `Invoke-RestMethod` via `frontend-helpers.ps1` | `az rest` | Fallback if CLI extension has bugs. |

> **NOTE**: ARM operations (create collaboration, enable workload, add collaborator) always
> use `az rest` regardless of `$ApiMode`. The mode flag only affects **frontend** operations
> (dataset publish, query run, consent, etc.).

### Collaboration Modes

| Mode | Terminals | Accounts | What happens |
|---|---|---|---|
| **Single-collaborator** | T1 (Owner) + T2 (Woodgrove) | 1 collaborator email | Woodgrove provisions resources, publishes input + output datasets, proposes query, votes, runs, downloads results. |
| **Multi-collaborator** | T1 (Owner) + T2 (Woodgrove) + T3 (Northwind) | 2 collaborator emails | Woodgrove and Northwind each provision own resources and publish own input datasets. Woodgrove also publishes the output dataset. A second query joins both datasets; both collaborators vote. |

**Single-collaborator** is the simpler path — only the Owner and Woodgrove are needed.
Woodgrove's query reads only its own input data and writes to its own output.

**Multi-collaborator** adds Northwind as a second data contributor. A second query can
join data from both collaborators. Both must vote to approve queries that use their data.
Woodgrove still owns the output dataset and downloads results.

### Terminal Setup

| Terminal | Persona | Role | Azure Login | Required? |
|---|---|---|---|---|
| **T1 — Owner** | Collaboration admin | Creates collab, adds collaborators | MSFT subscription | **Yes** |
| **T2 — Woodgrove** | Primary collaborator | Provisions resources, publishes input + output datasets, proposes & runs query | Personal subscription | **Yes** |
| **T3 — Northwind** | Additional collaborator | Provisions own resources, publishes own input dataset, votes | Personal subscription | **Optional** |

> **Single-collaborator mode**: Only T1 and T2 are needed. One email, one token, one
> collaborator terminal. No Northwind commands run at all. Steps marked
> `_multi-collaborator only_` are skipped entirely.
>
> **Multi-collaborator mode**: Northwind (T3) joins with their own Microsoft account.
> Steps 04-06 for Northwind run in T3 in parallel with Woodgrove's T2. Both vote on queries
> that reference both collaborators' datasets.

### Working Directory

All terminals must be in:
```
demos/analytics-using-managedcleanroom/
```

---

## Step 01: Prerequisites & Authentication `[ALL]`

> Install tools, generate MSAL tokens, and configure shared variables in each terminal.

### 1.1 Requirements

| Requirement | Details |
|---|---|
| Azure CLI | 2.75.0+ |
| `managedcleanroom` extension | `az extension add --name managedcleanroom` (v1.0.0b5+) |
| PowerShell | 7.x+ |
| MSAL.PS module | `Install-Module MSAL.PS -Scope CurrentUser -Force` |
| azcopy | v10+ (CPK mode only) |
| Python 3 + `cryptography` | CPK mode only — `pip install cryptography` |

### 1.2 Shared Variables (set in ALL terminals)

```powershell
# --- Mode flags ---
$ApiMode = "cli"           # "cli" or "rest"
$EncryptionMode = "SSE"    # "SSE" or "CPK"

# --- Endpoints ---
$frontend = "https://dogfood.workload-frontendwestus.cleanroom.cloudapp.azure-test.net"
$armEndpoint = "https://eastus2euap.management.azure.com"
$armApiVersion = "2025-10-31-preview"

# --- Subscriptions ---
$msftSubscription = "<msft-subscription-id>"        # Hosts the collaboration ARM resource
$personalSubscription = "<personal-subscription-id>" # Hosts storage, KV, MI resources
$personalTenantId = "<personal-tenant-id>"

# --- Collaboration ---
$collabName = "<collaboration-name>"
$collabRg = "<collaboration-resource-group>"         # In the MSFT subscription

# --- Resource groups (in personal subscription) ---
$woodgroveRg = "cr-e2e-woodgrove-rg"
$northwindRg = "cr-e2e-northwind-rg"    # Multi-collaborator only

# --- OIDC (whitelisted storage account) ---
$oidcStorageAccount = "cleanroomoidc"
```

### 1.3 Generate MSAL Tokens

#### Terminal T2 (Woodgrove)

```powershell
$token = Get-MsalToken -ClientId "8a3849c1-81c5-4d62-b83e-3bb2bb11251a" `
    -TenantId "common" -Scopes "User.Read" -DeviceCode
$token.IdToken | Out-File -FilePath "/tmp/msal-idtoken-woodgrove.txt" -NoNewline
$woodgroveTokenFile = "/tmp/msal-idtoken-woodgrove.txt"
```

Sign in with your Microsoft account when prompted.

#### Terminal T3 (Northwind) — _multi-collaborator only_

```powershell
$token = Get-MsalToken -ClientId "8a3849c1-81c5-4d62-b83e-3bb2bb11251a" `
    -TenantId "common" -Scopes "User.Read" -DeviceCode
$token.IdToken | Out-File -FilePath "/tmp/msal-idtoken-northwind.txt" -NoNewline
$northwindTokenFile = "/tmp/msal-idtoken-northwind.txt"
```

Sign in with the **Northwind** Microsoft account when prompted.

#### Extract JWT `oid`

> **CRITICAL**: You need the `oid` claim from each collaborator's JWT. This is used for
> federated credentials in Step 05. See [Appendix A](#appendix-a-federated-credential-subject-reference)
> for why this matters.

```powershell
# Run in T2 (and T3 for multi-collaborator)
$tokenFile = $woodgroveTokenFile    # or $northwindTokenFile in T3
$tokenB64 = (Get-Content $tokenFile -Raw).Split('.')[1]
$padLen = (4 - $tokenB64.Length % 4) % 4
$padded = $tokenB64 + ('=' * $padLen)
$claims = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($padded)) | ConvertFrom-Json
$myOid = $claims.oid
Write-Host "JWT oid: $myOid"
```

```powershell
# T2 (Woodgrove)
$woodgroveOid = "<oid-from-above>"

# T3 (Northwind) — multi-collaborator only
# $northwindOid = "<oid-from-northwind-token>"
```

> For MSA (personal Microsoft accounts), the OID typically starts with `00000000-0000-0000-`.
> This is **different** from `az ad signed-in-user show --query id` — always use the JWT `oid`.

### 1.4 Configure CLI Extension (CLI mode only)

```powershell
az managedcleanroom frontend configure `
    --endpoint $frontend
```

### 1.5 Token Lifetime

MSAL tokens last ~24 hours. If a token expires mid-flow, regenerate it (repeat 1.3).

---

## Step 02: Create Collaboration `[OWNER]`

> **Terminal: T1 (Owner)**
>
> Creates the collaboration ARM resource and enables the analytics workload.
> Requires access to the MSFT subscription.

### 2.1 Login & Create

```powershell
az login --tenant "<msft-tenant-id>"
az account set --subscription $msftSubscription

# Create collaboration
az rest --method PUT `
    --url "$armEndpoint/subscriptions/$msftSubscription/resourceGroups/$collabRg/providers/Private.CleanRoom/Collaborations/$collabName?api-version=$armApiVersion" `
    --resource "https://management.azure.com/" `
    --body "{
        `"location`": `"westus`",
        `"properties`": {
            `"consortiumType`": `"ConfidentialAKS`",
            `"userIdentity`": {
                `"tenantId`": `"<owner-tenant-id>`",
                `"objectId`": `"<owner-object-id>`",
                `"accountType`": `"MicrosoftAccount`"
            }
        }
    }"
```

**Expected**: 201 Created or 200 OK with `provisioningState: "Succeeded"`.
May return 202 Accepted — poll the `Location` header (2-5 minutes).

### 2.2 Enable Analytics Workload

```powershell
az rest --method POST `
    --url "$armEndpoint/subscriptions/$msftSubscription/resourceGroups/$collabRg/providers/Private.CleanRoom/Collaborations/$collabName/enableWorkload?api-version=$armApiVersion" `
    --resource "https://management.azure.com/" `
    --body '{"workloadType": "analytics"}'
```

> **CRITICAL**: Only pass `workloadType`. Do NOT pass `securityPolicyOption`.

**Expected**: 202 Accepted. Poll until complete (**5-15 minutes**).

### 2.3 Get Frontend UUID

> The ARM `properties.collaborationId` field is `null` (known bug).
> Retrieve the frontend UUID via the frontend API.

```powershell
# Get the frontend endpoint
$collabShow = az rest --method GET `
    --url "$armEndpoint/subscriptions/$msftSubscription/resourceGroups/$collabRg/providers/Private.CleanRoom/Collaborations/$collabName?api-version=$armApiVersion" `
    --resource "https://management.azure.com/" | ConvertFrom-Json

$frontendEndpoint = $collabShow.properties.workloads[0].endpoint

# Get the frontend UUID (need an MSAL token)
$env:MANAGEDCLEANROOM_ACCESS_TOKEN = Get-Content "/tmp/msal-idtoken-woodgrove.txt" -Raw
$env:AZURE_CLI_DISABLE_CONNECTION_VERIFICATION = "1"
$collabs = az managedcleanroom frontend collaboration list -o json | ConvertFrom-Json
$collabId = ($collabs | Where-Object { $_.name -eq $collabName }).id
Write-Host "Collaboration UUID: $collabId"
```

> **Share `$collabId` with all terminals.** Every `--collaboration-id` parameter takes this
> frontend UUID, NOT the ARM resource ID (slashes get URL-encoded → 404).

---

## Step 03: Add Collaborators & Accept Invitations `[OWNER → WOODGROVE]`

### 3.1 Add Collaborators — Terminal T1 (Owner)

```powershell
# Add Woodgrove (required)
az rest --method POST `
    --url "$armEndpoint/subscriptions/$msftSubscription/resourceGroups/$collabRg/providers/Private.CleanRoom/Collaborations/$collabName/addCollaborator?api-version=$armApiVersion" `
    --resource "https://management.azure.com/" `
    --body '{"email": "<woodgrove-email>"}'
```

**Multi-collaborator only** — add Northwind as a second collaborator:
```powershell
az rest --method POST `
    --url "$armEndpoint/subscriptions/$msftSubscription/resourceGroups/$collabRg/providers/Private.CleanRoom/Collaborations/$collabName/addCollaborator?api-version=$armApiVersion" `
    --resource "https://management.azure.com/" `
    --body '{"email": "<northwind-email>"}'
```

**Expected**: 202 Accepted for each.

### 3.2 Accept Invitation — Terminal T2 (Woodgrove)

```powershell
$env:MANAGEDCLEANROOM_ACCESS_TOKEN = Get-Content $woodgroveTokenFile -Raw
$env:AZURE_CLI_DISABLE_CONNECTION_VERIFICATION = "1"

az managedcleanroom frontend invitation accept `
    --collaboration-id $collabId
```

### 3.3 Accept Invitation — Terminal T3 (Northwind) — _multi-collaborator only_

```powershell
$env:MANAGEDCLEANROOM_ACCESS_TOKEN = Get-Content $northwindTokenFile -Raw
$env:AZURE_CLI_DISABLE_CONNECTION_VERIFICATION = "1"

az managedcleanroom frontend invitation accept `
    --collaboration-id $collabId
```

**Verify** (either terminal):
```powershell
az managedcleanroom frontend show `
    --collaboration-id $collabId -o json
```
Should show `"status": "Finalized"` once all invitations are accepted.

---

## Step 04: Resource Provisioning `[WOODGROVE]`

> **Terminal: T2 (Woodgrove)**
>
> Creates storage accounts, key vaults, managed identities, and uploads data.
>
> **Multi-collaborator**: Northwind also runs these commands in T3 with their own resource group.

### 4.1 Login

```powershell
az login --tenant $personalTenantId
az account set --subscription $personalSubscription
```

### 4.2 Prepare Resources

#### Terminal T2 (Woodgrove)

```powershell
./scripts/04-prepare-resources.ps1 -resourceGroup $woodgroveRg -persona woodgrove
```

#### Terminal T3 (Northwind) — _multi-collaborator only_

```powershell
./scripts/04-prepare-resources.ps1 -resourceGroup $northwindRg -persona northwind
```

**Verify**: `generated/<rg>/names.generated.ps1` and `resources.generated.json` exist.

### 4.3 Upload Data

> Data is **downloaded at runtime** from the `Azure-Samples/Synapse` GitHub repo (Twitter CSV).
> The `-dataDir` is created automatically.

#### SSE Mode

##### Terminal T2 (Woodgrove)

```powershell
./scripts/05-prepare-data-sse.ps1 -resourceGroup $woodgroveRg -persona woodgrove `
    -dataDir "./generated/datasource/woodgrove"
```

##### Terminal T3 (Northwind) — _multi-collaborator only_

```powershell
./scripts/05-prepare-data-sse.ps1 -resourceGroup $northwindRg -persona northwind `
    -dataDir "./generated/datasource/northwind"
```

#### CPK Mode

##### Terminal T2 (Woodgrove)

```powershell
./scripts/05-prepare-data-cpk.ps1 -resourceGroup $woodgroveRg -persona woodgrove `
    -dataDir "./generated/datasource/woodgrove/input/csv" `
    -datasetSuffix "-cpk"
```

##### Terminal T3 (Northwind) — _multi-collaborator only_

```powershell
./scripts/05-prepare-data-cpk.ps1 -resourceGroup $northwindRg -persona northwind `
    -dataDir "./generated/datasource/northwind/input/csv" `
    -datasetSuffix "-cpk"
```

> CPK mode generates a 32-byte DEK per dataset, uploads data via `azcopy --cpk-by-value`,
> and saves DEK files to `generated/datastores/keys/`.
> See [Appendix C: CPK Deep Dive](#appendix-c-cpk-deep-dive) for details.

**Verify**: `generated/datastores/<persona>-datastore-metadata.json` exists.

---

## Step 05: OIDC Identity & Federated Credentials `[WOODGROVE]`

> **Terminal: T2 (Woodgrove)**
>
> Sets up OIDC issuer documents and creates federated credentials so the Spark workload
> can authenticate as each collaborator's managed identity at runtime.
>
> **Multi-collaborator**: Northwind also runs these commands in T3.

### 5.1 Setup OIDC Issuer

#### Terminal T2 (Woodgrove)

```powershell
./scripts/06-setup-identity.ps1 -resourceGroup $woodgroveRg -persona woodgrove `
    -collaborationId $collabId -frontendEndpoint $frontend `
    -OidcStorageAccount $oidcStorageAccount `
    -ApiMode $ApiMode
```

#### Terminal T3 (Northwind) — _multi-collaborator only_

```powershell
./scripts/06-setup-identity.ps1 -resourceGroup $northwindRg -persona northwind `
    -collaborationId $collabId -frontendEndpoint $frontend `
    -OidcStorageAccount $oidcStorageAccount `
    -ApiMode $ApiMode
```

> **Cross-Tenant Issue**: If `cleanroomoidc` is in a different tenant, `az storage blob upload`
> fails with "Issuer validation failed". Open a separate terminal, `az login --tenant <msft-tenant-id>`,
> and run the OIDC upload from there.

**Verify**:
```powershell
Get-Content "generated/$woodgroveRg/issuer-url.txt"
# Should show: https://cleanroomoidc.z22.web.core.windows.net/<collab-uuid>
```

### 5.2 Grant Access & Create Federated Credentials

> **CRITICAL**: `-userId` must be the **JWT `oid`** from Step 01.3 — NOT a persona name.
> See [Appendix A](#appendix-a-federated-credential-subject-reference) for details.

#### Terminal T2 (Woodgrove)

```powershell
$setupKV = if ($EncryptionMode -eq "CPK") { "-setupKeyVault" } else { "" }

./scripts/07-grant-access.ps1 -resourceGroup $woodgroveRg `
    -collaborationId $collabId -contractId "Analytics" `
    -userId $woodgroveOid $setupKV
```

#### Terminal T3 (Northwind) — _multi-collaborator only_

```powershell
$setupKV = if ($EncryptionMode -eq "CPK") { "-setupKeyVault" } else { "" }

./scripts/07-grant-access.ps1 -resourceGroup $northwindRg `
    -collaborationId $collabId -contractId "Analytics" `
    -userId $northwindOid $setupKV
```

> **CRITICAL**: `contractId` must be `"Analytics"` (capital A). The federated credential
> subject is `Analytics-{oid}`. Lowercase causes silent token exchange failures at runtime.
>
> **CPK**: Pass `-setupKeyVault` to grant the managed identity `Key Vault Crypto User` and
> `Key Vault Secrets User` roles needed for KEK release and DEK unwrapping.

**Wait**: RBAC propagation takes 60-120 seconds. The script waits and retries.

**Verify**:
```powershell
. "generated/$woodgroveRg/names.generated.ps1"
az identity federated-credential list `
    --identity-name $MANAGED_IDENTITY_NAME `
    --resource-group $woodgroveRg -o table

# Subject should be: Analytics-<jwt-oid>
# NOT: Analytics-woodgrove or Analytics-<graph-api-oid>
```

---

## Step 06: Publish Datasets `[WOODGROVE]`

> **Terminal: T2 (Woodgrove)**
>
> Woodgrove publishes:
> - 1 input dataset (read) — `woodgrove-input-csv`
> - 1 output dataset (write) — `woodgrove-output-csv`
>
> **Multi-collaborator**: Northwind also publishes their input dataset in T3.

### SSE Mode

#### Terminal T2 (Woodgrove)

```powershell
./scripts/08-publish-dataset-sse.ps1 -collaborationId $collabId `
    -resourceGroup $woodgroveRg -persona woodgrove `
    -frontendEndpoint $frontend `
    -TokenFile $woodgroveTokenFile `
    -ApiMode $ApiMode
```

#### Terminal T3 (Northwind) — _multi-collaborator only_

```powershell
./scripts/08-publish-dataset-sse.ps1 -collaborationId $collabId `
    -resourceGroup $northwindRg -persona northwind `
    -frontendEndpoint $frontend `
    -TokenFile $northwindTokenFile `
    -ApiMode $ApiMode
```

### CPK Mode

> CPK publish is a longer step: publishes dataset metadata, then creates per-dataset KEKs,
> wraps DEKs, stores secrets, and enables consent. See [Appendix C](#appendix-c-cpk-deep-dive).

#### Terminal T2 (Woodgrove)

```powershell
./scripts/08-publish-dataset-cpk.ps1 -collaborationId $collabId `
    -resourceGroup $woodgroveRg -persona woodgrove `
    -frontendEndpoint $frontend `
    -TokenFile $woodgroveTokenFile `
    -ApiMode $ApiMode
```

#### Terminal T3 (Northwind) — _multi-collaborator only_

```powershell
./scripts/08-publish-dataset-cpk.ps1 -collaborationId $collabId `
    -resourceGroup $northwindRg -persona northwind `
    -frontendEndpoint $frontend `
    -TokenFile $northwindTokenFile `
    -ApiMode $ApiMode
```

> **CRITICAL**: The `issuerUrl` in the dataset body must be the **public OIDC URL**
> (e.g., `https://cleanroomoidc.z22.web.core.windows.net/<uuid>`), never `"https://cgs/oidc"`.
> The scripts read from `generated/<rg>/issuer-url.txt` automatically.

**Expected**: 204 No Content (the CLI may print warnings — this is normal).

**Verify**:
```powershell
$env:MANAGEDCLEANROOM_ACCESS_TOKEN = Get-Content $woodgroveTokenFile -Raw
$env:AZURE_CLI_DISABLE_CONNECTION_VERIFICATION = "1"
az managedcleanroom frontend analytics dataset show `
    --collaboration-id $collabId `
    --document-id "woodgrove-input-csv" -o json
```
Should show `"state": "Accepted"`.

---

## Step 07: CPK Key Management `[WOODGROVE]` _(CPK only)_

> **Skip this step for SSE mode.**
>
> For CPK, verify that the key management performed in Step 06 completed correctly.
> The `08-publish-dataset-cpk.ps1` script handles KEK creation, DEK wrapping, and
> consent automatically. This step is for **validation**.

### 7.1 Verify KEK Properties

```powershell
. "generated/$woodgroveRg/names.generated.ps1"

az keyvault key show --vault-name $KEYVAULT_NAME --name "<dataset-name>-kek" `
    --query '{exportable:attributes.exportable, keyType:key.kty, ops:key.keyOps}' -o json
```

**Expected**: `exportable: true`, `keyType: "RSA-HSM"`, `ops: ["encrypt", "wrapKey"]`.

### 7.2 Verify Wrapped DEK Secret

```powershell
az keyvault secret show --vault-name $KEYVAULT_NAME `
    --name "wrapped-<dataset-name>-dek-<kek-name>" `
    --query '{name:name, id:id}' -o json
```

**Expected**: Secret exists with a base64-encoded value.

### 7.3 Verify Dataset SKR Policy

```powershell
$env:MANAGEDCLEANROOM_ACCESS_TOKEN = Get-Content $woodgroveTokenFile -Raw
$env:AZURE_CLI_DISABLE_CONNECTION_VERIFICATION = "1"
az managedcleanroom frontend analytics dataset show `
    --collaboration-id $collabId `
    --document-id "<dataset-name>" -o json
```

Check that the response includes `kek.kid` (Key Vault key URL), `kek.maaUrl` (MAA endpoint),
and `dek.secretId` (Key Vault secret URL).

> If any verification fails, see [Appendix C: CPK Troubleshooting](#cpk-troubleshooting).

---

## Step 08: Publish Query `[WOODGROVE]`

> **Terminal: T2 (Woodgrove)**
>
> Woodgrove proposes the SQL query.

### Single-collaborator query (Woodgrove data only)

The query reads only Woodgrove's input dataset and writes to Woodgrove's output:

```powershell
./scripts/09-publish-query.ps1 -collaborationId $collabId `
    -queryName "query1" `
    -queryDir "../demos/query/woodgrove/query1" `
    -publisherInputDataset "woodgrove-input-csv" `
    -consumerInputDataset "woodgrove-input-csv" `
    -outputDataset "woodgrove-output-csv" `
    -frontendEndpoint $frontend `
    -TokenFile $woodgroveTokenFile `
    -ApiMode $ApiMode
```

### Multi-collaborator query (both datasets) — _multi-collaborator only_

A second query joins Northwind's and Woodgrove's input datasets:

```powershell
./scripts/09-publish-query.ps1 -collaborationId $collabId `
    -queryName "query2" `
    -queryDir "../demos/query/woodgrove/query1" `
    -publisherInputDataset "northwind-input-csv" `
    -consumerInputDataset "woodgrove-input-csv" `
    -outputDataset "woodgrove-output-csv" `
    -frontendEndpoint $frontend `
    -TokenFile $woodgroveTokenFile `
    -ApiMode $ApiMode
```

> **CPK**: Use CPK dataset names (e.g., `northwind-input-csv-cpk`, `woodgrove-output-csv-cpk`).

The query has 3 segments (in `demos/query/woodgrove/query1/`):
- **Segment 1** (seq=1): `CREATE OR REPLACE TEMP VIEW publisher_view AS SELECT * FROM publisher_data`
- **Segment 2** (seq=1): `CREATE OR REPLACE TEMP VIEW consumer_view AS SELECT * FROM consumer_data`
- **Segment 3** (seq=2): `SELECT author, COUNT(*) ... FROM (... UNION ALL ...) WHERE mentions LIKE '%MikeDoesBigData%' GROUP BY author`

Same-sequence segments run in parallel; higher sequences wait for lower ones.

**Expected**: 204 No Content.

**Verify**:
```powershell
$env:MANAGEDCLEANROOM_ACCESS_TOKEN = Get-Content $woodgroveTokenFile -Raw
$env:AZURE_CLI_DISABLE_CONNECTION_VERIFICATION = "1"
az managedcleanroom frontend analytics query show `
    --collaboration-id $collabId --document-id "query1" -o json
```
Should show `"state": "Proposed"`.

---

## Step 09: Approve Query `[WOODGROVE]`

> **Terminal: T2 (Woodgrove)**
>
> **Single-collaborator**: Only Woodgrove votes (one vote moves the query to `Accepted`).
>
> **Multi-collaborator**: Both collaborators must vote on queries that reference their datasets.

### 9.1 Vote — Terminal T2 (Woodgrove)

```powershell
./scripts/10-vote-query.ps1 -collaborationId $collabId -queryName "query1" `
    -frontendEndpoint $frontend `
    -TokenFile $woodgroveTokenFile `
    -ApiMode $ApiMode
```

### 9.2 Vote — Terminal T3 (Northwind) — _multi-collaborator only_

> Required for `query2` (which references Northwind's dataset). Not needed for `query1`
> (Woodgrove-only data).

```powershell
./scripts/10-vote-query.ps1 -collaborationId $collabId -queryName "query2" `
    -frontendEndpoint $frontend `
    -TokenFile $northwindTokenFile `
    -ApiMode $ApiMode
```

Woodgrove must also vote on `query2`:
```powershell
./scripts/10-vote-query.ps1 -collaborationId $collabId -queryName "query2" `
    -frontendEndpoint $frontend `
    -TokenFile $woodgroveTokenFile `
    -ApiMode $ApiMode
```

**Expected**: 204 No Content for each vote.

> **Consent ordering** (multi-collaborator only): Consent can only be enabled on an
> `Accepted` query. The query stays `Proposed` until all votes are in. The `10-vote-query.ps1`
> script enables consent automatically after voting. If the first voter's consent fails
> (query still `Proposed`), they must re-enable after the last voter's vote.
>
> **Workaround**: Run the consent enable for **all** collaborators **after** all votes are in.

**Verify**: Query shows `"state": "Accepted"` with `proposalId` populated.

---

## Step 10: Execute Query `[WOODGROVE]`

> **Terminal: T2 (Woodgrove)**
>
> Submits the Spark SQL query for execution in the confidential AKS cluster.

```powershell
./scripts/11-run-query.ps1 -collaborationId $collabId -queryName "query1" `
    -frontendEndpoint $frontend `
    -TokenFile $woodgroveTokenFile `
    -ApiMode $ApiMode
```

**Expected**: HTTP 200 with `"status": "success"` and a job ID:
```json
{"status": "success", "id": "cl-spark-<uuid>"}
```

Save the job ID:
```powershell
$jobId = "cl-spark-<uuid>"
```

> `"status": "success"` means the run was **accepted for scheduling**, not completed.
> The Spark job takes **10-20 minutes**.

---

## Step 11: Monitor Query `[ANY]`

> **Terminal: T2 (Woodgrove)** or any collaborator terminal.
>
> Poll the run status until completion.

### 11.1 Real-Time Status

```powershell
./scripts/13-run-status.ps1 -collaborationId $collabId `
    -jobId $jobId `
    -frontendEndpoint $frontend `
    -TokenFile $woodgroveTokenFile `
    -ApiMode $ApiMode `
    -poll -pollIntervalSeconds 30
```

**State transitions** (typical timeline):

| Time | State | Key Events |
|---|---|---|
| +0 min | `SUBMITTED` | `SparkApplicationSubmitted` |
| +5-8 min | `RUNNING` | `SparkDriverRunning` |
| +8-12 min | `RUNNING` | `DATASET_LOAD_STARTED/COMPLETED` |
| +10-15 min | `RUNNING` | `QUERY_SEGMENT_EXECUTION_*` |
| +15-20 min | `COMPLETED` | `QUERY_EXECUTION_COMPLETED`, `SparkDriverCompleted` |

> **`PENDING_RERUN`** is normal — the Spark operator may restart during initial setup.
> It will transition back to `SUBMITTED` → `RUNNING` automatically.

### 11.2 Audit Events (Real-Time)

```powershell
./scripts/15-audit-events.ps1 -collaborationId $collabId `
    -frontendEndpoint $frontend `
    -TokenFile $woodgroveTokenFile `
    -ApiMode $ApiMode
```

---

## Step 12: Results & Audit `[WOODGROVE]`

> **Terminal: T2 (Woodgrove)** — Woodgrove owns the output dataset.

### 12.1 Run History

```powershell
./scripts/14-run-history.ps1 -collaborationId $collabId `
    -queryName "query1" `
    -frontendEndpoint $frontend `
    -TokenFile $woodgroveTokenFile `
    -ApiMode $ApiMode
```

**Expected**:
```json
{
  "queryId": "query1",
  "latestRun": {
    "isSuccessful": true,
    "stats": { "rowsRead": 13872, "rowsWritten": 697 },
    "durationSeconds": 1039
  },
  "summary": { "totalRuns": 1, "successfulRuns": 1 }
}
```

> Returns 404 if no runs have reached terminal state — this is normal during execution.
>
> **Quirk**: `--document-id` may need the `contractId` value (`Analytics`) instead of
> the custom query name. If `runhistory list` returns empty for a completed run, try
> `--document-id Analytics`.

### 12.2 Download Output (SSE)

```powershell
az storage blob list --account-name "<woodgrove-sa>" `
    --container-name woodgrove-output `
    --prefix "Analytics/" --auth-mode login -o table

az storage blob download --account-name "<woodgrove-sa>" `
    --container-name woodgrove-output `
    --name "Analytics/<date>/<runId>/part-00000-<uuid>.csv" `
    --file ./output.csv --auth-mode login

Get-Content ./output.csv
```

Output path pattern: `Analytics/{date}/{runId}/part-00000-{uuid}.csv`

### 12.3 Download Output (CPK)

CPK output is encrypted — use `azcopy` with the output DEK:

```powershell
./scripts/12-view-results.ps1 -collaborationId $collabId `
    -queryName "query1" `
    -frontendEndpoint $frontend `
    -TokenFile $woodgroveTokenFile `
    -ApiMode $ApiMode `
    -DownloadCpkOutput `
    -OutputDekFile "generated/datastores/keys/woodgrove-output-csv-cpk-dek.bin" `
    -OutputStorageAccount "<woodgrove-sa-name>"
```

> The `--include-pattern "*.csv;*.crc;*_SUCCESS*"` filter is required for HNS-enabled
> storage accounts. See [Appendix F, Bug 6](#appendix-f-known-bugs--workarounds).

---

## Appendix A: Federated Credential Subject Reference

This is the **most common source of query failures**. Getting the subject wrong causes
a silent token exchange failure — the Spark driver exits with code 1, 0 rows read.

### Subject Format

```
{contractId}-{ownerId}
```

- `contractId` = `"Analytics"` (capital A, case-sensitive)
- `ownerId` = the collaborator's `oid` claim from their MSAL IdToken (JWT)

### How the Subject is Constructed (Server-Side)

From `QueriesController.cs` line 355:
```csharp
string subject = string.Join("-", inputJob.ContractId, dataset.OwnerId);
```

When a dataset is published, the frontend extracts the `oid` claim from the caller's JWT
and stores it as `proposerId`/`ownerId` in the CCF ledger.

### MSA vs AAD Object IDs

For MSA (personal Microsoft) accounts, the JWT `oid` and Graph API object ID are **different**:

| Source | Command | Use? |
|---|---|---|
| JWT `oid` (correct) | PowerShell decode from Step 01.3 | **YES** |
| Graph API (wrong) | `az ad signed-in-user show --query id` | **NO** |
| Dataset `proposerId` | `dataset show` response | **YES** (to verify) |

### Verification

After publishing a dataset, confirm the stored owner matches your JWT `oid`:

```powershell
$env:MANAGEDCLEANROOM_ACCESS_TOKEN = Get-Content $woodgroveTokenFile -Raw
$env:AZURE_CLI_DISABLE_CONNECTION_VERIFICATION = "1"
az managedcleanroom frontend analytics dataset show `
    --collaboration-id $collabId `
    --document-id "woodgrove-input-csv" -o json
# Check the proposerId field — should match $woodgroveOid
```

The federated credential subject should be `Analytics-<that-proposerId>`.

### Fixing Wrong Subjects

```powershell
# Delete wrong credential
az identity federated-credential delete `
    --name "Analytics-woodgrove-federation" `
    --identity-name "<mi-name>" --resource-group $woodgroveRg --yes

# Create correct credential
az identity federated-credential create `
    --name "Analytics-$woodgroveOid-federation" `
    --identity-name "<mi-name>" --resource-group $woodgroveRg `
    --issuer "$(Get-Content generated/$woodgroveRg/issuer-url.txt)" `
    --subject "Analytics-$woodgroveOid" `
    --audiences "api://AzureADTokenExchange"
```

Propagation takes **1-5 minutes**.

---

## Appendix B: Troubleshooting

### SPARK_JOB_FAILED: driver container failed with ExitCode: 1

**Cause**: Federated credential subject mismatch (most common). See [Appendix A](#appendix-a-federated-credential-subject-reference).

---

### AADSTS700211: No matching federated identity record

**Cause**: `issuerUrl` in the published dataset is wrong (commonly `"https://cgs/oidc"`).

**Fix**: Republish with the correct issuer URL from `generated/<rg>/issuer-url.txt`.

---

### SSL certificate verify failed / hostname mismatch

**Cause**: Dogfood cert CN=`ccr-proxy` doesn't match the hostname.

**Fix**: Always set `$env:AZURE_CLI_DISABLE_CONNECTION_VERIFICATION = "1"`.

---

### 404 Not Found on frontend commands

**Cause**: Using the ARM resource ID instead of the frontend UUID for `--collaboration-id`.

**Fix**: Use the frontend UUID from `az managedcleanroom frontend collaboration list`.

---

### "Issuer validation failed" on OIDC upload

**Cause**: Cross-tenant auth. Your `az login` is in a different tenant than the OIDC SA.

**Fix**: Login to the OIDC SA's tenant in a separate terminal.

---

### ContractNotFound on all analytics operations

**Cause**: Stale CCF/ACI endpoint mapping. Collaboration-specific.

**Fix**: Create a new collaboration.

---

### InvalidUserIdentifier

**Cause**: Using ARM access token from an MSA guest account (lacks `preferred_username`).

**Fix**: Use MSAL IdTokens (Step 01.3).

---

### Python 3.13 `'tuple' object has no attribute 'token'`

**Cause**: CLI extension bug on Python 3.13.

**Fix**: Upgrade to `managedcleanroom` v1.0.0b5+ or use `-ApiMode "rest"`.

---

### "Already voted" / "Conflict" on query vote

**Cause**: Idempotent — vote was already submitted.

**Action**: Safe to ignore. Verify query state is `"Accepted"`.

---

### Run submitted but not in runhistory

**Cause**: `runhistory list` only shows terminal runs (COMPLETED/FAILED).

**Fix**: Use `runresult show --job-id <id>` for live status.

---

### PENDING_RERUN state

**Cause**: Normal Spark scheduling. The operator restarts during setup.

**Action**: Keep polling. Transitions to `SUBMITTED` → `RUNNING` → `COMPLETED`.

---

## Appendix C: CPK Deep Dive

### CPK vs SSE

| Aspect | SSE | CPK |
|---|---|---|
| Encryption | Azure-managed keys | Customer-provided keys per dataset |
| Key Vault | Not required | Required (Premium SKU with HSM) |
| Upload tool | `az storage blob upload-batch` | `azcopy copy --cpk-by-value` |
| Output download | `az storage blob download` | `azcopy copy --cpk-by-value` (need DEK) |
| Key management | None | Per-dataset DEK + KEK with SKR policy |

### Concepts

- **DEK (Data Encryption Key)**: 32-byte random key. Passed via CPK HTTP headers.
  Azure Storage encrypts/decrypts server-side.
- **KEK (Key Encryption Key)**: RSA-2048 in Azure Key Vault (HSM-backed, exportable).
  Has an SKR release policy tied to the collaboration's attestation hash.
- **Wrapped DEK**: DEK encrypted with KEK's public key (RSA-OAEP-SHA256). Stored as KV secret.
- **SKR**: Releases KEK private key only to code in a verified confidential compute environment.

### CRITICAL: CPK is Server-Side Encryption, NOT Client-Side

Do NOT manually encrypt files before uploading. Upload **plaintext** files using
`azcopy copy --cpk-by-value`. Azure Storage encrypts server-side with the provided DEK.

### Architecture

```
Upload:   plaintext CSV → azcopy --cpk-by-value → Azure Storage (encrypted with DEK)
Keys:     DEK → RSA-OAEP wrap with KEK → KV Secret (wrapped DEK)
          KEK (RSA-2048) → az keyvault key import (with SKR policy) → KV Key
Runtime:  SKR release → KEK private → unwrap DEK → CPK header → Storage → plaintext
```

### CPK Key Management Details

**KEK creation** (done by `08-publish-dataset-cpk.ps1`):
- Generated locally as RSA-2048 using Python `cryptography` lib
- Imported via `az keyvault key import` (NOT `key create`)
- `--exportable true --protection hsm --immutable false`
- SKR release policy attached (fetched from published dataset's `skr-policy` endpoint)

**DEK wrapping** (done by `08-publish-dataset-cpk.ps1`):
- Client-side RSA-OAEP with SHA-256 (both MGF1 and hash)
- NOT via `az keyvault key encrypt` (server-side wrapping doesn't work for CPK)

### CPK Troubleshooting

**DATASET_LOAD_FAILED / java.io.IOException**:
Data was manually encrypted before upload. Re-upload with `azcopy --cpk-by-value`.

**409 BlobUsesCustomerSpecifiedEncryption**:
Downloading CPK blobs without CPK headers. Use `azcopy --cpk-by-value` with DEK env vars.

**KEK SKR release fails**:
Check: `--exportable true`, `--protection hsm`, SKR policy matches collaboration attestation.
```powershell
az keyvault key show --vault-name "<kv>" --name "<kek>" `
    --query '{exportable:attributes.exportable, keyType:key.kty}' -o json
```

**Wrong wrapped DEK**:
SKR succeeds but data read fails. Re-generate DEK, re-upload, re-wrap, update KV secret.

---

## Appendix D: CLI Command Reference

All `az managedcleanroom frontend` commands require:
```powershell
$env:MANAGEDCLEANROOM_ACCESS_TOKEN = Get-Content "/tmp/msal-idtoken.txt" -Raw
$env:AZURE_CLI_DISABLE_CONNECTION_VERIFICATION = "1"
```

| Command | Description |
|---|---|
| `frontend configure --endpoint <url>` | Set frontend endpoint |
| `frontend collaboration list` | List collaborations |
| `frontend show -c <id>` | Show collaboration details |
| `frontend invitation accept -c <id>` | Accept invitation |
| `frontend analytics dataset show -c <id> -d <name>` | Show dataset |
| `frontend analytics dataset publish -c <id> -d <name> ...` | Publish dataset |
| `frontend analytics query show -c <id> -d <name>` | Show query |
| `frontend analytics query publish -c <id> -d <name> ...` | Publish query |
| `frontend analytics query vote -c <id> -d <name> --vote-action accept --proposal-id <pid>` | Vote on query |
| `frontend analytics query run -c <id> -d <name>` | Run query |
| `frontend analytics query runresult show -c <id> --job-id <jid>` | Live run status |
| `frontend analytics query runhistory list -c <id> -d <name>` | Terminal run history |
| `frontend consent set -c <id> -d <name> --consent-action enable` | Enable consent |
| `frontend analytics auditevent list -c <id>` | List audit events |

---

## Appendix E: API Response Reference

### Dataset Publish
- `POST /collaborations/{id}/analytics/datasets/{docId}/publish` → 204 No Content

### Dataset Show
- `GET /collaborations/{id}/analytics/datasets/{docId}` → JSON with `name`, `state`, `version`, schema, access policy

### Query Run
- `POST /collaborations/{id}/analytics/queries/{docId}/run` → `{"status":"success","id":"cl-spark-<uuid>"}`

### Run Result (live)
- `GET /collaborations/{id}/analytics/runs/{jobId}` → `{"status":{"applicationState":{"state":"COMPLETED|RUNNING|..."}}}`

### Run History (terminal)
- `GET /collaborations/{id}/analytics/queries/{docId}/runs` → `{"queryId","latestRun","runs[]","summary"}`
- 404 if no terminal runs exist yet

### Audit Events
- `GET /collaborations/{id}/analytics/auditevents` → `{"value":[{"scope","id","timestamp","data"}]}`

### Consent
- `PUT /collaborations/{id}/consent/{docId}` + `{"consentAction":"enable"}` → 204

### Vote
- `POST /collaborations/{id}/analytics/queries/{docId}/vote` + `{"voteAction":"accept","proposalId":"..."}` → 204

---

## Appendix F: Known Bugs & Workarounds

### 1. ~~`Invoke-AzCli` stderr throws~~ (FIXED)

`Invoke-AzCli` in `frontend-helpers.ps1` now saves/restores `$PSNativeCommandUseErrorActionPreference`
in a `try/finally` block and filters `ErrorRecord` objects from the `2>&1` capture.

### 2. ARM `collaborationId` field is null

`az rest GET .../Collaborations/<name>` returns `properties.collaborationId: null`.
**Workaround**: Use `az managedcleanroom frontend collaboration list` (Step 02.3).

### 3. `--consortium-type` and `--user-identity` not valid CLI parameters

The `az managedcleanroom collaboration create` command doesn't accept these parameters.
**Workaround**: Use `az rest` for collaboration creation (Step 02.1).

### 4. ~~`08-publish-dataset-cpk.ps1` double `ConvertFrom-Json`~~ (FIXED)

Removed the redundant `ConvertFrom-Json` call on the already-parsed `Invoke-AzCli` result.

### 5. ~~`07-grant-access.ps1` missing `-setupKeyVault`~~ (FIXED)

Added `-setupKeyVault` as a passthrough switch parameter.

### 6. ~~`12-view-results.ps1` azcopy fails on HNS blobs~~ (FIXED)

Added `--include-pattern "*.csv;*.crc;*_SUCCESS*"` to skip HNS directory marker blobs.
