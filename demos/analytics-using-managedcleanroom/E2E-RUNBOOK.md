# E2E Runbook: Azure Managed CleanRoom Analytics

Complete step-by-step guide for running an analytics collaboration end-to-end.
Supports **SSE** and **CPK** encryption modes, with **CLI** or **REST API** operation modes.

---

## Table of Contents

- [Configuration](#configuration)
- [Step 01: Prerequisites & Authentication](#step-01-prerequisites--authentication) `[ALL]`
- [Step 02: Create Collaboration](#step-02-create-collaboration) `[OWNER]`
- [Step 03: Add Collaborators & Accept Invitations](#step-03-add-collaborators--accept-invitations) `[OWNER → EACH COLLABORATOR]`
- [Step 04: Resource Provisioning](#step-04-resource-provisioning) `[EACH COLLABORATOR]`
- [Step 05: OIDC Identity & Federated Credentials](#step-05-oidc-identity--federated-credentials) `[EACH COLLABORATOR]`
- [Step 06: Publish Datasets](#step-06-publish-datasets) `[EACH COLLABORATOR]`
- [Step 07: CPK Key Management](#step-07-cpk-key-management) `[EACH COLLABORATOR]` _(CPK only)_
- [Step 08: Publish Query](#step-08-publish-query) `[WOODGROVE]`
- [Step 09: Approve Query](#step-09-approve-query) `[EACH COLLABORATOR]`
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
$ApiMode = "cli"           # "cli" (az managedcleanroom frontend) or "rest" (Invoke-RestMethod)
$EncryptionMode = "SSE"    # "SSE" (Azure-managed keys) or "CPK" (customer-provided keys)
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
| **Multi-collaborator** | T1 (Owner) + T2 (Woodgrove) + T3 (Northwind) | 2 collaborator emails | Each collaborator provisions own resources and publishes own input datasets. Woodgrove also publishes the output dataset and proposes queries. Both vote on queries that use their data. |

**Single-collaborator** is the simpler path — only the Owner and Woodgrove are needed.

**Multi-collaborator** adds Northwind as a second data contributor. A second query can
join data from both collaborators. Both must vote to approve it.

### Terminal Setup

| Terminal | Persona | Role | Required? |
|---|---|---|---|
| **T1 — Owner** | Collaboration admin | Creates collab, adds collaborators | **Yes** |
| **T2 — Woodgrove** | Primary collaborator | Provisions resources, publishes datasets (input + output), proposes & runs query | **Yes** |
| **T3 — Northwind** | Additional collaborator | Provisions own resources, publishes own input dataset, votes | **Optional** |

> **Single-collaborator mode**: Only T1 and T2 are needed. Steps marked
> `_multi-collaborator only_` are skipped entirely.
>
> **Multi-collaborator mode**: Northwind (T3) runs the same commands as Steps 04-06
> in parallel with Woodgrove. Both vote on queries that reference both datasets.

### Working Directory

All terminals must be in:
```
demos/analytics-using-managedcleanroom/
```

---

## Step 01: Prerequisites & Authentication `[ALL]`

### 1.1 Requirements

| Requirement | Details |
|---|---|
| Azure CLI | 2.75.0+ |
| `managedcleanroom` extension | `az extension add --name managedcleanroom` (v1.0.0b5+) |
| PowerShell | 7.x+ |
| Python 3 | 3.10+ — `python --version` (Windows) or `python3 --version` (Linux/macOS) |
| MSAL.PS module | `Install-Module MSAL.PS -Scope CurrentUser -Force` |
| azcopy | v10+ (CPK mode only) — Windows: `winget install Microsoft.AzCopy` / Linux: `curl -sL https://aka.ms/downloadazcopy-v10-linux \| tar xz --strip-components=1 -C /usr/local/bin` |
| `cryptography` package | CPK mode only — `pip install cryptography` |

### 1.2 Azure Subscription Permissions

Each collaborator needs these RBAC roles on their **personal subscription**:

| Role | Scope | Why | Assigned by |
|---|---|---|---|
| `Contributor` | Subscription or RG | Create resource groups, storage accounts, key vaults, managed identities | Pre-existing (you) |
| `User Access Administrator` | Subscription or RG | Assign RBAC roles on resources to managed identity | Pre-existing (you) |
| `Storage Blob Data Contributor` | Storage account | Upload data, manage containers | Script `04-prepare-resources.ps1` |
| `Key Vault Crypto Officer` | Key vault | Import KEKs (CPK mode) | Script `04-prepare-resources.ps1` |
| `Key Vault Secrets Officer` | Key vault | Store wrapped DEKs (CPK mode) | Script `04-prepare-resources.ps1` |
| `Storage Blob Data Owner` | Storage account (on MI) | Cleanroom accesses storage at runtime | Script `07-grant-access.ps1` |
| `Key Vault Crypto User` | Key vault (on MI, CPK only) | Cleanroom unwraps KEK via SKR | Script `07-grant-access.ps1` |
| `Key Vault Secrets User` | Key vault (on MI, CPK only) | Cleanroom reads wrapped DEK | Script `07-grant-access.ps1` |

The **Owner (T1)** needs `Contributor` on `$collabRg` in their subscription.

> **Verify** (run after setting variables in Step 1.3/1.4):
> ```powershell
> az role assignment list --assignee $(az ad signed-in-user show --query id -o tsv) `
>     --scope "/subscriptions/$subscription" `
>     --query "[].roleDefinitionName" -o tsv
> # Should include: Contributor AND User Access Administrator (or Owner which includes both)
> ```

> **Minimum viable permissions**: Scope roles to just the resource group instead of subscription:
> ```powershell
> az role assignment create --role "Contributor" --assignee "<your-oid>" --scope "/subscriptions/$subscription/resourceGroups/$personaRg"
> az role assignment create --role "User Access Administrator" --assignee "<your-oid>" --scope "/subscriptions/$subscription/resourceGroups/$personaRg"
> ```

### 1.3 Terminal T1 (Owner) — Variables

```powershell
# --- Azure login (must be done first) ---
# az login
# az account set --subscription "<owner-subscription-name-or-id>"
$account = az account show -o json | ConvertFrom-Json
$subscription = $account.id
$tenantId = $account.tenantId
Write-Host "Subscription: $subscription, Tenant: $tenantId"

# --- Location ---
$location = "westus"

# --- ARM endpoints ---
$armEndpoint = "https://eastus2euap.management.azure.com"
$armApiVersion = "2026-03-31-preview"

# --- Collaboration ---
$collabName = "<collaboration-name>"
$collabRg = "<collaboration-resource-group>"
```

### 1.4 Each Collaborator Terminal — Variables

Run in **each collaborator terminal** (T2, and T3 for multi-collaborator).
Only the persona-specific values differ — the rest is identical.

```powershell
# --- Azure login (must be done first) ---
# az login
# az account set --subscription "<your-subscription-name-or-id>"
$account = az account show -o json | ConvertFrom-Json
$subscription = $account.id
$tenantId = $account.tenantId
Write-Host "Subscription: $subscription, Tenant: $tenantId"

# --- Location ---
$location = "westus"

# --- Mode flags ---
$ApiMode = "cli"           # "cli" or "rest"
$EncryptionMode = "SSE"    # "SSE" or "CPK"

# --- Derived names (auto-adjust for encryption mode) ---
$suffix = if ($EncryptionMode -eq "CPK") { "-cpk" } else { "" }
$queryName = "query1$suffix"

# --- Persona (set to YOUR persona) ---
$persona = "woodgrove"                # T2: "woodgrove"  |  T3: "northwind"
$personaRg = "cr-e2e-woodgrove-rg"    # T2: "cr-e2e-woodgrove-rg"  |  T3: "cr-e2e-northwind-rg"
$personaEmail = "<your-email>"

# --- Create resource group (if it doesn't exist) ---
az group create --name $personaRg --location $location -o none 2>$null

# --- Frontend ---
$frontend = "https://dogfood.workload-frontendwestus.cleanroom.cloudapp.azure-test.net"

# --- OIDC storage account ---
# MSFT tenant: use pre-provisioned whitelisted SA
$oidcStorageAccount = "cleanroomoidc"
# Other tenants: leave empty — script creates one automatically
# $oidcStorageAccount = ""
```

### 1.5 Generate MSAL Token & Extract OID (each collaborator terminal)

> **CRITICAL**: The `oid` claim from your JWT is used for federated credentials in Step 05.
> See [Appendix A](#appendix-a-federated-credential-subject-reference).

```powershell
# Generate MSAL token
$token = Get-MsalToken -ClientId "8a3849c1-81c5-4d62-b83e-3bb2bb11251a" `
    -TenantId "common" -Scopes "User.Read" -DeviceCode
$personaTokenFile = Join-Path ([System.IO.Path]::GetTempPath()) "msal-idtoken-$persona.txt"
$token.IdToken | Out-File -FilePath $personaTokenFile -NoNewline
Write-Host "Token saved to: $personaTokenFile"

# Extract JWT oid
$tokenB64 = (Get-Content $personaTokenFile -Raw).Split('.')[1]
$padLen = (4 - $tokenB64.Length % 4) % 4
$padded = $tokenB64 + ('=' * $padLen)
$claims = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($padded)) | ConvertFrom-Json
$personaOid = $claims.oid
Write-Host "JWT oid: $personaOid"
```

Sign in with **your** Microsoft account when prompted.

> For MSA (personal Microsoft accounts), the OID typically starts with `00000000-0000-0000-`.
> This is **different** from `az ad signed-in-user show --query id` — always use the JWT `oid`.

### 1.6 Configure CLI Extension (each collaborator terminal, CLI mode only)

```powershell
az managedcleanroom frontend configure --endpoint $frontend
```

### 1.7 Token Lifetime

MSAL tokens last ~24 hours. If a token expires mid-flow, regenerate it (repeat 1.5).

---

## Step 02: Create Collaboration `[OWNER]`

> **Terminal: T1 (Owner)**

### 2.1 Create Resource Group

```powershell
az group create --name $collabRg --location $location -o none
```

### 2.2 Create Collaboration

```powershell
$ownerObjectId = az ad signed-in-user show --query id -o tsv
$body = @{
    location = $location
    properties = @{
        consortiumType = "ConfidentialAKS"
        userIdentity = @{
            tenantId = $tenantId
            objectId = $ownerObjectId
            accountType = "MicrosoftAccount"
        }
    }
} | ConvertTo-Json -Depth 4 -Compress
$bodyFile = Join-Path ([System.IO.Path]::GetTempPath()) "create-collab.json"
[System.IO.File]::WriteAllText($bodyFile, $body)

az rest --method PUT `
    --url "$armEndpoint/subscriptions/$subscription/resourceGroups/$collabRg/providers/Private.CleanRoom/Collaborations/$collabName?api-version=$armApiVersion" `
    --resource "https://management.azure.com/" `
    --body "@$bodyFile"
```

**Expected**: 201 Created or 200 OK. May return 202 — poll the `Location` header (2-5 min).

### 2.3 Enable Analytics Workload

```powershell
$body = @{ workloadType = "analytics" } | ConvertTo-Json -Compress
$bodyFile = Join-Path ([System.IO.Path]::GetTempPath()) "enable-workload.json"
[System.IO.File]::WriteAllText($bodyFile, $body)

az rest --method POST `
    --url "$armEndpoint/subscriptions/$subscription/resourceGroups/$collabRg/providers/Private.CleanRoom/Collaborations/$collabName/enableWorkload?api-version=$armApiVersion" `
    --resource "https://management.azure.com/" `
    --body "@$bodyFile"
```

> **CRITICAL**: Only pass `workloadType`. Do NOT pass `securityPolicyOption`.

**Expected**: 202 Accepted. Poll until complete (**5-15 minutes**).

> After this step, the Owner is done with setup. Remaining ARM operations (add collaborators)
> are in Step 03. The Owner does NOT need the `managedcleanroom` CLI extension or frontend access.

---

## Step 03: Add Collaborators & Accept Invitations `[OWNER → EACH COLLABORATOR]`

### 3.1 Add Collaborators — Terminal T1 (Owner)

Repeat for each collaborator email:

```powershell
$collaboratorEmail = "<collaborator-email>"
$body = @{ Collaborator = @{ UserIdentifier = $collaboratorEmail } } | ConvertTo-Json -Depth 3 -Compress
$bodyFile = Join-Path ([System.IO.Path]::GetTempPath()) "add-collaborator.json"
[System.IO.File]::WriteAllText($bodyFile, $body)

az rest --method POST `
    --url "$armEndpoint/subscriptions/$subscription/resourceGroups/$collabRg/providers/Private.CleanRoom/Collaborations/$collabName/addCollaborator?api-version=$armApiVersion" `
    --resource "https://management.azure.com/" `
    --body "@$bodyFile"
```

**Expected**: 202 Accepted for each.

### 3.2 Get Collaboration UUID — Each Collaborator Terminal

> The ARM `properties.collaborationId` field is `null` (known bug). Each collaborator
> discovers the frontend UUID themselves via the frontend API.

```powershell
$env:MANAGEDCLEANROOM_ACCESS_TOKEN = Get-Content $personaTokenFile -Raw
$env:AZURE_CLI_DISABLE_CONNECTION_VERIFICATION = "1"

# List all collaborations visible to you
$collabs = (az managedcleanroom frontend collaboration list -o json | ConvertFrom-Json).collaborations
$collabs | Format-Table @{L='#';E={[array]::IndexOf($collabs,$_)+1}}, collaborationName, collaborationId, userStatus

# Choose your collaboration (enter the number)
$choice = Read-Host "Enter the number of your collaboration"
$collabId = $collabs[[int]$choice - 1].collaborationId
Write-Host "Selected: $collabId"
```

### 3.3 List & Accept Invitation — Each Collaborator Terminal

```powershell
# List pending invitations
$invitations = (az managedcleanroom frontend invitation list `
    --collaboration-id $collabId -o json | ConvertFrom-Json).invitations
$invitations | Format-Table invitationId, accountType, status

# Get the invitation ID
$invitationId = $invitations[0].invitationId
Write-Host "Invitation ID: $invitationId"

# Accept the invitation
az managedcleanroom frontend invitation accept `
    --collaboration-id $collabId `
    --invitation-id $invitationId
```

**Verify** (any collaborator terminal):
```powershell
az managedcleanroom frontend show --collaboration-id $collabId -o json
```
Should show `"status": "Active"` once all invitations are accepted.

---

## Step 04: Resource Provisioning `[EACH COLLABORATOR]`

> Run these commands in **each collaborator terminal**. The commands are identical —
> only the `$persona*` variables differ.
>
> In multi-collaborator mode, terminals run in parallel (independent resource groups).

### 4.1 Prepare Resources

> Login and resource group creation were already done in Step 1.4.

```powershell
./scripts/04-prepare-resources.ps1 -resourceGroup $personaRg -persona $persona -location $location
```

**Verify**: `generated/$personaRg/names.generated.ps1` and `resources.generated.json` exist.

### 4.2 Download Sample Data

Downloads Twitter CSV data from the `Azure-Samples/Synapse` GitHub repo into the local data directory.

```powershell
. "./scripts/common/get-input-data.ps1"
$dataDir = "./generated/datasource/$persona"

if ($persona -eq "woodgrove") {
    Get-ConsumerData -dataDir $dataDir -format csv -schemaFields "date:date,time:string,author:string,mentions:string"
} else {
    Get-PublisherData -dataDir $dataDir -format csv -schemaFields "date:date,time:string,author:string,mentions:string"
}
```

**Verify**: `generated/datasource/$persona/csv/` contains `.csv` files.

### 4.3 Upload Data

#### SSE Mode

```powershell
./scripts/05-prepare-data-sse.ps1 -resourceGroup $personaRg -persona $persona `
    -dataDir "./generated/datasource/$persona"
```

#### CPK Mode

```powershell
./scripts/05-prepare-data-cpk.ps1 -resourceGroup $personaRg -persona $persona `
    -dataDir "./generated/datasource/$persona/csv" `
    -datasetSuffix "-cpk"
```

> CPK mode generates a 32-byte DEK per dataset, uploads data via `azcopy --cpk-by-value`,
> and saves DEK files to `generated/datastores/keys/`.
> See [Appendix C: CPK Deep Dive](#appendix-c-cpk-deep-dive) for details.

**Verify**: `generated/datastores/$persona-datastore-metadata.json` exists.

---

## Step 05: OIDC Identity & Federated Credentials `[EACH COLLABORATOR]`

> Run these commands in **each collaborator terminal**.

### 5.1 Setup OIDC Issuer

```powershell
$identityParams = @{
    resourceGroup    = $personaRg
    persona          = $persona
    collaborationId  = $collabId
    frontendEndpoint = $frontend
    TokenFile        = $personaTokenFile
    ApiMode          = $ApiMode
}
if ($oidcStorageAccount) { $identityParams["OidcStorageAccount"] = $oidcStorageAccount }

./scripts/06-setup-identity.ps1 @identityParams
```

> **MSFT tenant**: Uses the pre-provisioned whitelisted SA (`cleanroomoidc`). If this SA
> is in a different tenant than your `az login` session, the upload will fail with
> "Issuer validation failed". Open a separate terminal, `az login --tenant <msft-tenant-id>`,
> and run the OIDC upload from there.
>
> **All other tenants**: Omit `-OidcStorageAccount`. The script automatically creates a new
> storage account with static website enabled in the collaborator's resource group.

**Verify**:
```powershell
Get-Content "generated/$personaRg/issuer-url.txt"
```

### 5.2 Grant Access & Create Federated Credentials

> **CRITICAL**: `-userId` must be the **JWT `oid`** from Step 01.5 — NOT a persona name.
> See [Appendix A](#appendix-a-federated-credential-subject-reference).

```powershell
./scripts/07-grant-access.ps1 -resourceGroup $personaRg `
    -collaborationId $collabId -contractId "Analytics" `
    -userId $personaOid -EncryptionMode $EncryptionMode
```

> **CRITICAL**: `contractId` must be `"Analytics"` (capital A). Lowercase causes silent
> token exchange failures at runtime.
>
> **CPK**: `-EncryptionMode CPK` automatically grants the managed identity `Key Vault Crypto Officer` and
> `Key Vault Secrets User` roles needed for KEK release and DEK unwrapping.

**Wait**: RBAC propagation takes 60-120 seconds. The script waits and retries.

**Verify**:
```powershell
. "generated/$personaRg/names.generated.ps1"
az identity federated-credential list `
    --identity-name $MANAGED_IDENTITY_NAME `
    --resource-group $personaRg -o table
# Subject should be: Analytics-<jwt-oid>
```

---

## Step 06: Publish Datasets `[EACH COLLABORATOR]`

> Run in **each collaborator terminal**.
>
> - Woodgrove publishes: 1 input dataset (read) + 1 output dataset (write)
> - Northwind publishes: 1 input dataset (read) only

### SSE Mode

```powershell
./scripts/08-publish-dataset-sse.ps1 -collaborationId $collabId `
    -resourceGroup $personaRg -persona $persona `
    -frontendEndpoint $frontend `
    -TokenFile $personaTokenFile `
    -ApiMode $ApiMode
```

### CPK Mode

```powershell
./scripts/08-publish-dataset-cpk.ps1 -collaborationId $collabId `
    -resourceGroup $personaRg -persona $persona `
    -frontendEndpoint $frontend `
    -TokenFile $personaTokenFile `
    -ApiMode $ApiMode
```

> **CRITICAL**: The `issuerUrl` in the dataset body must be the **public OIDC URL**
> (e.g., `https://cleanroomoidc.z22.web.core.windows.net/<uuid>`), never `"https://cgs/oidc"`.
> The scripts read from `generated/<rg>/issuer-url.txt` automatically.

**Expected**: 204 No Content (the CLI may print warnings — this is normal).

**Verify**:
```powershell
$env:MANAGEDCLEANROOM_ACCESS_TOKEN = Get-Content $personaTokenFile -Raw
$env:AZURE_CLI_DISABLE_CONNECTION_VERIFICATION = "1"
az managedcleanroom frontend analytics dataset show `
    --collaboration-id $collabId `
    --document-id "$persona-input-csv$suffix" -o json
```
Should show `"state": "Accepted"`.

---

## Step 07: CPK Key Management `[EACH COLLABORATOR]` _(CPK only)_

> **Skip this step for SSE mode.**
>
> Verify that the key management performed in Step 06 completed correctly.

```powershell
. "generated/$personaRg/names.generated.ps1"

$datasetName = "$persona-input-csv-cpk"   # e.g., "woodgrove-input-csv-cpk"
$kekName = "$datasetName-kek"

# Check KEK
az keyvault key show --vault-name $KEYVAULT_NAME --name $kekName `
    --query '{exportable:attributes.exportable, keyType:key.kty, ops:key.keyOps}' -o json
# Expected: exportable: true, keyType: "RSA-HSM", ops: ["encrypt", "wrapKey"]

# Check wrapped DEK secret
az keyvault secret show --vault-name $KEYVAULT_NAME `
    --name "wrapped-$datasetName-dek-$kekName" `
    --query '{name:name, id:id}' -o json

# Check published dataset has kek/dek references
$env:MANAGEDCLEANROOM_ACCESS_TOKEN = Get-Content $personaTokenFile -Raw
$env:AZURE_CLI_DISABLE_CONNECTION_VERIFICATION = "1"
az managedcleanroom frontend analytics dataset show `
    --collaboration-id $collabId --document-id $datasetName -o json
# Should include kek.kid, kek.maaUrl, dek.secretId
```

> If any verification fails, see [Appendix C: CPK Troubleshooting](#cpk-troubleshooting).

---

## Step 08: Publish Query `[WOODGROVE]`

> **Terminal: T2 (Woodgrove)** — Woodgrove proposes the SQL query.
>
> `$suffix` and `$queryName` were set in Step 1.4 based on `$EncryptionMode`.

### Single-collaborator query (Woodgrove data only)

```powershell
./scripts/09-publish-query.ps1 -collaborationId $collabId `
    -queryName $queryName `
    -queryDir "./demos/query/woodgrove/query1" `
    -publisherInputDataset "woodgrove-input-csv$suffix" `
    -consumerInputDataset "woodgrove-input-csv$suffix" `
    -outputDataset "woodgrove-output-csv$suffix" `
    -frontendEndpoint $frontend `
    -TokenFile $personaTokenFile `
    -ApiMode $ApiMode
```

### Multi-collaborator query (both datasets) — _multi-collaborator only_

```powershell
./scripts/09-publish-query.ps1 -collaborationId $collabId `
    -queryName "query2$suffix" `
    -queryDir "./demos/query/woodgrove/query1" `
    -publisherInputDataset "northwind-input-csv$suffix" `
    -consumerInputDataset "woodgrove-input-csv$suffix" `
    -outputDataset "woodgrove-output-csv$suffix" `
    -frontendEndpoint $frontend `
    -TokenFile $personaTokenFile `
    -ApiMode $ApiMode
```

The query has 3 segments (in `demos/query/woodgrove/query1/`):
- **Segment 1** (seq=1): `CREATE OR REPLACE TEMP VIEW publisher_view AS SELECT * FROM publisher_data`
- **Segment 2** (seq=1): `CREATE OR REPLACE TEMP VIEW consumer_view AS SELECT * FROM consumer_data`
- **Segment 3** (seq=2): `SELECT author, COUNT(*) ... FROM (... UNION ALL ...) WHERE mentions LIKE '%MikeDoesBigData%' GROUP BY author`

**Expected**: 204 No Content.

---

## Step 09: Approve Query `[EACH COLLABORATOR]`

> **Single-collaborator**: Only Woodgrove votes (one vote → `Accepted`).
>
> **Multi-collaborator**: Both collaborators must vote on queries that reference their data.

```powershell
./scripts/10-vote-query.ps1 -collaborationId $collabId -queryName $queryName `
    -frontendEndpoint $frontend `
    -TokenFile $personaTokenFile `
    -ApiMode $ApiMode
```

For `query2` _(multi-collaborator only)_ — **each** collaborator runs:
```powershell
./scripts/10-vote-query.ps1 -collaborationId $collabId -queryName "query2$suffix" `
    -frontendEndpoint $frontend `
    -TokenFile $personaTokenFile `
    -ApiMode $ApiMode
```

> **Consent ordering** (multi-collaborator): Consent can only be enabled on an `Accepted`
> query. The query stays `Proposed` until all votes are in. The script enables consent
> automatically after voting. If the first voter's consent fails (query still `Proposed`),
> re-run after the last voter votes.

**Verify**: Query shows `"state": "Accepted"` with `proposalId` populated.

---

## Step 10: Execute Query `[WOODGROVE]`

> **Terminal: T2 (Woodgrove)**

```powershell
./scripts/11-run-query.ps1 -collaborationId $collabId -queryName $queryName `
    -frontendEndpoint $frontend `
    -TokenFile $personaTokenFile `
    -ApiMode $ApiMode
```

**Expected**: `{"status": "success", "id": "cl-spark-<uuid>"}`

```powershell
$jobId = "cl-spark-<uuid>"
```

> `"status": "success"` means **accepted for scheduling**, not completed. Takes **10-20 min**.

---

## Step 11: Monitor Query `[ANY]`

> Run from any collaborator terminal.

### 11.1 Real-Time Status

```powershell
./scripts/13-run-status.ps1 -collaborationId $collabId `
    -jobId $jobId `
    -frontendEndpoint $frontend `
    -TokenFile $personaTokenFile `
    -ApiMode $ApiMode `
    -poll -pollIntervalSeconds 30
```

| Time | State | Key Events |
|---|---|---|
| +0 min | `SUBMITTED` | `SparkApplicationSubmitted` |
| +5-8 min | `RUNNING` | `SparkDriverRunning` |
| +8-12 min | `RUNNING` | `DATASET_LOAD_STARTED/COMPLETED` |
| +10-15 min | `RUNNING` | `QUERY_SEGMENT_EXECUTION_*` |
| +15-20 min | `COMPLETED` | `QUERY_EXECUTION_COMPLETED`, `SparkDriverCompleted` |

> **`PENDING_RERUN`** is normal — transitions to `SUBMITTED` → `RUNNING` automatically.

### 11.2 Audit Events

```powershell
./scripts/15-audit-events.ps1 -collaborationId $collabId `
    -frontendEndpoint $frontend `
    -TokenFile $personaTokenFile `
    -ApiMode $ApiMode
```

---

## Step 12: Results & Audit `[WOODGROVE]`

> **Terminal: T2 (Woodgrove)** — Woodgrove owns the output dataset.

### 12.1 Run History

```powershell
./scripts/14-run-history.ps1 -collaborationId $collabId `
    -queryName $queryName `
    -frontendEndpoint $frontend `
    -TokenFile $personaTokenFile `
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
  }
}
```

> Returns 404 if no terminal runs yet. **Quirk**: try `--document-id Analytics` if
> `runhistory list` returns empty for a completed run.

### 12.2 Download Output (SSE)

```powershell
. "generated/$personaRg/names.generated.ps1"

# Find the output CSV blob for the latest run (uses $jobId from Step 10)
# Strip the "cl-spark-" prefix to get the run UUID used in the blob path
$runUuid = $jobId -replace '^cl-spark-', ''
$blobs = az storage blob list --account-name $STORAGE_ACCOUNT_NAME `
    --container-name woodgrove-output `
    --prefix "Analytics/" --auth-mode login -o json | ConvertFrom-Json
$csvBlob = ($blobs | Where-Object {
    $_.name -match '\.csv$' -and $_.name -notmatch '\.crc$' -and $_.name -match $runUuid
}).name
Write-Host "Output blob: $csvBlob"

# Download it
$outputFile = Join-Path "." "output.csv"
az storage blob download --account-name $STORAGE_ACCOUNT_NAME `
    --container-name woodgrove-output `
    --name $csvBlob `
    --file $outputFile --auth-mode login

Get-Content $outputFile
```

> **NOTE**: Multiple runs produce separate output blobs under `Analytics/<date>/<run-uuid>/`.
> The filter uses `$jobId` to select the blob from the current run. If `$jobId` is not set,
> pick the latest blob manually:
> ```powershell
> $csvBlob = ($blobs | Where-Object { $_.name -match '\.csv$' -and $_.name -notmatch '\.crc$' } |
>     Sort-Object -Property @{E={$_.properties.lastModified}} -Descending | Select-Object -First 1).name
> ```

### 12.3 Download Output (CPK)

```powershell
. "generated/$personaRg/names.generated.ps1"

./scripts/12-view-results.ps1 -collaborationId $collabId `
    -queryName $queryName `
    -frontendEndpoint $frontend `
    -TokenFile $personaTokenFile `
    -ApiMode $ApiMode `
    -DownloadCpkOutput `
    -OutputDekFile "generated/datastores/keys/woodgrove-output-csv$suffix-dek.bin" `
    -OutputStorageAccount $STORAGE_ACCOUNT_NAME
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
| JWT `oid` (correct) | PowerShell decode from Step 01.5 | **YES** |
| Graph API (wrong) | `az ad signed-in-user show --query id` | **NO** |
| Dataset `proposerId` | `dataset show` response | **YES** (to verify) |

### Verification

```powershell
$env:MANAGEDCLEANROOM_ACCESS_TOKEN = Get-Content $personaTokenFile -Raw
$env:AZURE_CLI_DISABLE_CONNECTION_VERIFICATION = "1"
az managedcleanroom frontend analytics dataset show `
    --collaboration-id $collabId `
    --document-id "$persona-input-csv" -o json
# Check proposerId — should match $personaOid
```

### Fixing Wrong Subjects

```powershell
. "generated/$personaRg/names.generated.ps1"

# Delete wrong credential
az identity federated-credential delete `
    --name "Analytics-$persona-federation" `
    --identity-name $MANAGED_IDENTITY_NAME --resource-group $personaRg --yes

# Create correct credential
az identity federated-credential create `
    --name "Analytics-$personaOid-federation" `
    --identity-name $MANAGED_IDENTITY_NAME --resource-group $personaRg `
    --issuer "$(Get-Content generated/$personaRg/issuer-url.txt)" `
    --subject "Analytics-$personaOid" `
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

**Fix**: Use MSAL IdTokens (Step 01.5).

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
$env:MANAGEDCLEANROOM_ACCESS_TOKEN = Get-Content $personaTokenFile -Raw
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
**Workaround**: Use `az managedcleanroom frontend collaboration list` (Step 03.2).

### 3. `--consortium-type` and `--user-identity` not valid CLI parameters

The `az managedcleanroom collaboration create` command doesn't accept these parameters.
**Workaround**: Use `az rest` for collaboration creation (Step 02.1).

### 4. ~~`08-publish-dataset-cpk.ps1` double `ConvertFrom-Json`~~ (FIXED)

Removed the redundant `ConvertFrom-Json` call on the already-parsed `Invoke-AzCli` result.

### 5. ~~`07-grant-access.ps1` missing `-setupKeyVault`~~ (FIXED)

Added `-setupKeyVault` as a passthrough switch parameter.

### 6. ~~`12-view-results.ps1` azcopy fails on HNS blobs~~ (FIXED)

Added `--include-pattern "*.csv;*.crc;*_SUCCESS*"` to skip HNS directory marker blobs.
