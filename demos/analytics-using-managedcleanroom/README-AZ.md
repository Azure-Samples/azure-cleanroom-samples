# Big Data Analytics — Azure CLI (`az managedcleanroom`)

This guide uses **`az managedcleanroom`** CLI commands for all collaboration (ARM)
and frontend operations, with helper scripts for Azure resource provisioning.

For the REST API variant using `Invoke-RestMethod` and `az rest`, see
[README-API.md](README-API.md).

---

## Scenario

Woodgrove is an advertiser that wants to generate target audience segments by
performing an overlap analysis with a media publisher, Northwind. Both parties
contribute sensitive datasets to an
[Azure Confidential Clean Room](https://learn.microsoft.com/en-us/azure/confidential-computing/confidential-clean-rooms)
where a Spark SQL query joins the data, computes the overlap, and writes the
results — all without either party exposing raw data to the other.

This is only a sample scenario. You can try any scenario of your choice by
providing your own data and query.

## Overview

| Aspect | Details |
|---|---|
| **API mode** | `az managedcleanroom` CLI extension |
| **Encryption** | SSE (Azure-managed) or CPK (customer-provided keys via Key Vault Premium) |
| **Parties** | Woodgrove (owner / advertiser), Northwind (publisher) |
| **Data format** | CSV (Parquet and JSON also supported) |
| **Query engine** | Confidential Spark SQL |

### Parties Involved

| Party | Role |
|:---|:---|
| **Woodgrove** | Clean room **owner** — creates the collaboration, invites Northwind, publishes the query, runs it, and retrieves results. Also contributes sensitive first-party user data. |
| **Northwind** | Data **publisher** — accepts the invitation and contributes sensitive subscriber data which can be matched with Woodgrove's data to identify common users. |

### Which Party Runs Which Step?

| Step | Woodgrove | Northwind | Notes |
|:-----|:---------:|:---------:|:------|
| 01 — Prerequisites | &#10003; | &#10003; | Both authenticate and set variables |
| 02 — Create collaboration | &#10003; | | Owner only (ARM) |
| 03 — Accept invitation | | &#10003; | Each invited collaborator |
| 04 — Provision resources | &#10003; | &#10003; | Independent resource groups |
| 05 — OIDC identity | &#10003; | &#10003; | Federated credential per collaborator |
| 06 — Publish datasets | &#10003; (input + output) | &#10003; (input only) | Woodgrove also publishes output |
| 07 — Publish query | &#10003; | | Woodgrove proposes queries |
| 08 — Approve query | &#10003; | &#10003; | All affected collaborators vote |
| 09 — Execute query | &#10003; | | Woodgrove triggers execution |
| 10 — Monitor query | &#10003; | &#10003; | Any collaborator can poll |
| 11 — Results & audit | &#10003; | &#10003; | Woodgrove downloads; both view audit |

---

## Table of Contents

- [Scenario](#scenario)
- [Overview](#overview)
- [Step 01: Prerequisites](#step-01-prerequisites) `[ALL]`
- [Step 02: Create Collaboration](#step-02-create-collaboration) `[OWNER]`
- [Step 03: Accept Invitations](#step-03-accept-invitations) `[EACH COLLABORATOR]`
- [Step 04: Provision Resources & Upload Data](#step-04-provision-resources--upload-data) `[EACH COLLABORATOR]`
- [Step 05: OIDC Identity & Access](#step-05-oidc-identity--access) `[EACH COLLABORATOR]`
- [Step 06: Publish Datasets](#step-06-publish-datasets) `[EACH COLLABORATOR]`
- [Step 07: Publish Query](#step-07-publish-query) `[WOODGROVE]`
- [Step 08: Approve Query](#step-08-approve-query) `[EACH COLLABORATOR]`
- [Step 09: Execute Query](#step-09-execute-query) `[WOODGROVE]`
- [Step 10: Monitor Query](#step-10-monitor-query) `[ANY]`
- [Step 11: Results & Audit](#step-11-results--audit) `[WOODGROVE]`
- [Appendix A: Federated Credential Subject Reference](#appendix-a-federated-credential-subject-reference)
- [Appendix B: Troubleshooting](#appendix-b-troubleshooting)
- [Appendix C: CPK Deep Dive](#appendix-c-cpk-deep-dive)
- [Appendix D: Dataset Schema Reference](#appendix-d-dataset-schema-reference)
- [Appendix E: Query Structure Reference](#appendix-e-query-structure-reference)

---

## Step 01: Prerequisites `[ALL]`

### 1.1 Requirements

| Requirement | Details |
|---|---|
| Azure CLI | 2.75.0+ |
| `managedcleanroom` extension | `az extension add --name managedcleanroom` (v1.0.0b5+) |
| PowerShell | 7.x+ |
| MSAL.PS module | `Install-Module MSAL.PS -Scope CurrentUser -Force` |
| azcopy | v10+ (CPK mode only) |

### 1.2 Terminal T1 (Owner) — Variables

```powershell
az login
$account = az account show -o json | ConvertFrom-Json
$subscription = $account.id
$tenantId = $account.tenantId

$location = "westus"
$collabName = "<collaboration-name>"
$collabRg = "<collaboration-resource-group>"
```

### 1.3 Each Collaborator Terminal — Variables

```powershell
az login
$account = az account show -o json | ConvertFrom-Json
$subscription = $account.id
$tenantId = $account.tenantId

$location = "westus"
$EncryptionMode = "SSE"    # "SSE" or "CPK"
$iteration = 0

$persona = "woodgrove"                # T2: "woodgrove"  |  T3: "northwind"
$personaRg = "cr-e2e-$persona-rg"
$personaEmail = "<your-email>"

az group create --name $personaRg --location $location -o none 2>$null

$frontend = "https://dogfood.workload-frontendwestus.cleanroom.cloudapp.azure-test.net"
$oidcStorageAccount = "cleanroomoidc"   # MSFT tenant; omit for other tenants
```

### 1.4 Generate MSAL Token & Extract OID

```powershell
$token = Get-MsalToken -ClientId "8a3849c1-81c5-4d62-b83e-3bb2bb11251a" `
    -TenantId "common" -Scopes "User.Read" -DeviceCode
$personaTokenFile = Join-Path ([System.IO.Path]::GetTempPath()) "msal-idtoken-$persona.txt"
$token.IdToken | Out-File -FilePath $personaTokenFile -NoNewline

# Extract JWT oid (used for federated credentials in Step 05)
$tokenB64 = (Get-Content $personaTokenFile -Raw).Split('.')[1]
$padLen = (4 - $tokenB64.Length % 4) % 4
$padded = $tokenB64 + ('=' * $padLen)
$claims = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($padded)) | ConvertFrom-Json
$personaOid = $claims.oid
Write-Host "JWT oid: $personaOid"
```

> **CRITICAL**: Always use the JWT `oid`, NOT `az ad signed-in-user show --query id`.
> For MSA accounts these differ. See [Appendix A](#appendix-a-federated-credential-subject-reference).

### 1.5 Configure CLI Extension

```powershell
$env:MANAGEDCLEANROOM_ACCESS_TOKEN = Get-Content $personaTokenFile -Raw
$env:AZURE_CLI_DISABLE_CONNECTION_VERIFICATION = "1"
az managedcleanroom frontend configure --endpoint $frontend
```

---

## Step 02: Create Collaboration `[OWNER]`

> **Terminal: T1 (Owner)**

### 2.1 Configure Private CleanRoom Cloud

```powershell
$privateCloudName = "PrivateCleanroomAzureCloud"
az cloud register --name $privateCloudName `
    --endpoint-resource-manager "https://eastus2euap.management.azure.com/" 2>$null
az cloud set --name $privateCloudName
$env:UsePrivateCleanRoomNamespace = "true"
```

### 2.2 Create Collaboration & Enable Workload

```powershell
az group create --name $collabRg --location $location -o none

az managedcleanroom collaboration create `
    --collaboration-name $collabName `
    --resource-group $collabRg `
    --location $location
```

**Runtime**: ~25 minutes.

```powershell
az managedcleanroom collaboration enable-workload `
    --collaboration-name $collabName `
    --resource-group $collabRg `
    --workload-type Analytics
```

**Runtime**: ~7 minutes.

### 2.3 Add Collaborators

Repeat for each collaborator:

```powershell
# Add Woodgrove
az managedcleanroom collaboration add-collaborator `
    --collaboration-name $collabName `
    --resource-group $collabRg `
    --user-identifier "<woodgrove-email>"

# Add Northwind (multi-collaborator only)
az managedcleanroom collaboration add-collaborator `
    --collaboration-name $collabName `
    --resource-group $collabRg `
    --user-identifier "<northwind-email>"
```

**Verify**:
```powershell
az managedcleanroom collaboration show `
    --collaboration-name $collabName `
    --resource-group $collabRg
```

---

## Step 03: Accept Invitations `[EACH COLLABORATOR]`

### 3.1 Get Collaboration UUID

```powershell
$env:MANAGEDCLEANROOM_ACCESS_TOKEN = Get-Content $personaTokenFile -Raw
$env:AZURE_CLI_DISABLE_CONNECTION_VERIFICATION = "1"

$collabs = (az managedcleanroom frontend collaboration list -o json | ConvertFrom-Json).collaborations
$collabs | Format-Table @{L='#';E={[array]::IndexOf($collabs,$_)+1}}, collaborationName, collaborationId, userStatus

$choice = Read-Host "Enter the number of your collaboration"
$collabId = $collabs[[int]$choice - 1].collaborationId
Write-Host "Selected: $collabId"
```

### 3.2 Accept Invitation

```powershell
$invitations = (az managedcleanroom frontend invitation list `
    --collaboration-id $collabId -o json | ConvertFrom-Json).invitations
$invitations | Format-Table invitationId, accountType, status

$invitationId = $invitations[0].invitationId
az managedcleanroom frontend invitation accept `
    --collaboration-id $collabId `
    --invitation-id $invitationId
```

---

## Step 04: Provision Resources & Upload Data `[EACH COLLABORATOR]`

> Run Steps 04-06 in **each collaborator terminal**. Commands are identical —
> only `$persona` differs. In multi-collaborator mode, Woodgrove (T2) and
> Northwind (T3) run these steps **in parallel** (independent resource groups).

### 4.1 Prepare Resources

```powershell
./scripts/04-prepare-resources.ps1 -resourceGroup $personaRg -persona $persona -location $location
```

### 4.2 Generate Sample Data

```powershell
./demos/generate-data.ps1 -persona $persona
```

### 4.3 Set Dataset Names

```powershell
$iteration++
$suffix = if ($EncryptionMode -eq "CPK") { "-cpk-v$iteration" } else { "-v$iteration" }
$queryName = "query1$suffix"
Write-Host "Iteration: $iteration | Suffix: '$suffix' | Query: '$queryName'"
```

### 4.4 Upload Data

```powershell
$variant = if ($EncryptionMode -eq "CPK") { "cpk" } else { "sse" }
./scripts/05-prepare-data.ps1 -resourceGroup $personaRg `
    -variant $variant -persona $persona `
    -dataDir "./generated/datasource/$persona/csv" `
    -datasetSuffix "$suffix"
```

---

## Step 05: OIDC Identity & Access `[EACH COLLABORATOR]`

> **How OIDC works**: The clean room has no credentials of its own. At runtime it
> proves its identity via hardware attestation, receives a signed JWT from CGS, and
> exchanges it for an Azure AD token. The OIDC issuer URL makes this exchange work.

### 5.1 Fetch JWKS from Frontend

```powershell
$env:MANAGEDCLEANROOM_ACCESS_TOKEN = Get-Content $personaTokenFile -Raw
$env:AZURE_CLI_DISABLE_CONNECTION_VERIFICATION = "1"
az managedcleanroom frontend configure --endpoint $frontend

$jwksDir = "generated/$personaRg"
New-Item -ItemType Directory -Path $jwksDir -Force | Out-Null

az managedcleanroom frontend oidc keys `
    --collaboration-id $collabId -o json > "$jwksDir/jwks.json"
```

### 5.2 Setup OIDC Storage & Upload Documents

```powershell
$oidcParams = @{
    resourceGroup   = $personaRg
    persona         = $persona
    collaborationId = $collabId
    JwksFile        = "generated/$personaRg/jwks.json"
}
if ($oidcStorageAccount) { $oidcParams["OidcStorageAccount"] = $oidcStorageAccount }

./scripts/06-setup-oidc-storage.ps1 @oidcParams
```

### 5.3 Register Issuer URL with Frontend

```powershell
$issuerUrl = (Get-Content "generated/$personaRg/issuer-url.txt" -Raw).Trim()

az managedcleanroom frontend oidc set-issuer-url `
    --collaboration-id $collabId `
    --url $issuerUrl
```

### 5.4 Grant Access & Create Federated Credentials

```powershell
./scripts/07-grant-access.ps1 -resourceGroup $personaRg `
    -collaborationId $collabId -contractId "Analytics" `
    -userId $personaOid -EncryptionMode $EncryptionMode
```

> **CRITICAL**: `contractId` must be `"Analytics"` (capital A). `-userId` must be
> the JWT `oid` from Step 01.4.

**Verify**:
```powershell
. "generated/$personaRg/names.generated.ps1"
az identity federated-credential list `
    --identity-name $MANAGED_IDENTITY_NAME `
    --resource-group $personaRg -o table
```

---

## Step 06: Publish Datasets `[EACH COLLABORATOR]`

> Woodgrove publishes input + output datasets. Northwind publishes input only.
> See [Appendix D](#appendix-d-dataset-schema-reference) for schema details.

### 6.1 Build Dataset Body JSON

```powershell
./scripts/08-build-dataset-body.ps1 -resourceGroup $personaRg -persona $persona
```

### 6.2 Publish Input Dataset

```powershell
az managedcleanroom frontend analytics dataset publish `
    --collaboration-id $collabId `
    --document-id "$persona-input-csv$suffix" `
    --body "@generated/publish/$persona-input-dataset.json"
```

### 6.3 Publish Output Dataset (Woodgrove only)

```powershell
if ($persona -eq "woodgrove") {
    az managedcleanroom frontend analytics dataset publish `
        --collaboration-id $collabId `
        --document-id "woodgrove-output-csv$suffix" `
        --body "@generated/publish/woodgrove-output-dataset.json"
}
```

> Execution consent is enabled by default at publish time. To revoke or re-enable later:
> ```powershell
> az managedcleanroom frontend consent set `
>     --collaboration-id $collabId `
>     --document-id "<document-name>" `
>     --consent-action disable   # or "enable"
> ```

### 6.4 Prepare CPK Keys (CPK mode only)

> CPK keys must be created **after** publishing datasets. The script fetches the
> SKR (Secure Key Release) policy from the published dataset, which determines
> the attestation hash for the KEK release policy.

```powershell
if ($EncryptionMode -eq "CPK") {
    ./scripts/08-prepare-dataset-keys.ps1 -collaborationId $collabId `
        -resourceGroup $personaRg -persona $persona `
        -frontendEndpoint $frontend -TokenFile $personaTokenFile
}
```

**Verify**:
```powershell
az managedcleanroom frontend analytics dataset show `
    --collaboration-id $collabId `
    --document-id "$persona-input-csv$suffix" -o json
```

---

## Step 07: Publish Query `[WOODGROVE]`

> See [Appendix E](#appendix-e-query-structure-reference) for query format details.

### 7.1 Build Query Body

**Single-collaborator** (Woodgrove data only — both views point to the same dataset):

```powershell
./scripts/09-build-query-body.ps1 -queryName $queryName `
    -queryDir "./demos/query/woodgrove/query1" `
    -publisherInputDataset "woodgrove-input-csv$suffix" `
    -consumerInputDataset "woodgrove-input-csv$suffix" `
    -outputDataset "woodgrove-output-csv$suffix"
```

**Multi-collaborator** (cross-dataset JOIN — Northwind + Woodgrove):

> Get Northwind's exact dataset name (Northwind's suffix may differ from yours):
> ```powershell
> az managedcleanroom frontend analytics dataset list `
>     --collaboration-id $collabId -o json | ConvertFrom-Json |
>     Select-Object -ExpandProperty datasets |
>     Where-Object { $_.id -match "northwind" } |
>     ForEach-Object { Write-Host $_.id }
> ```

```powershell
$northwindDataset = "<northwind-input-csv-suffix>"   # e.g., "northwind-input-csv-v1"
./scripts/09-build-query-body.ps1 -queryName "query2$suffix" `
    -queryDir "./demos/query/woodgrove/query2" `
    -publisherInputDataset $northwindDataset `
    -consumerInputDataset "woodgrove-input-csv$suffix" `
    -outputDataset "woodgrove-output-csv$suffix"
```

### 7.2 Publish Query

```powershell
az managedcleanroom frontend analytics query publish `
    --collaboration-id $collabId `
    --document-id $queryName `
    --body "@generated/publish/$queryName.json"
```

---

## Step 08: Approve Query `[EACH COLLABORATOR]`

> **Single-collaborator**: Only Woodgrove votes (one vote → `Accepted`).
>
> **Multi-collaborator**: Both collaborators must vote. Northwind needs the
> `$queryName` from Woodgrove (or list queries to find it).

Each collaborator runs in their own terminal:

```powershell
# Get proposal ID
$queryInfo = az managedcleanroom frontend analytics query show `
    --collaboration-id $collabId `
    --document-id $queryName -o json | ConvertFrom-Json
$proposalId = $queryInfo.proposalId
Write-Host "Proposal ID: $proposalId"

# Vote
az managedcleanroom frontend analytics query vote `
    --collaboration-id $collabId `
    --document-id $queryName `
    --vote-action accept `
    --proposal-id $proposalId
```

> **Northwind (T3)**: If you don't have `$queryName`, list published queries:
> ```powershell
> az managedcleanroom frontend analytics query list `
>     --collaboration-id $collabId -o json
> ```

**Verify**: Query state should be `"Accepted"` after all required votes.

---

## Step 09: Execute Query `[WOODGROVE]`

```powershell
$runResult = az managedcleanroom frontend analytics query run `
    --collaboration-id $collabId `
    --document-id $queryName -o json | ConvertFrom-Json

$jobId = $runResult.id
Write-Host "Job ID: $jobId"
```

> The CLI auto-generates a run ID. Each invocation starts a new execution.

---

## Step 10: Monitor Query `[ANY]`

```powershell
do {
    Start-Sleep -Seconds 30
    $result = az managedcleanroom frontend analytics query runresult show `
        --collaboration-id $collabId `
        --job-id $jobId -o json | ConvertFrom-Json
    $state = $result.status.applicationState.state
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] State: $state"
} while ($state -notin @("COMPLETED", "FAILED", "SUBMISSION_FAILED"))

$result | ConvertTo-Json -Depth 10
```

| Time | State | Key Events |
|---|---|---|
| +0 min | `SUBMITTED` | `SparkApplicationSubmitted` |
| +5-8 min | `RUNNING` | `SparkDriverRunning` |
| +10-15 min | `RUNNING` | `QUERY_SEGMENT_EXECUTION_*` |
| +15-20 min | `COMPLETED` | `SparkDriverCompleted` |

> `PENDING_RERUN` is normal — transitions to `SUBMITTED` automatically.

---

## Step 11: Results & Audit `[WOODGROVE]`

### 11.1 Run History

```powershell
az managedcleanroom frontend analytics query runhistory list `
    --collaboration-id $collabId `
    --document-id $queryName -o json
```

### 11.2 Audit Events

```powershell
az managedcleanroom frontend analytics auditevent list `
    --collaboration-id $collabId -o json
```

### 11.3 Download Output

Auto-detects SSE/CPK mode from metadata. Pass `-JobId` to filter to a specific run.

```powershell
./scripts/11-download-output.ps1 -resourceGroup $personaRg `
    -datasetSuffix "$suffix" -JobId $jobId
```

> Output CSVs are saved to `generated/output/`. Without `-JobId`, downloads the latest.

---

## Appendix A: Federated Credential Subject Reference

Format: `{contractId}-{ownerId}` where `contractId` = `"Analytics"` (capital A)
and `ownerId` = JWT `oid` from Step 01.4.

MSA accounts: JWT `oid` ≠ `az ad signed-in-user show --query id`. Always use JWT `oid`.

**Fixing wrong subjects**:
```powershell
. "generated/$personaRg/names.generated.ps1"
az identity federated-credential delete --name "Analytics-$personaOid-federation" `
    --identity-name $MANAGED_IDENTITY_NAME --resource-group $personaRg --yes
az identity federated-credential create --name "Analytics-$personaOid-federation" `
    --identity-name $MANAGED_IDENTITY_NAME --resource-group $personaRg `
    --issuer "$(Get-Content generated/$personaRg/issuer-url.txt)" `
    --subject "Analytics-$personaOid" --audiences "api://AzureADTokenExchange"
```

---

## Appendix B: Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `SPARK_JOB_FAILED: ExitCode 1` | Federated credential subject mismatch | See [Appendix A](#appendix-a-federated-credential-subject-reference) |
| `AADSTS700211: No matching federated identity record` | Wrong issuer URL in dataset or stale FIC | Republish dataset; delete/recreate FIC |
| `SSL certificate verify failed` | Dogfood cert mismatch | Set `$env:AZURE_CLI_DISABLE_CONNECTION_VERIFICATION = "1"` |
| `404 Not Found` on frontend | Using ARM ID instead of frontend UUID | Use UUID from `frontend collaboration list` |
| `ContractNotFound` | Stale CCF endpoint | Create new collaboration |
| `Python 3.13 tuple error` | CLI extension bug | Upgrade to v1.0.0b5+ |
| `Already voted / Conflict` | Idempotent vote | Safe to ignore |
| `PENDING_RERUN` | Normal scheduling | Keep polling |

---

## Appendix C: CPK Deep Dive

| Aspect | SSE | CPK |
|---|---|---|
| Encryption | Azure-managed keys | Customer-provided keys per dataset |
| Key Vault | Not required | Required (Premium SKU with HSM) |
| Upload tool | `az storage blob upload-batch` | `azcopy copy --cpk-by-value` |
| Output download | `az storage blob download` | `azcopy copy --cpk-by-value` |

**Architecture**:
```
Upload:   plaintext CSV → azcopy --cpk-by-value → Azure Storage (encrypted with DEK)
Keys:     DEK → RSA-OAEP wrap with KEK → KV Secret (wrapped DEK)
          KEK (RSA-2048) → az keyvault key import (with SKR policy) → KV Key
Runtime:  SKR release → KEK private → unwrap DEK → CPK header → Storage → plaintext
```

> **CRITICAL**: CPK is server-side encryption. Do NOT manually encrypt files before upload.

---

## Appendix D: Dataset Schema Reference

| Dataset | Fields | Allowed Fields |
|---|---|---|
| **Northwind input** | `audience_id` (string), `hashed_email` (string), `annual_income` (long), `region` (string) | `hashed_email`, `annual_income`, `region` |
| **Woodgrove input** | `user_id` (string), `hashed_email` (string), `purchase_history` (string) | `hashed_email`, `purchase_history` |
| **Woodgrove output** | `user_id` (string) | `user_id` |

Fields not in `allowedFields` are excluded from query access — prevents PII exposure.
Supported formats: `csv`, `parquet`, `json`.

---

## Appendix E: Query Structure Reference

| Section | Purpose |
|---|---|
| `queryData.segments[]` | Ordered SQL statements with `executionSequence`, `data`, `preConditions`, `postFilters` |
| `inputDatasets[]` | Maps `datasetDocumentId` to SQL view names |
| `outputDataset` | Where results are written |

**Privacy controls**:
- `preConditions`: Min row count per view (query aborts if below threshold)
- `postFilters`: Remove output groups below aggregation threshold

---

## Appendix F: Collaboration Management

### Force Recover

If the collaboration becomes unresponsive (e.g., `ContractNotFound`, frontend errors on all operations):

```powershell
az managedcleanroom collaboration recover `
    --collaboration-name $collabName `
    --resource-group $collabRg `
    --force-recover $true
```

> Last-resort operation. Resets internal state. Existing datasets and queries
> need not be republished after recovery.

### Delete Collaboration

```powershell
az managedcleanroom collaboration delete `
    --collaboration-name $collabName `
    --resource-group $collabRg
```

> Permanently deletes the collaboration and all associated resources.

---

## Appendix: App-Based Authentication (SPN)

For CI/CD automation, service principals can replace interactive user login.

### Prerequisites

| Requirement | Details |
|---|---|
| Python 3 + `msal` + `cryptography` | `pip install msal cryptography` |
| App registration | With `serviceManagementReference` in MSFT tenant |
| OneCert certificate | Issued by integrated CA in a KV with OneCert issuer |
| `trustedCertificateSubjects` | Set in app manifest via Azure Portal |

### Token Acquisition

```powershell
# Use get-sp-token-sni.ps1 for MSAL SNI (x5c) auth
$token = ./scripts/common/get-sp-token-sni.ps1 `
    -appId "<clientAppId>" -tenantId "<tenantId>" -certPemPath "<cert.pem>"
$env:CLEANROOM_FRONTEND_TOKEN = $token
```

### Add SPN as Collaborator

```powershell
az managedcleanroom collaboration add-collaborator `
    --collaboration-name <name> --resource-group <rg> `
    --user-identifier <clientAppId> `
    --object-id <spObjectId> `
    --tenant-id <tenantId>
```

> **Note**: `--object-id` must be from the **Enterprise Application** (service principal), not the app registration. SPNs auto-activate — no invitation acceptance needed.

### Federated Credential Subject

Use the SP's Enterprise App object ID (same as the token's `oid` claim):

```
Analytics-{spObjectId}
```

### Troubleshooting

| Error | Fix |
|---|---|
| `AADSTS700027: certificate not registered` | Use Python MSAL with `public_certificate`, not `az login` |
| `Credential lifetime exceeds max value` | Use OneCert + `trustedCertificateSubjects` |
| `InvalidCollaboratorIdentifier` | Add `--object-id` and `--tenant-id` to `add-collaborator` |
| `is_schema_compatible: Missing field` | Output `allowedFields` must include all query output columns |
