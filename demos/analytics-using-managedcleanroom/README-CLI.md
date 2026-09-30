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
| **Data Encryption** | SSE (Microsoft Managed Keys) or [CPK](https://learn.microsoft.com/en-us/azure/storage/common/storage-service-encryption#about-encryption-key-management) (Customer Provided Keys) |
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
| 12 — Grafana dashboards | &#10003; | | Owner monitors via admin credentials |

---

## Table of Contents

- [Scenario](#scenario)
- [Overview](#overview)
- [Step 01: Prerequisites](#step-01-prerequisites) `[ALL]`
  - [1.1 Requirements](#11-requirements)
    - [IMPORTANT: Capacity planning for query execution](#important-capacity-planning-for-query-execution)
  - [1.2 Terminal T1 (Owner) — Variables](#12-terminal-t1-owner--variables)
  - [1.3 One-Time Owner Setup](#13-one-time-owner-setup)
  - [1.4 Each Collaborator Terminal — Variables](#14-each-collaborator-terminal--variables)
  - [1.5 Acquire Token, Extract OID & Configure CLI](#15-acquire-token-extract-oid--configure-cli-each-collaborator) `[EACH COLLABORATOR]`
- [Step 02: Create Collaboration](#step-02-create-collaboration) `[OWNER]`
  - [2.1 Create Resource Group](#21-create-resource-group)
  - [2.2 Create Collaboration](#22-create-collaboration)
  - [2.3 Enable Analytics Workload](#23-enable-analytics-workload)
  - [2.4 Add More Collaborators (Optional)](#24-add-more-collaborators-optional)
- [Step 03: Accept Invitations](#step-03-accept-invitations) `[EACH COLLABORATOR]`
- [Step 04: Provision Resources & Upload Data](#step-04-provision-resources--upload-data) `[EACH COLLABORATOR]`
- [Step 05: OIDC Identity & Access](#step-05-oidc-identity--access) `[EACH COLLABORATOR]`
- [Step 06: Publish Datasets](#step-06-publish-datasets) `[EACH COLLABORATOR]`
  - [6.1 Build Dataset Body JSON](#61-build-dataset-body-json)
  - [6.2 Publish Input Dataset](#62-publish-input-dataset)
  - [6.3 Publish Output Dataset (Woodgrove only)](#63-publish-output-dataset-woodgrove-only)
  - [6.4 Prepare CPK Keys (CPK mode only)](#64-prepare-cpk-keys-cpk-mode-only)
- [Step 07: Publish Query](#step-07-publish-query) `[WOODGROVE]`
- [Step 08: Approve Query](#step-08-approve-query) `[EACH COLLABORATOR]`
- [Step 09: Execute Query](#step-09-execute-query) `[WOODGROVE]`
- [Step 10: Monitor Query](#step-10-monitor-query) `[ANY]`
- [Step 11: Results & Audit](#step-11-results--audit) `[WOODGROVE]`
- [Step 12: Grafana Dashboards](#step-12-grafana-dashboards) `[OWNER]`
- [Appendix A: Federated Credential Subject Reference](#appendix-a-federated-credential-subject-reference)
- [Appendix B: Troubleshooting](#appendix-b-troubleshooting)
- [Appendix C: CPK Deep Dive](#appendix-c-cpk-deep-dive)
- [Appendix D: Dataset Schema Reference](#appendix-d-dataset-schema-reference)
- [Appendix E: Query Structure Reference](#appendix-e-query-structure-reference)
- [Appendix F: Collaboration Management](#appendix-f-collaboration-management)
- [Appendix: App-Based Authentication (SPN)](#appendix-app-based-authentication-spn)

---

## Step 01: Prerequisites `[ALL]`

### 1.1 Requirements

| Requirement | Details |
|---|---|
| Azure CLI | 2.75.0+ |
| `managedcleanroom` extension | `az extension add --name managedcleanroom --version 1.0.0b9 --upgrade` |
| PowerShell | 7.x+ |
| MSAL.PS module | `Install-Module MSAL.PS -Scope CurrentUser -Force` |
| azcopy | v10+ (CPK mode only) |
| kubectl | Latest stable |

#### IMPORTANT: Capacity planning for query execution

Complete capacity planning in `$resourceLocation` before creating the
collaboration. The AKS SKU and node-pool size are creation-time settings and
cannot be updated later.

**1. Select a query scale SKU based on input data size.**

| Input data processed by the query | `scaleSku` |
|---|---|
| Less than 300 GB | `small` |
| 300–600 GB | `medium` |
| More than 600 GB | `large` |

> **Preview:** `large` is not ready for production use. Validate it with your
> workload before relying on it.

**2. Decide the required maximum parallel queries and select the collaboration
configuration.**

The service derives the internal scheduling capacity from `$aksSku` and
`$nodePoolSize`. Customers must configure both values on the collaboration at
creation time and ensure the required Ddsv5 quota is available in the
subscription in `$resourceLocation`. Select a configuration whose maximum
parallel queries meets your requirement for the `scaleSku` chosen in step 1:

| `aksSku` | `nodePoolSize` | Max `small` queries | Max `medium` queries | Max `large` queries | Required Ddsv5 quota |
|---|---:|---:|---:|---:|---:|
| `Standard_D4ds_v5` | 3 | 10 | 5 | 3 | 12 vCPUs |
| `Standard_D4ds_v5` | 4 | 15 | 8 | 5 | 16 vCPUs |
| `Standard_D8ds_v5` | 3 | 20 | 10 | 6 | 24 vCPUs |

\* Higher configurations are possible but are not yet validated or supported.

**3. Calculate the Confidential ACI quota required for query execution.**

Let `N` be the maximum number of parallel queries you plan to run. Ensure the
subscription has the following Confidential ACI quota in `$resourceLocation`:

| `scaleSku` | Pods per query | Confidential vCPUs per query | Required Confidential vCPU quota | Required Confidential Container Groups quota |
|---|---:|---:|---:|---:|
| `small` | 6 | 34 | `34 × N` | `6 × N` |
| `medium` | 11 | 64 | `64 × N` | `11 × N` |
| `large` | 21 | 124 | `124 × N` | `21 × N` |

Actual runnable parallelism is the lower of the collaboration capacity above and
the available Confidential ACI quota. Leave headroom for retries,
terminating pods, other subscription workloads, and regional Confidential ACI
availability; these values are planning ceilings, not throughput guarantees.

> [!NOTE]
> Upcoming scheduler updates will raise query scheduling density.

The examples below target the **production RP and frontend** in `westus`, with
collaboration resources in `westus`. Choose a supported resource region where your
subscription has sufficient quota. Extension `1.0.0b9` uses
`2026-09-30-preview` for collaboration creation; the frontend API version is
separate and remains `2026-03-01-preview`.

### 1.2 Terminal T1 (Owner) — Variables

```powershell
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
az login
$account = az account show -o json | ConvertFrom-Json
$subscription = $account.id
$tenantId = $account.tenantId

$rpLocation = "westus"
$resourceLocation = "westus"   # Location where AKS, Container Groups, and all required resources are created
$aksSku = "Standard_D4ds_v5"
$nodePoolSize = 3
# Supported resourceLocation values:
# centralindia, eastasia, eastus, eastus2, germanywestcentral, italynorth,
# japaneast, northeurope, southcentralus, southeastasia, switzerlandnorth,
# uaenorth, westeurope, westus, westus2
$collabName = "<collaboration-name>"
$collabRg = "<collaboration-resource-group>"
```

### 1.3 One-Time Owner Setup

Register the resource provider (only needed once per subscription):

```powershell
az provider register --namespace Microsoft.CleanRoom
az provider register --namespace Microsoft.ContainerService
```

### 1.4 Each Collaborator Terminal — Variables

```powershell
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
az login
$account = az account show -o json | ConvertFrom-Json
$subscription = $account.id
$tenantId = $account.tenantId

$location = "westus"
$EncryptionMode = "SSE"    # "SSE" or "CPK"
$iteration = 0

$persona = "woodgrove"                # "woodgrove" or "northwind"
$personaRg = "cr-e2e-$persona-rg"
$personaEmail = "<your-email>"

az group create --name $personaRg --location $location -o none 2>$null

$frontend = "https://prod-nonattested.workload-frontendwestus.cleanroom.cloudapp.azure.net"
$oidcStorageUrl = "https://cleanroomoidc.z22.web.core.windows.net"   # Required for tenants where Federated Identity Credentials with MI are blocked by policy; specify a whitelisted pre-provisioned storage account name. For other tenants, leave blank ("") and a new storage account will be provisioned by the scripts.
```

> The `-nonattested` frontend uses standard, CA-issued TLS. Keep certificate
> verification enabled. TLS terminates outside the TEE at this endpoint, so it
> does not provide the attestation guarantee of the primary frontend endpoint.
> If you want TLS termination inside the TEE, set `$frontend` to the original
> production frontend endpoint:
> `https://prod.workload-frontendwestus.cleanroom.cloudapp.azure.net`.

### 1.5 Acquire Token, Extract OID & Configure CLI `[EACH COLLABORATOR]`

#### 1.5.1 Acquire Token

**Option A — MSAL device-code flow** (external / MSA accounts):

```powershell
$token = Get-MsalToken -ClientId "8a3849c1-81c5-4d62-b83e-3bb2bb11251a" `
    -TenantId "common" -Scopes "User.Read" -DeviceCode
$personaTokenFile = Join-Path ([System.IO.Path]::GetTempPath()) "msal-idtoken-$persona.txt"
$token.IdToken | Out-File -FilePath $personaTokenFile -NoNewline
```

**Option B — `az login`** (corporate @microsoft.com accounts):

```powershell
az login
$personaTokenFile = Join-Path ([System.IO.Path]::GetTempPath()) "msal-idtoken-$persona.txt"
az account get-access-token --resource "https://management.azure.com/" --query accessToken -o tsv | Out-File -FilePath $personaTokenFile -NoNewline
```

#### 1.5.2 Extract OID from Token

```powershell
$tokenB64 = (Get-Content $personaTokenFile -Raw).Split('.')[1].Replace('-', '+').Replace('_', '/')
$padLen = (4 - $tokenB64.Length % 4) % 4
$padded = $tokenB64 + ('=' * $padLen)
$claims = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($padded)) | ConvertFrom-Json
$personaOid = $claims.oid
Write-Host "JWT oid: $personaOid"
```

> **CRITICAL**: Always use the JWT `oid`, NOT `az ad signed-in-user show --query id`.
> For MSA accounts these differ. See [Appendix A](#appendix-a-federated-credential-subject-reference).

#### 1.5.3 Configure CLI Extension

```powershell
$env:MANAGEDCLEANROOM_ACCESS_TOKEN = Get-Content $personaTokenFile -Raw
$env:AZURE_CLI_DISABLE_CONNECTION_VERIFICATION = $null
az managedcleanroom frontend configure --endpoint $frontend
```

> Run `frontend configure` explicitly when changing environments. Do not rely
> solely on `MANAGEDCLEANROOM_ENDPOINT`: the `1.0.0b9` frontend client reads the
> endpoint from Azure CLI configuration.

---

## Step 02: Create Collaboration `[OWNER]`

> **Terminal: T1 (Owner)**

### 2.1 Create Resource Group

```powershell
az group create --name $collabRg --location $rpLocation -o none
```

### 2.2 Create Collaboration

```powershell
$collaboratorEmail = "<woodgrove-email>"
az managedcleanroom collaboration create `
    --collaboration-name $collabName `
    --resource-group $collabRg `
    --location $rpLocation `
    --resource-location $resourceLocation `
    --target-resource-configuration "{aks-sku:$aksSku,node-pool-size:$nodePoolSize}" `
    --collaborators "[{user-identifier:'$collaboratorEmail'}]" `
    --no-wait
```

> The `--collaborators` flag adds collaborators at creation time itself.
> To add more collaborators later, see [Step 2.4](#24-add-more-collaborators-optional).

> **NOTE**: `--location` is the ARM RP location (`$rpLocation`). `--resource-location` controls where
> actual resources (AKS cluster, CACI instances) are deployed — set via `$resourceLocation`.

The `--target-resource-configuration` argument (alias `--target-config`) controls
the AKS node pool at creation time:

| Option | Supported values | Default |
|---|---|---|
| `aks-sku` | `Standard_D4ds_v5`, `Standard_D8ds_v5` | `Standard_D4ds_v5` |
| `node-pool-size` | Integer from `3` through `10` | `3` |

Both `aks-sku` and `node-pool-size` are optional. When omitted, they default to
`Standard_D4ds_v5` and `3`, respectively. These are creation-time settings, not
collaboration-update arguments.

#### Optional public IP tagging

The same target configuration can also accept `i-p-tag-configuration`, with both
`type` and `value`, to tag public IP resources created for the collaboration.
These are networking IP tags, not ordinary ARM resource `--tags`.

| Type | Example value | Intended use |
|---|---|---|
| `FirstPartyUsage` | `/AzureCleanRoomsProd` | Approved Microsoft first-party production usage |
| `FirstPartyUsage` | `/AzureCleanRoomsNonProd` | Approved Microsoft first-party non-production usage |

Use only IP-tag values approved for your subscription and scenario. Specify IP
tags in `az managedcleanroom collaboration create` with this argument:

```text
    --target-resource-configuration "{i-p-tag-configuration:{type:FirstPartyUsage,value:/AzureCleanRoomsProd}}" `
```

**Runtime**: ~25 minutes. Poll `provisioningState` until `Succeeded`:

```powershell
do {
    $collab = az managedcleanroom collaboration show `
        --collaboration-name $collabName `
        --resource-group $collabRg -o json | ConvertFrom-Json
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] provisioningState: $($collab.provisioningState)"
    Start-Sleep -Seconds 60
} while ($collab.provisioningState -notin @("Succeeded", "Failed"))
```

### 2.3 Enable Analytics Workload

```powershell
az managedcleanroom collaboration enable-workload `
    --collaboration-name $collabName `
    --resource-group $collabRg `
    --workload-type Analytics `
    --no-wait
```

**Runtime**: ~7 minutes. Poll until the workload endpoint is populated:

```powershell
do {
    $collab = az managedcleanroom collaboration show `
        --collaboration-name $collabName `
        --resource-group $collabRg -o json | ConvertFrom-Json
    $wl = $collab.workloads | Where-Object { $_.workloadType -eq "Analytics" }
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] provisioningState: $($collab.provisioningState) | workload endpoint: $($wl.endpoint)"
    Start-Sleep -Seconds 30
} while (-not $wl.endpoint -and $collab.provisioningState -ne "Failed")
```

Then wait for `healthState` to become `Ok`:

```powershell
do {
    $collab = az managedcleanroom collaboration show `
        --collaboration-name $collabName `
        --resource-group $collabRg -o json | ConvertFrom-Json
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] healthState: $($collab.health.healthState)"
    if ($collab.health.healthState -ne "Ok" -and $collab.health.healthIssues) {
        $collab.health.healthIssues | ForEach-Object { Write-Host "  Issue: $($_ | ConvertTo-Json -Compress)" }
    }
    Start-Sleep -Seconds 30
} while ($collab.health.healthState -ne "Ok")
```

### 2.4 Add More Collaborators (Optional)

> The owner was already added as a collaborator during `create` (Step 2.2).
> Use this step to invite additional collaborators (e.g. Northwind in a multi-party scenario).

> To add Service Principals (SPNs) instead of user email IDs for automation, see
> [Appendix: App-Based Authentication (SPN)](#appendix-app-based-authentication-spn).

```powershell
# Add Northwind
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

$collabs = (az managedcleanroom frontend collaboration list -o json | ConvertFrom-Json).collaborations
$collabs | Format-Table @{L='#';E={[array]::IndexOf($collabs,$_)+1}}, collaborationName, collaborationId, userStatus

$choice = Read-Host "Enter the number of your collaboration"
$collabId = $collabs[[int]$choice - 1].collaborationId
Write-Host "Selected: $collabId"
```

### 3.2 Accept Invitation

```powershell
$invitations = (az managedcleanroom frontend invitation list `
    --collaboration-id $collabId --pending-only -o json | ConvertFrom-Json).invitations
$invitations | Format-Table invitationId, accountType, status

if ($invitations) {
    $invitationId = $invitations[0].invitationId
    az managedcleanroom frontend invitation accept `
        --collaboration-id $collabId `
        --invitation-id $invitationId
} elseif (($collabs | Where-Object collaborationId -eq $collabId).userStatus -ne "Active") {
    throw "No pending invitation and collaborator is not Active."
}
```

List the active frontend collaborators:

```powershell
az managedcleanroom frontend collaborator list `
    --collaboration-id $collabId -o table
```

---

## Step 04: Provision Resources & Upload Data `[EACH COLLABORATOR]`

> Run Steps 04-06 in **each collaborator terminal**. Commands are identical —
> only `$persona` differs. In multi-collaborator mode, Woodgrove and
> Northwind run these steps **in parallel** (independent resource groups).

### 4.1 Prepare Resources

```powershell
./scripts/04-prepare-resources.ps1 -resourceGroup $personaRg -persona $persona -location $location
```

> This script provisions a storage account, Key Vault (premium), and managed identity.
> It also assigns RBAC roles to the caller:
> - **Storage Blob Data Contributor** on the storage account (required to upload data)
> - **Key Vault Crypto Officer** and **Key Vault Secrets Officer** on the Key Vault (required for CPK mode)

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

Use a distinct suffix for each dataset iteration. The scripts preserve
datastore metadata by suffix so earlier outputs can still be downloaded.

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
if ($oidcStorageUrl) { $oidcParams["OidcStorageUrl"] = $oidcStorageUrl }

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
> the JWT `oid` from Step 1.5.2.

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
if ($persona -eq "woodgrove") {
    # Scope Woodgrove's input dataset to a subfolder inside its container.
    ./scripts/08-build-dataset-body.ps1 -resourceGroup $personaRg -persona $persona `
        -subdirectory "2025-09-01"
} else {
    # Northwind's input dataset maps to the entire container.
    ./scripts/08-build-dataset-body.ps1 -resourceGroup $personaRg -persona $persona
}
```

> [!IMPORTANT]
> The Woodgrove branch above passes `-subdirectory "2025-09-01"` so its input
> dataset is scoped to a single date folder inside the container. Northwind's
> input dataset is left at the container root and sees all four days produced
> by `generate-data.ps1`. Omit `-subdirectory` to use the entire container.
> For the full parameter reference, see
> [Optional dataset parameters](#optional-dataset-parameters) in Appendix D.

> **Bring your own data**: If you want to provide your own datasets, upload your data directly to the
> storage accounts created for your persona and update `datasetSchema` and `datasetAccessPolicy` in the dataset
> body files: `generated/publish/$persona-input-dataset.json` and `generated/publish/$persona-output-dataset.json`.

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
>
> Requires **Key Vault Crypto Officer** and **Key Vault Secrets Officer** roles
> (assigned by `04-prepare-resources.ps1` in Step 4.1).

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
>     Select-Object -ExpandProperty value |
>     Where-Object { $_.id -match "northwind" } |
>     ForEach-Object { Write-Host $_.id }
> ```

```powershell
$northwindDataset = "<northwind-input-csv-suffix>"   # e.g., "northwind-input-csv-v1"
$queryName = "query2$suffix"   # Update queryName for multi-collaborator
./scripts/09-build-query-body.ps1 -queryName $queryName `
    -queryDir "./demos/query/woodgrove/query2" `
    -publisherInputDataset $northwindDataset `
    -consumerInputDataset "woodgrove-input-csv$suffix" `
    -outputDataset "woodgrove-output-csv$suffix"
```

> **Bring your own query**: If you want to use a custom query, update `generated/publish/$queryName.json` with your required query segments before publishing.

### 7.2 Publish Query

```powershell
az managedcleanroom frontend analytics query publish `
    --collaboration-id $collabId `
    --document-id $queryName `
    --body "@generated/publish/$queryName.json"
```

---

## Step 08: Approve Query `[EACH COLLABORATOR]`

> **Single-collaborator**: Publishing casts Woodgrove's accept vote; no separate vote is required.
>
> **Multi-collaborator**: Only the remaining affected collaborators must vote. Northwind needs the
> `$queryName` from Woodgrove (or list queries to find it).

Each collaborator runs in their own terminal:

```powershell
# View query and get proposal ID
$queryInfo = az managedcleanroom frontend analytics query show `
    --collaboration-id $collabId `
    --document-id $queryName -o json | ConvertFrom-Json
$queryInfo.data.queryData | Format-Table executionSequence, preConditions, postFilters, data -Wrap
$proposalId = $queryInfo.proposalId
Write-Host "Proposal ID: $proposalId"

# Vote
if ($persona -ne "woodgrove" -and $queryInfo.state -ne "Accepted") {
    az managedcleanroom frontend analytics query vote `
        --collaboration-id $collabId `
        --document-id $queryName `
        --vote-action accept `
        --proposal-id $proposalId
}
```

> **Northwind**: If you don't have `$queryName`, list published queries and set it:
> ```powershell
> az managedcleanroom frontend analytics query list `
>     --collaboration-id $collabId -o json
>
> $queryName = "<query-name-from-list>"   # e.g., "query2-v1"
> ```

**Verify**: Query state should be `"Accepted"` after all required votes.

```powershell
az managedcleanroom frontend analytics query show `
    --collaboration-id $collabId `
    --document-id $queryName --query state -o tsv
```

---

## Step 09: Execute Query `[WOODGROVE]`

Set `scaleSku` in the run request to `small`, `medium`, or `large`. The service
defaults to `small` when omitted. This query setting is separate from the AKS
VM SKU and node count selected when creating the collaboration.

Extension `1.0.0b9` does not expose a `--scale-sku` flag. Use a JSON request body:

```powershell
$scaleSku = "small"
$runBody = @{ scaleSku = $scaleSku }
[System.IO.File]::WriteAllText("$PWD/generated/run-config.json", ($runBody | ConvertTo-Json))

$runResult = az managedcleanroom frontend analytics query run `
    --collaboration-id $collabId `
    --document-id $queryName `
    --body "@generated/run-config.json" -o json | ConvertFrom-Json

$jobId = $runResult.id
Write-Host "Job ID: $jobId"
```

Do not combine `--body` with `--start-date`, `--end-date`, `--dry-run`, or
`--use-optimizer`; include any of these settings in the JSON body instead.

> The CLI auto-generates a run ID. Each invocation starts a new execution.
> `"status": "success"` means accepted for scheduling, not completed. Takes 10-20 min.

### Scale SKU configuration

The current Spark frontend profiles map the run's `scaleSku` as follows:

| `scaleSku` | Driver memory | Memory per executor | Maximum executors | Input data guidance |
|---|---|---|---:|---|
| `small` (default) | `4g` | `8g` | 5 | Less than 300 GB |
| `medium` | `8g` | `16g` | 10 | 300–600 GB |
| `large` (preview) | `12g` | `24g` | 20 | More than 600 GB |

`large` is not ready for production use. See
[IMPORTANT: Capacity planning for query execution](#important-capacity-planning-for-query-execution)
before choosing a scale SKU or collaboration size.

Memory values use Spark's notation. Driver cores, executor cores, and minimum
executors are inherited from the deployment configuration, not set by these
profiles. The chart defaults are **1 driver core, 1 core per executor, and
1 minimum executor**; deployments can override them. Maximum executors is a
scaling limit, not a fixed count. Spark container cores do not represent the
Confidential ACI group quota used in the capacity calculation above. Account for
memory overhead and available regional capacity in addition to CPU quota.

To inspect the effective settings without executing the query, send
`dryRun: true` in the same request body and inspect `skuSettings` in the response.
Omit `dryRun` (or set it to `false`) for actual execution.

> **Network connectivity**: This step requires the ACCR Frontend Service to reach the Analytics Endpoint of the Collaboration. It can time out due to tenant-specific network configurations:
>
> 1. **NSG (Network Security Group)**: If your tenant has NSGs blocking inbound internet access to the AKS Analytics endpoint on port 443, the query will fail. Contact the ACCR team with the `tenantId` of the collaboration so we can whitelist your tenant — an NSG rule will be updated to allow port 443 access to the AKS cluster.
> 2. **[AVNM (Azure Virtual Network Manager)](https://learn.microsoft.com/en-us/azure/virtual-network-manager/)**: This is a tenant-level policy. Your tenant admin needs to create an AVNM rule to allow port 443 access from the internet by following the documentation linked above.

**Optional: cancel a run.** After submitting a query, use the following command
only if you want to cancel it before completion:

```powershell
az managedcleanroom frontend analytics query cancel-run `
    --collaboration-id $collabId `
    --document-id $queryName `
    --run-id $jobId
```

> **Date-range filtering**: To read datasets within a specific date range,
> include `startDate` and `endDate` alongside `scaleSku` in the request body:
>
> ```powershell
> $runBody = @{
>     scaleSku = $scaleSku
>     startDate = "2025-09-01"
>     endDate = "2025-09-02"
> }
> [System.IO.File]::WriteAllText("$PWD/generated/run-config.json", ($runBody | ConvertTo-Json))
> $runResult = az managedcleanroom frontend analytics query run `
>     --collaboration-id $collabId `
>     --document-id $queryName `
>     --body "@generated/run-config.json" -o json | ConvertFrom-Json
> $jobId = $runResult.id
> ```

---

## Step 10: Monitor Query `[ANY]`

```powershell
do {
    $result = az managedcleanroom frontend analytics query runresult show `
        --collaboration-id $collabId `
        --job-id $jobId -o json | ConvertFrom-Json
    $state = $result.status.applicationState.state
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] State: $state"
    Start-Sleep -Seconds 30
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

> **Query fails or times out?** If the query stays in `SUBMITTED` or `RUNNING` for
> an extended period, or transitions to `FAILED`/`SUBMISSION_FAILED`, check the
> collaboration health for pod-level or capacity issues:
>
> ```powershell
> az managedcleanroom collaboration show `
>     --collaboration-name $collabName `
>     --resource-group $collabRg `
>     --query "health"
> ```
>
> If `healthState` is `Error`, the `healthIssues` array will list specific pod
> failures — such as CACI capacity shortages in the region (e.g.,
> `FailedCreatePodSandBox: resource not available`), executor pods stuck in init,
> or container crashes. These issues indicate infrastructure-level problems that
> prevent Spark executors from starting.

---

## Step 11: Results & Audit `[WOODGROVE]`

### 11.1 Run History

```powershell
az managedcleanroom frontend analytics query runhistory list `
    --collaboration-id $collabId `
    --document-id $queryName -o json
```

> The output includes execution stats such as **total rows read**, **total rows written**, and **duration** of the query.

### 11.2 Audit Events

```powershell
az managedcleanroom frontend analytics auditevent list `
    --collaboration-id $collabId -o json
```

### 11.3 Download Output

Auto-detects SSE/CPK mode from suffix-specific output metadata and uses its
container and, for CPK, local DEK file. In both modes, `-JobId` selects the exact
run directory under `Analytics/<date>/<run-id>/` and downloads all its CSV
partitions. Blob paths are preserved under the output directory.

```powershell
./scripts/11-download-output.ps1 -resourceGroup $personaRg `
    -datasetSuffix "$suffix" -JobId $jobId `
    -OutputDir "./generated/output/$EncryptionMode-$jobId"
```

> Use a distinct `-OutputDir` for each run. Without `-JobId`, the
> script downloads all matching CSV outputs in the selected container.

---

## Step 12: Grafana Dashboards `[OWNER]`

> Grafana dashboards let the owner monitor Spark query execution,
> resource usage, and logs in real time.

### 12.1 Get Readonly Kubeconfig

```powershell
$kc = az managedcleanroom collaboration get-readonly-kube-config `
    --collaboration-name $collabName `
    --resource-group $collabRg -o json | ConvertFrom-Json

$bytes = [Convert]::FromBase64String($kc.kubeconfig)
[System.Text.Encoding]::UTF8.GetString($bytes) |
    Out-File "./readonly.kubeconfig" -Encoding utf8
```

### 12.2 Open Grafana Dashboard

Uses the read-only kubeconfig to access diagnostics through Grafana, prints the
URL to open manually in your browser, and port-forwards until you press Ctrl+C.

```powershell
./scripts/12-open-grafana-dashboard.ps1 -KubeConfigPath "./readonly.kubeconfig"
```

Login with `admin` and the password printed by the script.

---

## Appendix A: Federated Credential Subject Reference

Format: `{contractId}-{ownerId}` where `contractId` = `"Analytics"` (capital A)
and `ownerId` = JWT `oid` from Step 1.5.2.

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
| `SSL certificate verify failed` | Wrong frontend endpoint or stale endpoint configuration | Re-run `az managedcleanroom frontend configure --endpoint $frontend`; do not disable TLS verification |
| `404 Not Found` on frontend | Using ARM ID instead of frontend UUID | Use UUID from `frontend collaboration list` |
| `ContractNotFound` | Stale CCF endpoint | Create new collaboration |
| `Python 3.13 tuple error` | CLI extension bug | Upgrade to `managedcleanroom` extension `1.0.0b9` |
| `Already voted / Conflict` | Publisher already voted | Check query state; skip an accept vote if already Accepted. Do not ignore other conflicts |
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
| **Woodgrove input** | `user_id` (string), `hashed_email` (string), `purchase_history` (string) | `user_id`, `hashed_email`, `purchase_history` |
| **Woodgrove output** | `user_id` (string) | `user_id` |

Fields not in `allowedFields` are excluded from query access — prevents PII exposure.
Supported formats: `csv`, `parquet`, `json`.

### Optional dataset parameters

| Parameter | Description |
|---|---|
| `subdirectory` | Prefix inside the dataset's container to scope the dataset to a subfolder (e.g. `2025-09-01`). Optional, defaults to `""` (entire container). Pass it via the `-subdirectory` parameter of `scripts/08-build-dataset-body.ps1` - see [Step 6.1](#61-build-dataset-body-json). |

---

## Appendix E: Query Structure Reference

| Section | Purpose |
|---|---|
| `queryData[]` | SQL segments with integer `executionSequence` and string `data`, `preConditions`, `postFilters` |
| `inputDatasets` | Comma-separated string of `datasetDocumentId:viewName` bindings |
| `outputDataset` | String: `datasetDocumentId:output` |

**Privacy controls**:

- **Pre-conditions** enforce a minimum row count per view. If any view has fewer rows than `minRowCount`, the query aborts.
- **Post-filters** remove groups from the output whose aggregation count is below a threshold, preventing identification of individuals.

Both are defined in the query segments. Edit the thresholds before publishing the query (Step 07).

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
$personaTokenFile = Join-Path ([System.IO.Path]::GetTempPath()) "msal-idtoken-$persona.txt"
$token | Out-File -FilePath $personaTokenFile -NoNewline
$env:CLEANROOM_FRONTEND_TOKEN = $token
$env:MANAGEDCLEANROOM_ACCESS_TOKEN = $token
az managedcleanroom frontend configure --endpoint $frontend
```

Use this instead of interactive token acquisition in Step 1.5.1, then extract
the token's `oid` as shown in Step 1.5.2.

### Add SPN as Collaborator

```powershell
az managedcleanroom collaboration add-collaborator `
    --collaboration-name $collabName --resource-group $collabRg `
    --user-identifier "<clientAppId>" `
    --object-id "<spObjectId>" `
    --tenant-id "<tenantId>"
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
