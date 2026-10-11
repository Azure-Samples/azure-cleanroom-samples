# Big Data Analytics — SDK CLI (`cleanroom-mgmt` + `afe`)

This guide uses the **`cleanroom-mgmt` SDK CLI** (built from
`packages/mgmt-sample/Program.cs`, wrapping the generated
`Azure.ResourceManager.CleanRoom` management-plane SDK) for ARM collaboration
operations, and the **`afe` SDK CLI** (built from `packages/sample/Program.cs`,
wrapping the generated `AnalyticsFrontendAPI` SDK) for all frontend service
operations. The same helper scripts are used for Azure resource provisioning.

The **`cleanroom-mgmt`** CLI uses **`az login` (DefaultAzureCredential)** only —
ARM management-plane operations rely on the Azure CLI / managed-identity / SPN
chain. The **`afe`** CLI (frontend) supports both `az login` and MSAL
device-code (`--use-msal`) for accounts that need it.

For the overall sample and entry points, see [README.md](README.md).
For the `cleanroom-mgmt` + `afe` flow, continue with this guide.

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
| **API mode** | `cleanroom-mgmt` SDK CLI (ARM) + `afe` SDK CLI (frontend) |
| **Auth** | `cleanroom-mgmt`: `az login` only · `afe`: `az login` **or** MSAL device-code (`--use-msal`) |
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
    - [1.2 Prepare the SDK CLI Command](#12-prepare-the-sdk-cli-command)
  - [1.3 Terminal T1 (Owner) — Variables](#13-terminal-t1-owner--variables)
  - [1.4 One-Time Owner Setup](#14-one-time-owner-setup)
  - [1.5 Each Collaborator Terminal — Variables & Auth](#15-each-collaborator-terminal--variables--auth) `[EACH COLLABORATOR]`
  - [1.6 Extract OID for Federated Credentials](#16-extract-oid-for-federated-credentials) `[EACH COLLABORATOR]`
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
- [Step 12: Grafana Dashboards](#step-12-grafana-dashboards) `[OWNER]`
- [Appendix A: Federated Credential Subject Reference](#appendix-a-federated-credential-subject-reference)
- [Appendix B: Troubleshooting](#appendix-b-troubleshooting)
- [Appendix C: CPK Deep Dive](#appendix-c-cpk-deep-dive)
- [Appendix D: Dataset Schema Reference](#appendix-d-dataset-schema-reference)
- [Appendix E: Query Structure Reference](#appendix-e-query-structure-reference)
- [Appendix F: SDK CLI Verb Reference](#appendix-f-sdk-cli-verb-reference)
- [Appendix G: Collaboration Management](#appendix-g-collaboration-management)
- [Appendix: App-Based Authentication (SPN)](#appendix-app-based-authentication-spn)

---

## Step 01: Prerequisites `[ALL]`

### 1.1 Requirements

| Requirement | Details |
|---|---|
| Azure CLI | 2.75.0+ |
| PowerShell | 7.x+ |
| .NET SDK | 10.0+ (project targets `net10.0`) |
| MSAL.PS module | `Install-Module MSAL.PS -Scope CurrentUser -Force` (only for OID extraction in MSAL flow) |
| azcopy | v10+ (CPK mode only) |

> **Quota check:** This sample deploys an AKS cluster and Confidential ACI
> container groups in the `$resourceLocation` region (**West US** by default). Ensure your subscription has the
> following minimum quota in that region before proceeding:
>
> | Resource | Minimum vCPUs | SKU / Family |
> |---|---|---|
> | AKS node pool | 8 | Standard_D4ds_v5 (Ddsv5 family) |
> | Confidential ACI | 6 | Confidential container groups |
>
> The above covers a single query execution (1 Spark driver + up to 3
> executors, each using 1 vCPU). Spark pods are provisioned at runtime and
> removed after query execution completes. Multiple queries can run
> concurrently — add 4 vCPUs of Confidential ACI quota per additional concurrent query.

> The `managedcleanroom` CLI extension is **not required** for this guide.

### 1.2 Prepare the SDK CLI Commands

Two CLIs are used in this guide:

- `cleanroom-mgmt` — ARM management-plane operations
  (`packages/mgmt-sample/Program.cs`).
- `afe` — frontend / data-plane operations
  (`packages/sample/Program.cs`).

Run them via `dotnet run` directly from each project:

```powershell
# Run this block in every terminal that will call the SDK CLIs.
# If your terminal starts in C:\Users\...\Downloads\sample, move into the repo first.

dotnet restore .\packages\mgmt-sample\CleanRoomMgmtSample.csproj
dotnet restore .\packages\sample\AnalyticsFrontendSample.csproj

# Verify
dotnet run --project .\packages\mgmt-sample\CleanRoomMgmtSample.csproj -- --help
dotnet run --project .\packages\sample\AnalyticsFrontendSample.csproj -- --help
```

> The `afe` CLI supports both auth modes — add `--use-msal` for MSAL
> device-code, omit it to use `DefaultAzureCredential` (which picks up
> `az login`, environment variables, managed identity, etc.).
>
> The `cleanroom-mgmt` CLI uses `DefaultAzureCredential` only — sign in with
> `az login` (or use the standard SPN / managed-identity env vars).

### 1.3 Terminal T1 (Owner) — Variables

```powershell
az login
$account = az account show -o json | ConvertFrom-Json
$subscription = $account.id
$tenantId = $account.tenantId

$rpLocation = "westus"
$resourceLocation = "westus"   # Location where AKS, Container Groups, and all required resources are created
# Supported resourceLocation values:
# centralindia, eastasia, eastus, eastus2, germanywestcentral, italynorth,
# japaneast, northeurope, southcentralus, southeastasia, switzerlandnorth,
# uaenorth, westeurope, westus, westus2
$collabName = "<collaboration-name>"
$collabRg = "<collaboration-resource-group>"

# Project paths
$mgmtProject = ".\packages\mgmt-sample\CleanRoomMgmtSample.csproj"
$afeProject  = ".\packages\sample\AnalyticsFrontendSample.csproj"

# Make subscription / resource group available to cleanroom-mgmt
# so we don't have to pass them on every invocation.
$env:AZURE_SUBSCRIPTION_ID = $subscription
$env:AZURE_RESOURCE_GROUP  = $collabRg
```

> Add `--use-msal` to any `afe` command below if you need MSAL device-code
> auth. The `cleanroom-mgmt` CLI does not accept `--use-msal` — it always
> uses `az login` / `DefaultAzureCredential`.

### 1.4 One-Time Owner Setup

Register the resource provider (only needed once per subscription):

```powershell
az provider register --namespace Microsoft.CleanRoom
```

### 1.5 Each Collaborator Terminal — Variables & Auth

```powershell
$location = "westus"
$EncryptionMode = "SSE"    # "SSE" or "CPK"
$iteration = 0

$persona = "woodgrove"                # "woodgrove" or "northwind"
$personaRg = "cr-e2e-$persona-rg"

az group create --name $personaRg --location $location -o none 2>$null

# --- Frontend SDK CLI configuration ----------------------------------------
$frontend = "https://prod.workload-frontendwestus.cleanroom.cloudapp.azure.net"
$oidcStorageUrl = "https://cleanroomoidc.z22.web.core.windows.net"   # Required for tenants where Federated Identity Credentials with MI are blocked by policy. Leave blank ("") otherwise.

# Pick auth mode:
#   $UseMsal = $true   -> MSAL device-code (external / MSA accounts)
#   $UseMsal = $false  -> az login / DefaultAzureCredential (corporate accounts)
$UseMsal = $false

# Common environment for the SDK CLI
$env:ANALYTICS_FRONTEND_ENDPOINT = $frontend
$env:PERSONA = $persona
$afeProject = "./packages/sample/AnalyticsFrontendSample.csproj"

$afeCommon = @("--insecure")
if ($UseMsal) { $afeCommon += "--use-msal" }

$parseAfeJson = {
    param([string[]]$Raw)
    (($Raw | Select-Object -Skip 1) -join [Environment]::NewLine | ConvertFrom-Json -Depth 50)
}

if ($UseMsal) {
    $env:AZURE_CLIENT_ID = "8a3849c1-81c5-4d62-b83e-3bb2bb11251a"
    $env:AZURE_TENANT_ID = "common"
} else {
    az login | Out-Null
}
```

> **First MSAL call**: on the first `dotnet run` CLI call with `--use-msal`,
> the CLI prints a device-code prompt — visit the URL, enter the code, and
> sign in. Subsequent calls reuse the cached token silently. The CLI also
> persists the ID token to `${env:TEMP}\msal-idtoken-$persona.txt` for OID
> extraction (see [Step 1.6](#16-extract-oid-for-federated-credentials)).

### 1.6 Extract OID for Federated Credentials

The federated credential subject (Step 05) needs the JWT `oid` claim of the
calling principal.

**MSAL flow** — make any `dotnet run` CLI call first so the CLI persists the ID
token, then decode it:

```powershell
dotnet run --project $afeProject -- @afeCommon collaborations list | Out-Null   # forces token acquisition + persist

$personaTokenFile = Join-Path $env:TEMP "msal-idtoken-$persona.txt"
$tokenB64 = ((Get-Content $personaTokenFile -Raw).Trim()).Split('.')[1]
$tokenB64 = $tokenB64.Replace('-', '+').Replace('_', '/')
$padLen = (4 - $tokenB64.Length % 4) % 4
$claims = [System.Text.Encoding]::UTF8.GetString(
    [Convert]::FromBase64String($tokenB64 + ('=' * $padLen))) | ConvertFrom-Json
$personaOid = $claims.oid
Write-Host "JWT oid: $personaOid"
```

**`az login` flow** — decode the ARM access token:

```powershell
$armToken = az account get-access-token --resource "https://management.azure.com/" --query accessToken -o tsv
$tokenB64 = ($armToken.Trim()).Split('.')[1]
$tokenB64 = $tokenB64.Replace('-', '+').Replace('_', '/')
$padLen = (4 - $tokenB64.Length % 4) % 4
$claims = [System.Text.Encoding]::UTF8.GetString(
    [Convert]::FromBase64String($tokenB64 + ('=' * $padLen))) | ConvertFrom-Json
$personaOid = $claims.oid
Write-Host "JWT oid: $personaOid"
```

> **CRITICAL**: Always use the JWT `oid`, NOT `az ad signed-in-user show --query id`.
> For MSA accounts these differ. See [Appendix A](#appendix-a-federated-credential-subject-reference).

---

## Step 02: Create Collaboration `[OWNER]`

### 2.1 Create Resource Group

```powershell
az group create --name $collabRg --location $rpLocation -o none
```

### 2.2 Create Collaboration

```powershell
$collaboratorEmail = "<woodgrove-email>"

dotnet run --project $mgmtProject -- collaborations create `
    $collabName `
    --location $rpLocation `
    --resource-location $resourceLocation `
    --collaborator $collaboratorEmail
```

> `--collaborator` may be repeated to add multiple collaborators at creation
> time. To add more collaborators later, see
> [Step 2.4](#24-add-more-collaborators-optional).

> **NOTE**: `--location` is the ARM RP location (`$rpLocation`).
> `--resource-location` controls where actual resources (AKS cluster, CACI
> instances) are deployed.

**Runtime**: ~25 minutes. The CLI awaits the long-running operation and
prints the final resource. To poll explicitly while it runs (separate
terminal):

```powershell
do {
    $collab = dotnet run --project $mgmtProject -- collaborations get $collabName | ConvertFrom-Json
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] provisioningState: $($collab.properties.provisioningState)"
    Start-Sleep -Seconds 60
} while ($collab.properties.provisioningState -notin @("Succeeded", "Failed"))
```

### 2.3 Enable Analytics Workload

```powershell
dotnet run --project $mgmtProject -- collaborations enable-workload `
    $collabName `
    --workload-type AnalyticsStrict
```

> The SDK enum value is `AnalyticsStrict` (the wire-string `"Analytics"` used
> by the raw ARM call maps to this member on the current SDK build).

**Runtime**: ~7 minutes. The CLI awaits the LRO and prints the refreshed
collaboration. To poll explicitly while it runs:

```powershell
do {
    $collab = dotnet run --project $mgmtProject -- collaborations get $collabName | ConvertFrom-Json
    $wl = $collab.properties.workloads | Where-Object { $_.workloadType -eq "AnalyticsStrict" -or $_.workloadType -eq "Analytics" }
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] provisioningState: $($collab.properties.provisioningState) | workload endpoint: $($wl.endpoint)"
    Start-Sleep -Seconds 30
} while (-not $wl.endpoint -and $collab.properties.provisioningState -ne "Failed")
```

Then wait for `healthState` to become `Ok`:

```powershell
do {
    $collab = dotnet run --project $mgmtProject -- collaborations get $collabName | ConvertFrom-Json
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] healthState: $($collab.properties.health.healthState)"
    if ($collab.properties.health.healthState -ne "Ok" -and $collab.properties.health.healthIssues) {
        $collab.properties.health.healthIssues | ForEach-Object { Write-Host "  Issue: $($_ | ConvertTo-Json -Compress)" }
    }
    Start-Sleep -Seconds 30
} while ($collab.properties.health.healthState -ne "Ok")
```

### 2.4 Add More Collaborators (Optional)

> The owner was already added as a collaborator during `create` (Step 2.2).
> Use this step to invite additional collaborators (e.g. Northwind in a multi-party scenario).

> To add Service Principals (SPNs) instead of user email IDs for automation, see
> [Appendix: App-Based Authentication (SPN)](#appendix-app-based-authentication-spn).

```powershell
# Add Northwind
$collaboratorEmail = "<northwind-email>"

dotnet run --project $mgmtProject -- collaborations add-collaborator `
    $collabName `
    --user $collaboratorEmail
```

> The CLI handles the request body shape internally — no `body.json` to
> manage and no encoding pitfalls.

**Verify**:
```powershell
dotnet run --project $mgmtProject -- collaborations get $collabName
```

---

## Step 03: Accept Invitations `[EACH COLLABORATOR]`

### 3.1 Get Collaboration UUID

```powershell
$raw = dotnet run --project $afeProject -- @afeCommon collaborations list
$collabs = (& $parseAfeJson $raw).collaborations
$collabs | Format-Table @{L='#';E={[array]::IndexOf($collabs,$_)+1}}, collaborationName, collaborationId, userStatus

$choice = Read-Host "Enter the number of your collaboration"
$collabId = $collabs[[int]$choice - 1].collaborationId
Write-Host "Selected: $collabId"
```

### 3.2 Accept Invitation

```powershell
$raw = dotnet run --project $afeProject -- @afeCommon invitations list $collabId
$invitations = (& $parseAfeJson $raw).invitations
$invitations | Format-Table invitationId, accountType, status

if (-not $invitations -or $invitations.Count -eq 0) {
    Write-Host "No invitations found for this collaboration (already accepted or none pending)."
}
else {
    $pending = @($invitations | Where-Object { $_.status -eq "Pending" })
    if ($pending.Count -eq 0) {
        Write-Host "No pending invitations to accept."
    }
    else {
        $invitationId = $pending[0].invitationId
        dotnet run --project $afeProject -- @afeCommon invitations accept $collabId $invitationId
    }
}
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
$jwksDir = "generated/$personaRg"
New-Item -ItemType Directory -Path $jwksDir -Force | Out-Null

$raw = dotnet run --project $afeProject -- @afeCommon oidc keys $collabId
$jwks = & $parseAfeJson $raw
$jwks | ConvertTo-Json -Depth 10 | Out-File "$jwksDir/jwks.json" -Encoding utf8
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

$issuerBody = @{ url = $issuerUrl } | ConvertTo-Json -Compress
dotnet run --project $afeProject -- @afeCommon oidc set-issuer-url $collabId --body $issuerBody
```

### 5.4 Grant Access & Create Federated Credentials

```powershell
./scripts/07-grant-access.ps1 -resourceGroup $personaRg `
    -collaborationId $collabId -contractId "Analytics" `
    -userId $personaOid -EncryptionMode $EncryptionMode
```

> **CRITICAL**: `contractId` must be `"Analytics"` (capital A). `-userId` must be
> the JWT `oid` from Step 1.6.

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
    # Scope Woodgrove's input dataset to a sub folder inside its container
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
> by `generate-data.ps1`. For the full parameter reference, see
> [Optional dataset parameters](#optional-dataset-parameters) in Appendix D.

> **Bring your own data**: If you want to provide your own datasets, upload your data directly to the
> storage accounts created for your persona and update the `schema` and `accessPolicy` in the dataset
> body files: `generated/publish/$persona-input-dataset.json` and `generated/publish/$persona-output-dataset.json`.

### 6.2 Publish Input Dataset

The CLI accepts request bodies via `--body @path/to/file.json`:

```powershell
$inputDoc  = "$persona-input-csv$suffix"
$inputFile = "generated/publish/$persona-input-dataset.json"

dotnet run --project $afeProject -- @afeCommon datasets publish $collabId $inputDoc --body "@$inputFile"
```

### 6.3 Publish Output Dataset (Woodgrove only)

```powershell
if ($persona -eq "woodgrove") {
    $outputDoc  = "woodgrove-output-csv$suffix"
    $outputFile = "generated/publish/woodgrove-output-dataset.json"

    dotnet run --project $afeProject -- @afeCommon datasets publish $collabId $outputDoc --body "@$outputFile"
}
```

> Execution consent is enabled by default at publish time. To revoke or re-enable later:
> ```powershell
> $consentBody = @{ consentAction = "disable" } | ConvertTo-Json -Compress   # or "enable"
> dotnet run --project $afeProject -- @afeCommon consent put $collabId "<document-name>" --body $consentBody
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
    # Fetch SKR policy via the SDK CLI and pass it to the script.
    $raw = dotnet run --project $afeProject -- @afeCommon analytics skr-policy $collabId "<kid>"
    $skrPolicy = & $parseAfeJson $raw
    $skrPolicy | ConvertTo-Json -Depth 20 | Out-File "generated/$personaRg/skr-policy.json" -Encoding utf8

    ./scripts/08-prepare-dataset-keys.ps1 -collaborationId $collabId `
        -resourceGroup $personaRg -persona $persona `
        -SkrPolicyFile "generated/$personaRg/skr-policy.json"
}
```

**Verify**:
```powershell
$raw = dotnet run --project $afeProject -- @afeCommon datasets get $collabId "$persona-input-csv$suffix"
(& $parseAfeJson $raw) | ConvertTo-Json -Depth 10
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
> $raw = dotnet run --project $afeProject -- @afeCommon datasets list $collabId
> $datasets = & $parseAfeJson $raw
> $datasets.datasets | Where-Object { $_.id -match "northwind" } | ForEach-Object { Write-Host $_.id }
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
dotnet run --project $afeProject -- @afeCommon queries publish $collabId $queryName --body "@generated/publish/$queryName.json"
```

---

## Step 08: Approve Query `[EACH COLLABORATOR]`

> **Single-collaborator**: Only Woodgrove votes (one vote → `Accepted`).
>
> **Multi-collaborator**: Both collaborators must vote. Northwind needs the
> `$queryName` from Woodgrove (or list queries to find it).

Each collaborator runs in their own terminal:

```powershell
# View query and get proposal ID
$raw = dotnet run --project $afeProject -- @afeCommon queries get $collabId $queryName
$queryInfo = & $parseAfeJson $raw
$queryInfo.data.queryData | Format-Table executionSequence, preConditions, postFilters, data -Wrap
$proposalId = $queryInfo.proposalId
Write-Host "Proposal ID: $proposalId"

# Vote
$voteBody = @{ voteAction = "accept"; proposalId = $proposalId } | ConvertTo-Json -Compress
dotnet run --project $afeProject -- @afeCommon queries vote $collabId $queryName --body $voteBody
```

> **Northwind**: If you don't have `$queryName`, list published queries and set it:
> ```powershell
> $raw = dotnet run --project $afeProject -- @afeCommon queries list $collabId
> (& $parseAfeJson $raw) | ConvertTo-Json -Depth 5
>
> $queryName = "<query-name-from-list>"   # e.g., "query2-v1"
> ```

**Verify**: Query state should be `"Accepted"` after all required votes.

```powershell
$raw = dotnet run --project $afeProject -- @afeCommon queries get $collabId $queryName
$state = (& $parseAfeJson $raw).state
Write-Host "Query state: $state"
```

---

## Step 09: Execute Query `[WOODGROVE]`

```powershell
$runBody = @{ runId = [guid]::NewGuid().ToString() } | ConvertTo-Json -Compress
$raw = dotnet run --project $afeProject -- @afeCommon queries run $collabId $queryName --body $runBody
$runResult = & $parseAfeJson $raw

$jobId = $runResult.id
Write-Host "Job ID: $jobId"
```

> `"status": "success"` means accepted for scheduling, not completed. Takes 10-20 min.

> **Network connectivity**: This step requires the ACCR Frontend Service to reach the Analytics Endpoint of the Collaboration. It can time out due to tenant-specific network configurations:
>
> 1. **NSG (Network Security Group)**: If your tenant has NSGs blocking inbound internet access to the AKS Analytics endpoint on port 443, the query will fail. Contact the ACCR team with the `tenantId` of the collaboration so we can whitelist your tenant — an NSG rule will be updated to allow port 443 access to the AKS cluster.
> 2. **[AVNM (Azure Virtual Network Manager)](https://learn.microsoft.com/en-us/azure/virtual-network-manager/)**: This is a tenant-level policy. Your tenant admin needs to create an AVNM rule to allow port 443 access from the internet by following the documentation linked above.

> **Date-range filtering**: To read datasets within a specific date range,
> pass `startDate` and `endDate` in the request body:
>
> ```powershell
> $runBody = @{
>     runId     = [guid]::NewGuid().ToString()
>     startDate = "2025-09-01"
>     endDate   = "2025-09-02"
> } | ConvertTo-Json -Compress
> $raw = dotnet run --project $afeProject -- @afeCommon queries run $collabId $queryName --body $runBody
> $runResult = & $parseAfeJson $raw
> ```

---

## Step 10: Monitor Query `[ANY]`

```powershell
do {
    $raw = dotnet run --project $afeProject -- @afeCommon runs get $collabId $jobId
    $result = & $parseAfeJson $raw
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
> dotnet run --project $mgmtProject -- collaborations get $collabName `
>     | ConvertFrom-Json | % { $_.properties.health } | ConvertTo-Json -Depth 5
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
$raw = dotnet run --project $afeProject -- @afeCommon queries runs $collabId $queryName
(& $parseAfeJson $raw) | ConvertTo-Json -Depth 10
```

> The output includes execution stats such as **total rows read**, **total rows written**, and **duration** of the query.

### 11.2 Audit Events

```powershell
# All events
$raw = dotnet run --project $afeProject -- @afeCommon audit-events list $collabId
(& $parseAfeJson $raw) | ConvertTo-Json -Depth 10

# Filtered (any subset of --from / --to / --type)
$raw = dotnet run --project $afeProject -- @afeCommon audit-events list $collabId `
    --from "2025-09-01T00:00:00Z" --to "2025-09-30T23:59:59Z" `
    --type "QueryExecution"
(& $parseAfeJson $raw) | ConvertTo-Json -Depth 10
```

### 11.3 Download Output

Auto-detects SSE/CPK mode from metadata. Pass `-JobId` to filter to a specific run.

```powershell
./scripts/11-download-output.ps1 -resourceGroup $personaRg `
    -datasetSuffix "$suffix" -JobId $jobId
```

> Output CSVs are saved to `generated/output/`. Without `-JobId`, downloads the latest.

---

## Step 12: Grafana Dashboards `[OWNER]`

> Grafana dashboards let the owner monitor Spark query execution,
> resource usage, and logs in real time.

### 12.1 Get Readonly Kubeconfig

```powershell
dotnet run --project $mgmtProject -- collaborations get-readonly-kubeconfig `
    $collabName `
    --out "./readonly.kubeconfig"
```

> The CLI fetches the kubeconfig, base64-decodes it, and writes the
> resulting UTF-8 file to the path given by `--out`.

### 12.2 Open Grafana Dashboard

Retrieves admin credentials, opens the browser, and port-forwards to Grafana.

```powershell
./scripts/12-open-grafana-dashboard.ps1 -KubeConfigPath "./readonly.kubeconfig"
```

Login with `admin` and the password printed by the script.

---

## Appendix A: Federated Credential Subject Reference

Format: `{contractId}-{ownerId}` where `contractId` = `"Analytics"` (capital A)
and `ownerId` = JWT `oid` from Step 1.6.

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
| `AADSTS700211: No matching federated identity record` | Wrong issuer URL or stale FIC | Republish dataset; delete/recreate FIC |
| `SSL certificate verify failed` | Endpoint cert mismatch | The CLI is invoked with `--insecure` in the examples for dev/test; do not use this in production |
| `404 Not Found` on frontend | Using ARM resource ID instead of frontend UUID | Use UUID from `afe collaborations list` |
| `ContractNotFound` | Stale CCF endpoint | Create new collaboration |
| `Already voted / Conflict` | Idempotent vote | Safe to ignore |
| `PENDING_RERUN` | Normal scheduling | Keep polling |
| `AZURE_CLIENT_ID must be set when using --use-msal.` | MSAL flow but env var missing | Set `$env:AZURE_CLIENT_ID` before running `dotnet run --project $afeProject -- @afeCommon ...` |
| `dotnet: command not found` | .NET SDK missing or not on PATH | Install .NET SDK 10+ and reopen terminal |

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

### Optional dataset parameters

| Parameter | Description |
|---|---|
| `subdirectory` | Prefix inside the dataset's container to scope the dataset to a sub folder(e.g. `2025-09-01`). Optional, defaults to `""` (entire container). Pass it via the `-subdirectory` parameter of `scripts/08-build-dataset-body.ps1` — see [Step 6.1](#61-build-dataset-body-json) for the call site. |

---

## Appendix E: Query Structure Reference

| Section | Purpose |
|---|---|
| `queryData.segments[]` | Ordered SQL statements with `executionSequence`, `data`, `preConditions`, `postFilters` |
| `inputDatasets[]` | Maps `datasetDocumentId` to SQL view names |
| `outputDataset` | Where results are written |

**Privacy controls**:

- **Pre-conditions** enforce a minimum row count per view. If any view has fewer rows than `minRowCount`, the query aborts.
- **Post-filters** remove groups from the output whose aggregation count is below a threshold, preventing identification of individuals.

Both are defined in the query segments. Edit the thresholds before publishing the query (Step 08).

---

## Appendix F: SDK CLI Verb Reference

The `afe` CLI is defined in `packages/sample/Program.cs`. It wraps the
generated `AnalyticsFrontendAPI` SDK (`CollaborationClient`) and exposes a
verb / sub-verb command structure.

### Global options

| Option | Effect | Env var |
|---|---|---|
| `--endpoint <url>` | Frontend base URL | `ANALYTICS_FRONTEND_ENDPOINT` |
| `--use-msal` | Use MSAL device-code auth instead of `DefaultAzureCredential` | — |
| `--insecure` | Skip TLS validation (dev/test only) | — |
| `--body <json\|@file>` | Inline JSON or `@path` to a JSON file | — |
| `-h`, `--help` | Show help | — |
| — | MSAL client ID (required when `--use-msal`) | `AZURE_CLIENT_ID` |
| — | MSAL tenant (default `common`) | `AZURE_TENANT_ID` |
| — | Persona label for ID-token temp file | `PERSONA` |
| — | AAD scope override | `ANALYTICS_FRONTEND_SCOPE` |

### Verb / sub-verb table

| Verb | Sub-verb | Args | Notes |
|---|---|---|---|
| `collaborations` | `list` | — | Optional `--include-deleted` |
| `collaborations` | `get` | `<collaborationId>` | Optional `--include-deleted` |
| `collaborations` | `report` | `<collaborationId>` | |
| `analytics` | `get` | `<collaborationId>` | |
| `analytics` | `skr-policy` | `<collaborationId> <kid>` | |
| `oidc` | `issuer-info` | `<collaborationId>` | |
| `oidc` | `set-issuer-url` | `<collaborationId>` | `--body` required |
| `oidc` | `keys` | `<collaborationId>` | |
| `invitations` | `list` | `<collaborationId>` | Optional `--include-deleted` |
| `invitations` | `get` | `<collaborationId> <invitationId>` | |
| `invitations` | `accept` | `<collaborationId> <invitationId>` | |
| `datasets` | `list` | `<collaborationId>` | |
| `datasets` | `get` | `<collaborationId> <documentId>` | |
| `datasets` | `publish` | `<collaborationId> <documentId>` | `--body` required |
| `datasets` | `queries` | `<collaborationId> <documentId>` | |
| `consent` | `get` | `<collaborationId> <documentId>` | |
| `consent` | `put` | `<collaborationId> <documentId>` | `--body` required |
| `queries` | `list` | `<collaborationId>` | |
| `queries` | `get` | `<collaborationId> <documentId>` | |
| `queries` | `publish` | `<collaborationId> <documentId>` | `--body` required |
| `queries` | `vote` | `<collaborationId> <documentId>` | `--body` required |
| `queries` | `run` | `<collaborationId> <documentId>` | `--body` required |
| `queries` | `runs` | `<collaborationId> <documentId>` | |
| `runs` | `get` | `<collaborationId> <jobId>` | |
| `secrets` | `put` | `<collaborationId> <secretName>` | `--body` required |
| `audit-events` | `list` | `<collaborationId>` | Optional `--from`, `--to`, `--type` |

Aliases (`collabs-list`, `queries-run`, …) and command flags
(`--list-collaborations`, `--run-query`, …) map to the same verbs — see
`dotnet run --project ./packages/sample/AnalyticsFrontendSample.csproj -- --help` for the full list.

### ARM API (via `cleanroom-mgmt` SDK CLI)

Project: `./packages/mgmt-sample/CleanRoomMgmtSample.csproj`
SDK: `Azure.ResourceManager.CleanRoom`

| Verb | Subverb | Args | Notes |
|---|---|---|---|
| `collaborations` | `create` | `<collaborationName> --location --resource-location [--collaborator]*` | LRO; ~25 min |
| `collaborations` | `get` | `<collaborationName>` | |
| `collaborations` | `list` | (uses `$env:AZURE_RESOURCE_GROUP` or `--resource-group`) | |
| `collaborations` | `delete` | `<collaborationName>` | LRO |
| `collaborations` | `enable-workload` | `<collaborationName> --workload-type AnalyticsStrict` | LRO; ~7 min |
| `collaborations` | `add-collaborator` | `<collaborationName> (--user <email> \| --object-id <oid> [--tenant-id <tenantId>])` | LRO |
| `collaborations` | `pause` / `resume` | `<collaborationName>` | LRO |
| `collaborations` | `recover` | `<collaborationName> [--force]` | LRO |
| `collaborations` | `get-readonly-kubeconfig` | `<collaborationName> --out <file>` | Writes decoded kubeconfig |
| `consortia` | `create` / `get` / `list` / `delete` | (consortium-level operations) | LRO where applicable |

Run `dotnet run --project ./packages/mgmt-sample/CleanRoomMgmtSample.csproj -- --help`
for the full list and per-verb flags.

---

## Appendix G: Collaboration Management

### Force Recover

If the collaboration becomes unresponsive (e.g., `ContractNotFound`, frontend errors on all operations):

```powershell
dotnet run --project $mgmtProject -- collaborations recover `
    $collabName `
    --force
```

> Last-resort operation. Resets internal state. Existing datasets and queries
> need not be republished after recovery.

### Delete Collaboration

```powershell
dotnet run --project $mgmtProject -- collaborations delete $collabName
```

> Permanently deletes the collaboration and all associated resources.

---

## Appendix: App-Based Authentication (SPN)

For CI/CD automation, service principals can replace interactive user login.
The SDK CLI picks SPN credentials up automatically through
`DefaultAzureCredential` when the standard env vars are set — no `--use-msal`
needed.

### Prerequisites

| Requirement | Details |
|---|---|
| Python 3 + `msal` + `cryptography` | `pip install msal cryptography` (only needed for SNI cert flow) |
| App registration | With `serviceManagementReference` in MSFT tenant |
| OneCert certificate | Issued by integrated CA in a KV with OneCert issuer |
| `trustedCertificateSubjects` | Set in app manifest via Azure Portal |

### Token Acquisition

For interactive testing, sign in once and let `DefaultAzureCredential` use the
session:

```powershell
az login --service-principal -u <clientAppId> -p <cert.pem> --tenant <tenantId>
# Subsequent afe calls (without --use-msal) will use the SPN.
```

For automated pipelines, set the env vars consumed by `DefaultAzureCredential`:

```powershell
$env:AZURE_CLIENT_ID       = "<clientAppId>"
$env:AZURE_TENANT_ID       = "<tenantId>"
$env:AZURE_CLIENT_CERTIFICATE_PATH = "<cert.pem>"   # or AZURE_CLIENT_SECRET
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
