# Big Data Analytics — REST API (`az rest` + `Invoke-RestMethod`)

This guide uses **`az rest`** for ARM collaboration operations and
**`Invoke-RestMethod`** for frontend service operations, with the same helper
scripts for Azure resource provisioning.

For the CLI variant using `az managedcleanroom`, see [README-CLI.md](README-CLI.md).

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
| **API mode** | `az rest` (ARM) + `Invoke-RestMethod` (frontend) |
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

---

## Table of Contents

- [Scenario](#scenario)
- [Overview](#overview)
- [Step 01: Prerequisites](#step-01-prerequisites) `[ALL]`
  - [1.1 Requirements](#11-requirements)
  - [1.2 Terminal T1 (Owner) — Variables](#12-terminal-t1-owner--variables)
  - [1.3 Each Collaborator Terminal — Variables](#13-each-collaborator-terminal--variables)
  - [1.4 Acquire Token & Extract OID](#14-acquire-token--extract-oid-each-collaborator) `[EACH COLLABORATOR]`
- [Step 02: Create Collaboration](#step-02-create-collaboration) `[OWNER]`
  - [2.1 Create Resource Group](#21-create-resource-group)
  - [2.2 Create Collaboration](#22-create-collaboration)
  - [2.3 Enable Analytics Workload](#23-enable-analytics-workload)
  - [2.4 Add More Collaborators (Optional)](#24-add-more-collaborators-optional)
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
- [Appendix F: REST API Endpoint Reference](#appendix-f-rest-api-endpoint-reference)
- [Appendix: App-Based Authentication (SPN)](#appendix-app-based-authentication-spn)

---

## Step 01: Prerequisites `[ALL]`

### 1.1 Requirements

| Requirement | Details |
|---|---|
| Azure CLI | 2.75.0+ |
| PowerShell | 7.x+ |
| MSAL.PS module | `Install-Module MSAL.PS -Scope CurrentUser -Force` |
| azcopy | v10+ (CPK mode only) |
| Resource provider | `Microsoft.CleanRoom` registered in the owner's subscription |
| Feature flags | `EUAPParticipation` (see below) |

```powershell
az feature register --namespace Microsoft.Resources --name EUAPParticipation

# Check registration status (wait until all show "Registered")
az feature show --namespace Microsoft.Resources --name EUAPParticipation --query properties.state -o tsv

az provider register --namespace Microsoft.CleanRoom
```

> **Quota check:** This sample deploys an AKS cluster and Confidential ACI
> container groups in the **West US** region. Ensure your subscription has the
> following minimum quota in that region before proceeding:
>
> | Resource | Minimum vCPUs | SKU / Family |
> |---|---|---|
> | AKS node pool | 8 | Standard_D4ds_v5 (Ddsv5 family) |
> | Confidential ACI | 6 | Confidential container groups |

> The `managedcleanroom` CLI extension is **not required** for this guide.

### 1.2 Terminal T1 (Owner) — Variables

```powershell
az login
$account = az account show -o json | ConvertFrom-Json
$subscription = $account.id
$tenantId = $account.tenantId

$location = "eastus2euap"
$collabName = "<collaboration-name>"
$collabRg = "<collaboration-resource-group>"

# ARM API
$armEndpoint = "https://eastus2euap.management.azure.com"
$armApiVersion = "2026-03-31-preview"
$armResource = "https://management.azure.com/"   # az rest needs explicit resource for EUAP endpoint
$collabArmUrl = "$armEndpoint/subscriptions/$subscription/resourceGroups/$collabRg/providers/Microsoft.CleanRoom/Collaborations/$collabName"
```

### 1.3 Each Collaborator Terminal — Variables

```powershell
az login
$account = az account show -o json | ConvertFrom-Json
$subscription = $account.id
$tenantId = $account.tenantId

$location = "eastus2euap"
$EncryptionMode = "SSE"    # "SSE" or "CPK"
$iteration = 0

$persona = "woodgrove"                # "woodgrove" or "northwind"
$personaRg = "cr-e2e-$persona-rg"
$personaEmail = "<your-email>"

az group create --name $personaRg --location $location -o none 2>$null

$frontend = "https://prod.workload-frontendcentraluseuap.cleanroom.cloudapp.azure.net"
$feApiVersion = "2026-03-01-preview"
$oidcStorageAccount = "cleanroomoidc"   # MSFT tenant; omit for other tenants

# Helper for all frontend REST calls (used throughout this guide)
function Invoke-Frontend {
    param([string]$Method = "GET", [string]$Path, [object]$Body)
    $feToken = (Get-Content $personaTokenFile -Raw).Trim()
    $headers = @{ Authorization = "Bearer $feToken"; "Content-Type" = "application/json" }
    $url = if ($Path) { "$frontend/collaborations/$Path" } else { "$frontend/collaborations" }
    if ($url -notmatch '\?') { $url += "?api-version=$feApiVersion" }
    else { $url += "&api-version=$feApiVersion" }
    $params = @{ Uri = $url; Method = $Method; Headers = $headers; SkipCertificateCheck = $true }
    if ($Body) {
        $params.Body = if ($Body -is [string]) { $Body } else { $Body | ConvertTo-Json -Depth 20 }
        $params.ContentType = "application/json"
    }
    return Invoke-RestMethod @params
}
```

### 1.4 Acquire Token & Extract OID `[EACH COLLABORATOR]`

#### 1.4.1 Acquire Token

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

#### 1.4.2 Extract OID from Token

```powershell
$tokenB64 = (Get-Content $personaTokenFile -Raw).Split('.')[1]
$padLen = (4 - $tokenB64.Length % 4) % 4
$padded = $tokenB64 + ('=' * $padLen)
$claims = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($padded)) | ConvertFrom-Json
$personaOid = $claims.oid
Write-Host "JWT oid: $personaOid"
```

> **CRITICAL**: Always use the JWT `oid`, NOT `az ad signed-in-user show --query id`.
> For MSA accounts these differ. See [Appendix A](#appendix-a-federated-credential-subject-reference).

---

## Step 02: Create Collaboration `[OWNER]`

### 2.1 Create Resource Group

```powershell
az group create --name $collabRg --location $location -o none
```

### 2.2 Create Collaboration

```powershell
$collaboratorEmail = "<woodgrove-email>"
$createBody = @{
    location = $location
    properties = @{
        collaborators = @(@{ userIdentifier = $collaboratorEmail })
    }
} | ConvertTo-Json -Depth 5
[System.IO.File]::WriteAllText("$PWD/body.json", $createBody)
az rest --method PUT `
    --url "$collabArmUrl`?api-version=$armApiVersion" `
    --resource $armResource `
    --headers "Content-Type=application/json" `
    --body "@body.json"
```

> The `collaborators` array adds collaborators at creation time itself.
> To add more collaborators later, see [Step 2.4](#24-add-more-collaborators-optional).

> **NOTE**: `location` must be `eastus2euap` — this is where the Microsoft.CleanRoom RP is deployed.
> Actual resources (AKS cluster, CACI instances) are created in `westus`. Configurable region support is coming soon.

**Runtime**: ~25 minutes. Poll `provisioningState` until `Succeeded`:

```powershell
do {
    $collab = az rest --method GET --url "$collabArmUrl`?api-version=$armApiVersion" --resource $armResource -o json | ConvertFrom-Json
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] provisioningState: $($collab.properties.provisioningState)"
    Start-Sleep -Seconds 60
} while ($collab.properties.provisioningState -notin @("Succeeded", "Failed"))
```

### 2.3 Enable Analytics Workload

```powershell
$enableBody = @{ workloadType = "analytics" } | ConvertTo-Json
[System.IO.File]::WriteAllText("$PWD/body.json", $enableBody)
az rest --method POST `
    --url "$collabArmUrl/enableWorkload`?api-version=$armApiVersion" `
    --resource $armResource `
    --headers "Content-Type=application/json" `
    --body "@body.json"
```

**Runtime**: ~7 minutes. Poll `collaborationState` until `Provisioned`:

```powershell
do {
    $collab = az rest --method GET --url "$collabArmUrl`?api-version=$armApiVersion" --resource $armResource -o json | ConvertFrom-Json
    $wl = $collab.properties.workloads | Where-Object { $_.workloadType -eq "analytics" }
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] collaborationState: $($collab.properties.collaborationState) | workload: $($wl.endpoint)"
    Start-Sleep -Seconds 30
} while ($collab.properties.collaborationState -notin @("Provisioned", "Failed"))
```

Then wait for `healthState` to become `Ok`:

```powershell
do {
    $collab = az rest --method GET --url "$collabArmUrl`?api-version=$armApiVersion" --resource $armResource -o json | ConvertFrom-Json
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
$addBody = @{ collaborator = @{ userIdentifier = $collaboratorEmail } } | ConvertTo-Json
[System.IO.File]::WriteAllText("$PWD/body.json", $addBody)
az rest --method POST `
    --url "$collabArmUrl/addCollaborator`?api-version=$armApiVersion" `
    --resource $armResource `
    --headers "Content-Type=application/json" `
    --body "@body.json"
```

> **IMPORTANT**: The body requires **camelCase** keys (`collaborator`, `userIdentifier`).
> We use `[System.IO.File]::WriteAllText()` to write body files (avoids BOM encoding issues).

**Verify**:
```powershell
az rest --method GET --url "$collabArmUrl`?api-version=$armApiVersion" --resource $armResource -o json
```

---

## Step 03: Accept Invitations `[EACH COLLABORATOR]`

### 3.1 Get Collaboration UUID

```powershell
$collabs = (Invoke-Frontend -Path "" -Method GET).collaborations
$collabs | Format-Table @{L='#';E={[array]::IndexOf($collabs,$_)+1}}, collaborationName, collaborationId, userStatus

$choice = Read-Host "Enter the number of your collaboration"
$collabId = $collabs[[int]$choice - 1].collaborationId
Write-Host "Selected: $collabId"
```

### 3.2 Accept Invitation

```powershell
$invitations = (Invoke-Frontend -Path "$collabId/invitations" -Method GET).invitations
$invitations | Format-Table invitationId, accountType, status

$invitationId = $invitations[0].invitationId

Invoke-Frontend -Path "$collabId/invitations/$invitationId/accept" -Method POST
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

$jwks = Invoke-Frontend -Path "$collabId/oidc/keys" -Method GET
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
if ($oidcStorageAccount) { $oidcParams["OidcStorageAccount"] = $oidcStorageAccount }

./scripts/06-setup-oidc-storage.ps1 @oidcParams
```

### 5.3 Register Issuer URL with Frontend

```powershell
$issuerUrl = (Get-Content "generated/$personaRg/issuer-url.txt" -Raw).Trim()

Invoke-Frontend -Path "$collabId/oidc/setIssuerUrl" -Method POST `
    -Body @{ url = $issuerUrl }
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

> **Bring your own data**: If you want to provide your own datasets, upload your data directly to the
> storage accounts created for your persona and update the `schema` and `accessPolicy` in the dataset
> body files: `generated/publish/$persona-input-dataset.json` and `generated/publish/$persona-output-dataset.json`.

### 6.2 Publish Input Dataset

```powershell
$inputBody = Get-Content "generated/publish/$persona-input-dataset.json" -Raw

Invoke-Frontend -Path "$collabId/analytics/datasets/$persona-input-csv$suffix/publish" `
    -Method POST -Body $inputBody
```

### 6.3 Publish Output Dataset (Woodgrove only)

```powershell
if ($persona -eq "woodgrove") {
    $outputBody = Get-Content "generated/publish/woodgrove-output-dataset.json" -Raw

    Invoke-Frontend -Path "$collabId/analytics/datasets/woodgrove-output-csv$suffix/publish" `
        -Method POST -Body $outputBody
}
```

> Execution consent is enabled by default at publish time. To revoke or re-enable later:
> ```powershell
> Invoke-Frontend -Path "$collabId/consent/<document-name>" `
>     -Method PUT -Body @{ consentAction = "disable" }   # or "enable"
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
Invoke-Frontend -Path "$collabId/analytics/datasets/$persona-input-csv$suffix" | ConvertTo-Json -Depth 10
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
> $datasets = Invoke-Frontend -Path "$collabId/analytics/datasets" -Method GET
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
$queryBody = Get-Content "generated/publish/$queryName.json" -Raw

Invoke-Frontend -Path "$collabId/analytics/queries/$queryName/publish" `
    -Method POST -Body $queryBody
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
$queryInfo = Invoke-Frontend -Path "$collabId/analytics/queries/$queryName"
$queryInfo.data.queryData | Format-Table executionSequence, preConditions, postFilters, data -Wrap
$proposalId = $queryInfo.proposalId
Write-Host "Proposal ID: $proposalId"

# Vote
Invoke-Frontend -Path "$collabId/analytics/queries/$queryName/vote" `
    -Method POST -Body @{ voteAction = "accept"; proposalId = $proposalId }
```

> **Northwind**: If you don't have `$queryName`, list published queries and set it:
> ```powershell
> $queries = Invoke-Frontend -Path "$collabId/analytics/queries" -Method GET
> $queries | ConvertTo-Json -Depth 5
>
> $queryName = "<query-name-from-list>"   # e.g., "query2-v1"
> ```

**Verify**: Query state should be `"Accepted"` after all required votes.

```powershell
$state = (Invoke-Frontend -Path "$collabId/analytics/queries/$queryName").state
Write-Host "Query state: $state"
```

---

## Step 09: Execute Query `[WOODGROVE]`

```powershell
$runBody = @{ runId = [guid]::NewGuid().ToString() }
$runResult = Invoke-Frontend -Path "$collabId/analytics/queries/$queryName/run" `
    -Method POST -Body $runBody

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
> $runBody = @{ runId = [guid]::NewGuid().ToString(); startDate = "2025-09-01"; endDate = "2025-09-02" }
> $runResult = Invoke-Frontend -Path "$collabId/analytics/queries/$queryName/run" `
>     -Method POST -Body $runBody
> ```

---

## Step 10: Monitor Query `[ANY]`

```powershell
do {
    $result = Invoke-Frontend -Path "$collabId/analytics/runs/$jobId"
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
> az rest --method GET --resource "https://management.azure.com/" `
>     --url "$collabArmUrl`?api-version=$armApiVersion" `
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
$history = Invoke-Frontend -Path "$collabId/analytics/queries/$queryName/runs"
$history | ConvertTo-Json -Depth 10
```

> The output includes execution stats such as **total rows read**, **total rows written**, and **duration** of the query.

### 11.2 Audit Events

```powershell
$audit = Invoke-Frontend -Path "$collabId/analytics/auditevents"
$audit | ConvertTo-Json -Depth 10
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
| `AADSTS700211: No matching federated identity record` | Wrong issuer URL or stale FIC | Republish dataset; delete/recreate FIC |
| `SSL certificate verify failed` | EUAP endpoint cert mismatch | Use `-SkipCertificateCheck` on `Invoke-RestMethod` |
| `404 Not Found` on frontend | Using ARM resource ID instead of frontend UUID | Use UUID from `Invoke-Frontend -Path ""` |
| `ContractNotFound` | Stale CCF endpoint | Create new collaboration |
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

- **Pre-conditions** enforce a minimum row count per view. If any view has fewer rows than `minRowCount`, the query aborts.
- **Post-filters** remove groups from the output whose aggregation count is below a threshold, preventing identification of individuals.

Both are defined in the query segments. Edit the thresholds before publishing the query (Step 08).

---

## Appendix F: REST API Endpoint Reference

### ARM API (via `az rest`)

Base: `https://eastus2euap.management.azure.com`
API version: `2026-03-31-preview`

| Operation | Method | URL |
|---|---|---|
| Create collaboration | PUT | `.../providers/Microsoft.CleanRoom/Collaborations/{name}` |
| Show collaboration | GET | `.../providers/Microsoft.CleanRoom/Collaborations/{name}` |
| Enable workload | POST | `.../Collaborations/{name}/enableWorkload` |
| Add collaborator | POST | `.../Collaborations/{name}/addCollaborator` |

### Frontend API (via `Invoke-RestMethod`)

Base: `{frontendEndpoint}/collaborations`
API version: `2026-03-01-preview`

| Operation | Method | Path |
|---|---|---|
| List collaborations | GET | `/` |
| List invitations | GET | `/{id}/invitations` |
| Accept invitation | POST | `/{id}/invitations/{invId}/accept` |
| OIDC keys | GET | `/{id}/oidc/keys` |
| Set issuer URL | POST | `/{id}/oidc/setIssuerUrl` |
| Publish dataset | POST | `/{id}/analytics/datasets/{docId}/publish` |
| Show dataset | GET | `/{id}/analytics/datasets/{docId}` |
| Set consent | PUT | `/{id}/consent/{docId}` |
| Publish query | POST | `/{id}/analytics/queries/{docId}/publish` |
| Show query | GET | `/{id}/analytics/queries/{docId}` |
| Vote on query | POST | `/{id}/analytics/queries/{docId}/vote` |
| Run query | POST | `/{id}/analytics/queries/{docId}/run` |
| Run result | GET | `/{id}/analytics/runs/{jobId}` |
| Run history | GET | `/{id}/analytics/queries/{docId}/runs` |
| Audit events | GET | `/{id}/analytics/auditevents` |

---

## Appendix G: Collaboration Management

### Force Recover

If the collaboration becomes unresponsive (e.g., `ContractNotFound`, frontend errors on all operations):

```powershell
$recoverBody = @{ forceRecover = $true } | ConvertTo-Json
[System.IO.File]::WriteAllText("$PWD/body.json", $recoverBody)
az rest --method POST `
    --url "$collabArmUrl/recover`?api-version=$armApiVersion" `
    --resource $armResource `
    --headers "Content-Type=application/json" `
    --body "@body.json"
```

> Last-resort operation. Resets internal state. Existing datasets and queries
> need not be republished after recovery.

### Delete Collaboration

```powershell
az rest --method DELETE `
    --url "$collabArmUrl`?api-version=$armApiVersion" `
    --resource $armResource
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
