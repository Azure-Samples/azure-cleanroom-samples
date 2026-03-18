# Big Data Analytics &mdash; CPK (Customer-Provided Key) <!-- omit from toc -->

This sample illustrates how
[Azure Confidential Clean Rooms](https://learn.microsoft.com/en-us/azure/confidential-computing/confidential-clean-rooms)
can be used to set up a fully managed clean room for confidential Spark SQL
analytics with **Customer-Provided Key (CPK)** encryption. You encrypt data with
your own keys (DEK/KEK stored in Azure Key Vault Premium) before uploading,
and the keys are securely released only to the confidential environment.

If you prefer a simpler setup where Azure Storage handles encryption of data at
rest, use the [SSE variant](README-SSE.md) instead.

## Table of Contents <!-- omit from toc -->

- [Scenario](#scenario)
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Walkthrough](#step-0-prerequisites-setup-northwind-woodgrove)
  - [Step 0 &mdash; Prerequisites setup](#step-0-prerequisites-setup-northwind-woodgrove)
  - [Step 1 &mdash; Create the collaboration](#step-1-create-the-collaboration-woodgrove)
  - [Step 2 &mdash; Accept invitation](#step-2-accept-invitation-northwind)
  - [Step 3 &mdash; Prepare Azure resources](#step-3-prepare-azure-resources-northwind-woodgrove)
  - [Step 4 &mdash; Prepare data](#step-4-prepare-data-northwind-woodgrove)
  - [Step 5 &mdash; Set up identity & OIDC](#step-5-set-up-identity--oidc-northwind-woodgrove)
  - [Step 6 &mdash; Grant clean room access](#step-6-grant-clean-room-access-northwind-woodgrove)
  - [Step 7 &mdash; Publish datasets](#step-7-publish-datasets-northwind-woodgrove)
  - [Step 8 &mdash; Publish the query](#step-8-publish-query-woodgrove)
  - [Step 9 &mdash; Approve the query](#step-9-vote-on-query-northwind-woodgrove)
  - [Step 10 &mdash; Run the query](#step-10-run-query-woodgrove)
  - [Step 11 &mdash; View results](#step-11-view-results)
- [Advanced Topics](#advanced-topics)
- [Appendix &mdash; CLI commands per step](#appendix--cli-commands-per-step)

## Scenario

Woodgrove is an advertiser that wants to generate target audience segments by
performing an overlap analysis with a media publisher, Northwind. Both parties
contribute sensitive datasets to an Azure Confidential Clean Room where a Spark
SQL query joins the data, computes the overlap, and writes the results &mdash;
all without either party exposing raw data to the other.

This is only a sample scenario. You can try any scenario of your choice by providing your own data and query.

## Overview

| Aspect | Details |
|--------|---------|
| **Encryption** | CPK &mdash; DEK wraps data, KEK wraps DEK (Key Vault Premium) |
| **Parties** | Woodgrove (owner), Northwind (publisher) |
| **Data format** | CSV (Parquet and JSON are also supported) |
| **Query engine** | Confidential Spark SQL |

### Parties involved

| Party | Role |
|:---|:---|
| **Woodgrove** | Clean room **owner** &mdash; creates the collaboration, invites Northwind, publishes the query, runs it, and retrieves results. Also contributes sensitive first-party data of its users. |
| **Northwind** | Data **publisher** &mdash; accepts the invitation and contributes sensitive subscriber data which can be matched with data coming in from Woodgrove (advertiser) to identify common users. |

**Which party runs which step?**

| Step | Woodgrove | Northwind |
|:-----|:---------:|:---------:|
| 0 &ndash; Prerequisites | &#10003; | &#10003; |
| 1 &ndash; Create collaboration | &#10003; | |
| 2 &ndash; Accept invitation | | &#10003; |
| 3 &ndash; Prepare resources | &#10003; | &#10003; |
| 4 &ndash; Prepare data | &#10003; | &#10003; |
| 5 &ndash; OIDC issuer | &#10003; | &#10003; |
| 6 &ndash; Grant access | &#10003; | &#10003; |
| 7 &ndash; Publish datasets | &#10003; | &#10003; |
| 8 &ndash; Publish query | &#10003; | |
| 9 &ndash; Approve query | &#10003; | &#10003; |
| 10 &ndash; Run query | &#10003; | |
| 11 &ndash; View results | &#10003; | &#10003; |

---

## Prerequisites

1. **Azure CLI 2.75+** &mdash; [install](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)
2. **Managed Clean Room extension**:
   ```powershell
   az extension add --name managedcleanroom --upgrade
   ```
3. **PowerShell 7.x+** &mdash; [install](https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell)
4. **Azure subscription** with permissions to create storage accounts, Key Vaults (Premium SKU), managed identities, and resource groups

---

## Step 0: Prerequisites setup (northwind, woodgrove)

Open **two separate PowerShell sessions** &mdash; one for each collaborator.
Install CLI extensions, log in, and set your configuration variables:

```powershell
# Login to Azure
az login --use-device-code
```

Now set the session variables. **Each terminal sets its own persona**
&mdash; adjust the values to match your environment:

```powershell
# ── Choose your persona ──────────────────────────────────────────────
$Persona = "woodgrove"           # "northwind" in the other session

# ── Azure target ─────────────────────────────────────────────────────
$Location      = "westeurope"
$ResourceGroup = "rg-accr-$Persona"

# ── Derived names (unique per persona) ───────────────────────────────
$Suffix             = -join ((48..57) + (97..122) | Get-Random -Count 8 | ForEach-Object { [char]$_ })
$StorageAccount     = "accr$($Persona.Substring(0,4))$Suffix"
$OidcStorageAccount = "oidc$($Persona.Substring(0,4))$Suffix"
$KeyVault           = "kv-$($Persona.Substring(0,4))-$Suffix"

# ── Collaboration (must match in both sessions) ─────────────────────
$CollaborationName = "woodgrove-northwind-analytics"

# ── Dataset names (derived from persona) ─────────────────────────────
$DatasetName       = "$Persona-input-csv"
$OutputDatasetName = "woodgrove-output-csv"          # woodgrove only
$QueryName         = "overlap-analysis"

# ── Output directory ─────────────────────────────────────────────────
$OutDir = "./generated"

# ── Error handling ───────────────────────────────────────────────────
$ErrorActionPreference = 'Stop'
```

> [!TIP]
> The `$CollaborationName` must be identical in both sessions since it
> identifies the shared collaboration resource. All other values are
> derived from `$Persona` so they differ per session automatically.

> [!IMPORTANT]
> **Multi-tenant users:** If your Microsoft account is a guest in multiple Azure AD tenants, you **must** specify the correct tenant via `az login --tenant`. Using the wrong tenant will cause access failures at query execution time.

## Step 1: Create the collaboration (woodgrove)

Woodgrove creates the collaboration and invites Northwind.

**1a. Get the owner's identity** (needed for `--user-identity`):

```powershell
$TenantId     = (az account show --query tenantId -o tsv)
$ObjectId     = (az ad signed-in-user show --query id -o tsv)
$UserIdentity = "{tenant-id:$TenantId,object-id:$ObjectId,account-type:microsoft}"
```

**1b. Create the collaboration:**

```powershell
az managedcleanroom collaboration create `
    --collaboration-name $CollaborationName `
    --resource-group $ResourceGroup `
    --location $Location `
    --consortium-type ConfidentialACI `
    --user-identity $UserIdentity
```

**1c. Add each collaborator by email:**

```powershell
az managedcleanroom collaboration add-collaborator `
    --collaboration-name $CollaborationName `
    --resource-group $ResourceGroup `
    --email "northwind-user@contoso.com"         # ← replace with Northwind's email

az managedcleanroom collaboration add-collaborator `
    --collaboration-name $CollaborationName `
    --resource-group $ResourceGroup `
    --email "woodgrove-user@contoso.com"         # ← replace with Woodgrove's email
```

**1d. Enable the analytics workload:**

```powershell
az managedcleanroom collaboration enable-workload `
    --collaboration-name $CollaborationName `
    --resource-group $ResourceGroup `
    --workload-type analytics
```

**1e. Retrieve the collaboration details** (frontend endpoint + ARM ID):

```powershell
az managedcleanroom collaboration show `
    --collaboration-name $CollaborationName `
    --resource-group $ResourceGroup

# Extract the frontend endpoint (PowerShell):
$collab = az managedcleanroom collaboration show `
    --collaboration-name $CollaborationName `
    --resource-group $ResourceGroup | ConvertFrom-Json
$CollaborationId  = $collab.id
$FrontendEndpoint = ($collab.workloads | Where-Object { $_.workloadType -eq "analytics" }).endpoint
Write-Host "Collaboration ID:  $CollaborationId"
Write-Host "Frontend Endpoint: $FrontendEndpoint"
```

> [!NOTE]
> Share the **Collaboration ARM ID** (`$CollaborationId`) and **Frontend Endpoint** (`$FrontendEndpoint`) with Northwind — they need both for Step 2.

Also configure the frontend and log in:

```powershell
az managedcleanroom frontend configure --endpoint $FrontendEndpoint
az managedcleanroom frontend login
```

---

## Step 2: Accept invitation (northwind)

Northwind accepts the invitation to join the collaboration.

Set the values shared by Woodgrove (from Step 1e output):

```powershell
$CollaborationId  = "<paste-collaboration-id-from-woodgrove>"
$FrontendEndpoint = "<paste-frontend-endpoint-from-woodgrove>"
```

**2a. Configure the frontend endpoint:**

```powershell
az managedcleanroom frontend configure --endpoint $FrontendEndpoint
```

**2b. Authenticate** (interactive device code flow):

```powershell
az managedcleanroom frontend login
```

**2c. List pending invitations** and extract the invitation ID:

```powershell
# List invitations
az managedcleanroom frontend invitation list `
    --collaboration-id $CollaborationId

# Extract the invitation ID (PowerShell):
$invitations = (az managedcleanroom frontend invitation list `
    --collaboration-id $CollaborationId | ConvertFrom-Json).invitations
$InvitationId = $invitations[0].invitationId
Write-Host "Invitation ID: $InvitationId"
```

**2d. Accept the invitation:**

```powershell
az managedcleanroom frontend invitation accept `
    --collaboration-id $CollaborationId `
    --invitation-id $InvitationId
```

---

## Step 3: Prepare Azure resources (northwind, woodgrove)

Each collaborator provisions Azure resources.
**Unlike SSE, CPK requires a Key Vault (Premium SKU)** for encryption keys.

```powershell
./scripts/04-prepare-resources.ps1 `
    -resourceGroup $ResourceGroup `
    -variant cpk `
    -location $Location `
    -outDir $OutDir
```

The script creates a resource group, storage account, Key Vault (Premium),
managed identity, and RBAC assignments.

> [!NOTE]
> Allow a few minutes for RBAC assignments to propagate before running Step 4.

---

## Step 4: Prepare data (northwind, woodgrove)

**4a. Generate demo data:**

> In a real scenario you would already have your own data &mdash; skip to 4b and
> point `-dataDir` at your existing files.

```powershell
./demos/generate-data.ps1 -persona $Persona
```

This generates ~8 MB of synthetic CSV data per persona:
- **Northwind**: `audience_id`, `hashed_email`, `annual_income`, `region` (US / UK / IN / CA)
- **Woodgrove**: `user_id`, `hashed_email`, `purchase_history`

**4b. Encrypt and upload data to Azure Storage:**

Each collaborator encrypts their data with AES-256 and uploads the encrypted files.

```powershell
./scripts/05-prepare-data.ps1 `
    -resourceGroup $ResourceGroup `
    -variant cpk `
    -persona $Persona `
    -dataDir "./demos/datasource/$Persona/input/csv" `
    -collaborationId $CollaborationId `
    -outDir $OutDir
```

The script generates a random AES-256 DEK, fetches the clean room policy via
`az managedcleanroom frontend analytics cleanroompolicy`, creates an exportable
RSA-HSM KEK in Key Vault with a Secure Key Release (SKR) policy tied to the
clean room, wraps the DEK with the KEK, uploads data with CPK encryption headers,
and saves metadata for Step 7.

---

## Step 5: Set up identity & OIDC (northwind, woodgrove)

Each collaborator sets up OIDC issuer infrastructure and saves identity metadata.

```powershell
./scripts/06-setup-identity.ps1 `
    -resourceGroup $ResourceGroup `
    -persona $Persona `
    -collaborationId $CollaborationId `
    -frontendEndpoint $FrontendEndpoint `
    -outDir $OutDir
```

The script creates an OIDC storage account, fetches JWKS from the frontend,
uploads OpenID discovery documents, registers the issuer URL with CGS, and
saves identity metadata (client ID, tenant ID, issuer URL) for Steps 6 and 7.

> [!NOTE]
> The OIDC issuer lets the clean room prove (via hardware attestation) that it is
> running approved code, so Azure AD grants it the permissions you assign in Step 6.

---

## Step 6: Grant clean room access (northwind, woodgrove)

Each collaborator grants the clean room workload access.
**Unlike SSE, CPK also requires Key Vault RBAC** (Crypto Officer + Secrets User)
so the clean room can unwrap encryption keys.

```powershell
./scripts/07-grant-access.ps1 `
    -resourceGroup $ResourceGroup `
    -variant cpk `
    -collaborationId $CollaborationId `
    -contractId "analytics" `
    -userId (az ad signed-in-user show --query id -o tsv) `
    -outDir $OutDir
```

The script assigns Storage Blob Data Owner, Key Vault Crypto Officer,
and Key Vault Secrets User RBAC roles on the managed identity, then creates
a federated credential linking the OIDC issuer to the identity.

> [!TIP]
> The federated credential subject uses the format `{contractId}-{userId}`.
> Get your `userId` via `az ad signed-in-user show --query id -o tsv`.

---

## Step 7: Publish datasets (northwind, woodgrove)

Each collaborator publishes their dataset metadata to the collaboration.
The CPK encryption secret references (DEK/KEK) are filled in automatically from
the metadata generated in previous steps.

> [!IMPORTANT]
> **7a. Review the dataset config.**
> Open [`templates/dataset-config.json`](templates/dataset-config.json) and adjust
> `schema` and `allowedFields` for each dataset if your data differs from the demo
> defaults. All other values (storage account, identity, key vault, encryption
> secrets) are filled in automatically by the script in the next step.

**7b. Generate the publish-ready JSON files** from metadata produced by Steps 3–5:

```powershell
./scripts/populate-templates.ps1 -variant cpk -persona $Persona -resourceGroup $ResourceGroup
```

Output files are written to `generated/datasets/cpk/`.

**7c. Publish the datasets:**

```powershell
# Publish your input dataset
az managedcleanroom frontend analytics dataset publish `
    --collaboration-id $CollaborationId `
    --document-id $DatasetName `
    --body "@generated/datasets/cpk/$Persona-input-dataset.json"
```

Woodgrove also publishes the output dataset:

```powershell
# Woodgrove only
az managedcleanroom frontend analytics dataset publish `
    --collaboration-id $CollaborationId `
    --document-id $OutputDatasetName `
    --body "@generated/datasets/cpk/woodgrove-output-dataset.json"
```

**7d. Enable execution consent** on each published dataset (required for the clean room to access the data at query run time):

```powershell
# Enable consent for your input dataset
az managedcleanroom frontend consent set `
    --collaboration-id $CollaborationId `
    --document-id $DatasetName `
    --consent-action enable
```

Woodgrove also enables consent for the output dataset:

```powershell
# Woodgrove only
az managedcleanroom frontend consent set `
    --collaboration-id $CollaborationId `
    --document-id $OutputDatasetName `
    --consent-action enable
```

**7e. Verify** each dataset was published:

```powershell
az managedcleanroom frontend analytics dataset show `
    --collaboration-id $CollaborationId `
    --document-id $DatasetName
```

> [!NOTE]
> **Northwind** publishes 1 dataset (read-only). **Woodgrove** publishes 2 datasets (read-only input + write output for results).

---

## Step 8: Publish query (woodgrove)

Woodgrove proposes a Spark SQL query that combines both datasets.

> [!IMPORTANT]
> **8a. Review the query config.**
> Open [`templates/query.json`](templates/query.json) and adjust the SQL segments
> and dataset document IDs if your query differs from the demo defaults.

**8b. Publish the query:**

```powershell
az managedcleanroom frontend analytics query publish `
    --collaboration-id $CollaborationId `
    --document-id $QueryName `
    --body "@templates/query.json"
```

**8c. Enable execution consent** on the query:

```powershell
az managedcleanroom frontend consent set `
    --collaboration-id $CollaborationId `
    --document-id $QueryName `
    --consent-action enable
```

**8d. Verify** the query was published:

```powershell
az managedcleanroom frontend analytics query show `
    --collaboration-id $CollaborationId `
    --document-id $QueryName
```

All collaborators must now vote to approve this query before execution.

---

## Step 9: Vote on query (northwind, woodgrove)

Both collaborators must vote to approve the query before it can execute.

```powershell
az managedcleanroom frontend analytics query vote accept `
    --collaboration-id $CollaborationId `
    --document-id $QueryName
```

No `--body` is needed — per CLI help, the body is optional (accepts reason/metadata only).

Verify query state after both votes:

```powershell
az managedcleanroom frontend analytics query show `
    --collaboration-id $CollaborationId `
    --document-id $QueryName
```

After both votes, the query state changes from "Proposed" to "Accepted".

---

## Step 10: Run query (woodgrove)

Execute the approved query in the confidential clean room.

**10a. Submit the query run:**

```powershell
$runResponse = az managedcleanroom frontend analytics query run `
    --collaboration-id $CollaborationId `
    --document-id $QueryName | ConvertFrom-Json

$JobId = $runResponse.jobId
Write-Host "Job ID: $JobId"
```

> [!NOTE]
> Each invocation submits a new query run — re-running the command starts
> a fresh execution.

**10b. Poll for completion** using `runresult show`:

```powershell
# Poll every 15 seconds until COMPLETED or FAILED
do {
    Start-Sleep -Seconds 15
    $result = az managedcleanroom frontend analytics query runresult show `
        --collaboration-id $CollaborationId `
        --job-id $JobId | ConvertFrom-Json
    $state = $result.status.applicationState.state
    Write-Host "Status: $state"
} while ($state -notin @("COMPLETED", "FAILED", "SUBMISSION_FAILED"))

# Display the result
$result | ConvertTo-Json -Depth 10
```

---

## Step 11: View results

Once the query completes, the results are written to Woodgrove's output storage
container (`woodgrove-output`). In CPK mode the results are encrypted with the
same DEK used during upload. To download and read them, you need to reconstruct
the DEK from the wrapped secret in Key Vault using the local KEK private key.

**11a. Reconstruct the DEK:**

The raw DEK was generated in memory during Step 4 and only its wrapped (RSA-OAEP-SHA256
encrypted) form was stored in Key Vault. The KEK private PEM was saved locally.

```powershell
# Retrieve the wrapped DEK (hex string) from Key Vault
$wrappedDekHex = az keyvault secret show `
    --vault-name $KeyVault `
    --name "woodgrove-output-dek-wrapped" `
    --query value -o tsv

# Convert hex string back to bytes
$wrappedDekBytes = [byte[]]::new($wrappedDekHex.Length / 2)
for ($i = 0; $i -lt $wrappedDekHex.Length; $i += 2) {
    $wrappedDekBytes[$i / 2] = [Convert]::ToByte($wrappedDekHex.Substring($i, 2), 16)
}

# Load the KEK private key (saved during Step 4)
$kekPemPath = Join-Path $OutDir $ResourceGroup "woodgrove-kek-private.pem"
$kekPem = Get-Content $kekPemPath -Raw
$rsa = [System.Security.Cryptography.RSA]::Create()
$rsa.ImportFromPem($kekPem)

# Unwrap (decrypt) the DEK
$dekBytes = $rsa.Decrypt($wrappedDekBytes,
    [System.Security.Cryptography.RSAEncryptionPadding]::OaepSHA256)
$dekBase64 = [Convert]::ToBase64String($dekBytes)
$dekSha256Base64 = [Convert]::ToBase64String(
    [System.Security.Cryptography.SHA256]::HashData($dekBytes))

$rsa.Dispose()
Write-Host "DEK reconstructed successfully." -ForegroundColor Green
```

**11b. List output blobs:**

```powershell
az storage blob list `
    --account-name $StorageAccount `
    --container-name "woodgrove-output" `
    --auth-mode login `
    --output table
```

**11c. Download and decrypt results:**

Since CPK blobs require the encryption key in the request headers, use the
REST API to download each blob:

```powershell
$ResultsDir = "./results"
New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null

$tokenJson = az account get-access-token `
    --resource https://storage.azure.com --output json | ConvertFrom-Json
$accessToken = $tokenJson.accessToken

# Get blob endpoint
$storageJson = az storage account show `
    --name $StorageAccount `
    --resource-group $ResourceGroup `
    --output json | ConvertFrom-Json
$blobEndpoint = $storageJson.primaryEndpoints.blob

# List and download each blob
$blobs = az storage blob list `
    --account-name $StorageAccount `
    --container-name "woodgrove-output" `
    --auth-mode login `
    --output json | ConvertFrom-Json

foreach ($blob in $blobs) {
    $blobUrl = "${blobEndpoint}woodgrove-output/$($blob.name)"
    $localPath = Join-Path $ResultsDir $blob.name
    New-Item -ItemType Directory -Path (Split-Path $localPath) -Force | Out-Null

    $headers = @{
        "Authorization"              = "Bearer $accessToken"
        "x-ms-version"               = "2024-11-04"
        "x-ms-encryption-key"        = $dekBase64
        "x-ms-encryption-key-sha256" = $dekSha256Base64
        "x-ms-encryption-algorithm"  = "AES256"
    }

    Invoke-RestMethod -Uri $blobUrl -Method Get -Headers $headers `
        -OutFile $localPath
    Write-Host "Downloaded: $($blob.name)" -ForegroundColor Green
}
```

**11d. View the downloaded data:**

```powershell
Get-ChildItem $ResultsDir -Recurse -File | ForEach-Object {
    Write-Host "--- $($_.Name) ---" -ForegroundColor Cyan
    Get-Content $_.FullName | Select-Object -First 20
}
```

**11e. View run history and audit events:**

```powershell
# Query run history
az managedcleanroom frontend analytics query runhistory list `
    --collaboration-id $CollaborationId `
    --document-id $QueryName

# Audit events
az managedcleanroom frontend analytics auditevent list `
    --collaboration-id $CollaborationId
```

---

## Advanced Topics

### Date range filtering

Filter input data by date range when executing a query:

```powershell
# The query run CLI supports an optional --body for date range filtering:
$runBody = @{
    startDate = "2025-09-01"
    endDate   = "2025-09-02"
} | ConvertTo-Json

az managedcleanroom frontend analytics query run `
    --collaboration-id $CollaborationId `
    --document-id $QueryName `
    --body $runBody
```

When a date range is provided, only matching partitions are read.

### Privacy controls

**Pre-conditions** enforce a minimum row count per view. If any view has fewer
rows than `minRowCount`, the query aborts.

**Post-filters** remove groups from the output whose aggregation count is
below a threshold, preventing identification of individuals.

Both are defined in the query segments. Edit the thresholds before publishing
the query (Step 8).

### Audit events

View all audit events for the collaboration directly via the CLI:

```powershell
az managedcleanroom frontend analytics auditevent list `
    --collaboration-id $CollaborationId
```

Events include query execution start/completion, input/output row counts, dataset access, and failures. The audit log is maintained in the CCF ledger and is tamper-proof.

---

## Appendix &mdash; CLI commands per step

| Step | Description | Approach | CLI Commands Used |
|------|-------------|----------|-------------------|
| 0 | Prerequisites | **Direct CLI** | `az extension add`, `az login` |
| 1 | Create collaboration | **Direct CLI** | `az account show`, `az ad signed-in-user show`, `az managedcleanroom collaboration create`, `add-collaborator` (×2), `enable-workload`, `collaboration show` |
| 2 | Accept invitation | **Direct CLI** | `az managedcleanroom frontend configure`, `login`, `invitation list`, `invitation accept` |
| 3 | Prepare resources | **Helper script** | `az group create`, `az storage account create`, `az keyvault create --sku Premium`, `az identity create`, `az role assignment create` (×3) |
| 4 | Prepare data | **Helper script** | `./demos/generate-data.ps1` (demo only) + `./scripts/05-prepare-data.ps1 -variant cpk` — `az managedcleanroom frontend analytics cleanroompolicy`, `az storage account show`, `az keyvault show`, `az storage container create`, `az keyvault key import` (RSA-HSM, exportable, SKR policy), `az keyvault secret set`, `Invoke-RestMethod` (CPK upload) |
| 5 | Identity & OIDC | **Helper script** | `az storage account create`, `az storage blob service-properties update`, `az role assignment create`, `az managedcleanroom frontend oidc issuerinfo show`, OIDC REST calls, `az storage blob upload` (×2), `az identity show` |
| 6 | Grant access (CPK) | **Helper script** | `az identity show`, `az storage account show`, `az keyvault show`, `az role assignment create` (×3: Storage + KV Crypto + KV Secrets), `az identity federated-credential create` |
| 7 | Publish datasets | Config + script | Edit `templates/dataset-config.json`, run `populate-templates.ps1`, then `az managedcleanroom frontend analytics dataset publish` (×3), `consent set --consent-action enable` (×3), `dataset show` |
| 8 | Publish query | **Direct CLI** | `az managedcleanroom frontend analytics query publish --body @file`, `consent set --consent-action enable`, `query show` |
| 9 | Vote on query | **Direct CLI** | `az managedcleanroom frontend analytics query vote accept` (×2), `query show` |
| 10 | Run query | **Direct CLI** | `az managedcleanroom frontend analytics query run`, `query runresult show` (polling) |
| 11 | View results | **Direct CLI** | `az keyvault secret show` (wrapped DEK), RSA unwrap (local KEK PEM), `az storage blob list`, `Invoke-RestMethod` (CPK download), `az managedcleanroom frontend analytics query runhistory list`, `auditevent list` |

> **No `az cleanroom` dependency.** All steps use standard Azure CLI or the `az managedcleanroom` extension.
