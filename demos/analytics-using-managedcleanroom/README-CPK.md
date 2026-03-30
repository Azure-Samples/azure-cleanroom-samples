# Big Data Analytics — CPK (Customer-Provided Key) — Native CLI Variant <!-- omit from toc -->

This walkthrough demonstrates multi-party big data analytics using a **Managed Clean Room** with **Customer-Provided Key (CPK)** encryption. You control the encryption keys (DEK/KEK stored in Azure Key Vault) for your data. Azure Storage encrypts data server-side using your CPK — you upload plaintext via `azcopy --cpk-by-value`, and the storage service handles encryption/decryption transparently using the key you provide in HTTP headers.

> [!IMPORTANT]
> **This variant has zero `az cleanroom` dependency.** Every step uses standard Azure CLI or the public `az managedcleanroom` extension. Step 8 (Publish datasets) constructs the DatasetSpecification JSON natively in PowerShell — including CPK encryption secret references (DEK/KEK). The JSON structure was derived from the cleanroom extension source code (`CleanRoomSpecification` model in cleanroom v5.0.0). See the [Architecture](#architecture) section for details.

> [!WARNING]
> **CLI help is insufficient for building automation.** The `az managedcleanroom` CLI help (`--help`) documents input parameters but does not describe output schemas, error codes, or response fields. For example, `collaboration show` returns JSON but the field names (e.g., where the frontend endpoint URL lives) are undocumented. `add-collaborator` has no documented error codes to distinguish "already added" from a real failure. Building idempotent scripts required trial-and-error and error-text matching.
>
> 🔴 **ACCR team: Please enrich `--help` output with response schemas, error codes, and examples.**

> [!NOTE]
> **Internal references used to validate this walkthrough:**
> - **POC architecture docs:** [`azure-core/azure-cleanroom`](https://github.com/azure-core/azure-cleanroom) repo → [`poc/managed-cleanroom/`](https://github.com/azure-core/azure-cleanroom/tree/develop/poc/managed-cleanroom) on the `develop` branch — design docs, sequence diagrams, security model
> - **Frontend service source code:** [`azure-core/azure-cleanroom`](https://github.com/azure-core/azure-cleanroom) repo → [`src/workloads/frontend/`](https://github.com/azure-core/azure-cleanroom/tree/user/ashank/frontendAuth/src/workloads/frontend) on the `user/ashank/frontendAuth` branch — C# models, OpenAPI schema, controllers
> - **`az managedcleanroom` CLI extension source:** [`Azure/azure-cli-extensions`](https://github.com/Azure/azure-cli-extensions) repo → [`src/managedcleanroom/`](https://github.com/Azure/azure-cli-extensions/tree/main/src/managedcleanroom) on `main` — Python CLI commands for collaboration lifecycle, publishing, voting, execution
> - **`az cleanroom` CLI extension source (reference only):** [`azure-core/azure-cleanroom`](https://github.com/azure-core/azure-cleanroom) repo → [`src/tools/azure-cli-extension/cleanroom/`](https://github.com/azure-core/azure-cleanroom/tree/user/ashank/frontendAuth/src/tools/azure-cli-extension/cleanroom) on the `user/ashank/frontendAuth` branch — used to derive the DatasetSpecification JSON structure (`CleanRoomSpecification` model, `AccessPoint`, `PrivacyProxySettings`, `EncryptionSecrets`)



> **New to clean rooms?** The SSE (Server-Side Encryption) native variant is simpler and covers the same workflow. Consider starting there if this is your first clean room deployment.

## Table of Contents <!-- omit from toc -->

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Personas](#personas)
- [Architecture](#architecture)
- [How CPK Differs from SSE](#how-cpk-differs-from-sse)
- [Step 0: Prerequisites setup (all parties)](#step-0-prerequisites-setup-all-parties)
- [Step 1: Create the collaboration (Owner)](#step-1-create-the-collaboration-owner)
- [Step 2: Accept invitation (each collaborator)](#step-2-accept-invitation-each-collaborator)
- [Step 3: Generate demo data](#step-3-generate-demo-data)
- [Step 4: Prepare Azure resources (each collaborator)](#step-4-prepare-azure-resources-each-collaborator)
- [Step 5: Prepare data with encryption (each collaborator)](#step-5-prepare-data-with-encryption-each-collaborator)
- [Step 6: Set up identity & OIDC (each collaborator)](#step-6-set-up-identity--oidc-each-collaborator)
- [Step 7: Grant clean room access (each collaborator)](#step-7-grant-clean-room-access-each-collaborator)
- [Step 8: Publish datasets (each collaborator)](#step-8-publish-datasets-each-collaborator)
- [Step 9: Publish query (Woodgrove)](#step-9-publish-query-woodgrove)
- [Step 10: Vote on query (each collaborator)](#step-10-vote-on-query-each-collaborator)
- [Step 11: Run query (Woodgrove)](#step-11-run-query-woodgrove)
- [Step 12: View results](#step-12-view-results)
- [Advanced Topics](#advanced-topics)
  - [Date range filtering](#date-range-filtering)
  - [Privacy controls](#privacy-controls)
  - [Audit events](#audit-events)

- [Appendix: CLI commands per step](#appendix-cli-commands-per-step)

## Overview

| Aspect | Details |
|--------|---------|
| Encryption | CPK — Data Encryption Key (DEK) wraps data, Key Encryption Key (KEK) wraps DEK |
| Key storage | Azure Key Vault (Premium SKU) |
| Personas | Northwind (publisher), Woodgrove (consumer/query runner) |
| Data format | CSV |
| Query engine | Confidential Spark SQL |
| Data schema | `date:date, time:string, author:string, mentions:string` |
| Output schema | `author:string, Number_Of_Mentions:long, Restricted_Sum:number` |

---

## Prerequisites

- **Azure CLI** 2.75.0+ ([install](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli))
- **PowerShell** 7.x+ ([install](https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell))
- **Azure subscription** with permissions to create storage accounts, Key Vaults (Premium SKU), managed identities, and resource groups
- **Microsoft account** (work, school, or personal) for authentication

> [!NOTE]
> You do **not** need to create new Microsoft accounts — any existing account will work. For testing, you can use two of your own email addresses.

## Personas

| Persona | Role | Description |
|---------|------|-------------|
| **Clean Room Owner** | Collaboration creator | Creates the collaboration, invites collaborators, enables workload. Needs an Azure subscription. Can also be one of the collaborators (e.g., Woodgrove). |
| **Northwind** | Data publisher | Contributes sensitive datasets. Needs an Azure subscription for their own storage. |
| **Woodgrove** | Data consumer & query runner | Contributes datasets, publishes and runs queries, views results. Needs an Azure subscription. |

## Architecture

This variant uses **zero `az cleanroom`** — the DatasetSpecification JSON (including CPK encryption secrets) is constructed natively in PowerShell:

```
Your Environment                              Azure Managed Service
┌──────────────────────────────┐              ┌──────────────────────────────┐
│                              │              │                              │
│  Standard Azure CLI          │              │  az managedcleanroom         │
│  ├─ az storage (upload data) │              │  ├─ collaboration create     │
│  ├─ az keyvault (DEK/KEK)    │              │  ├─ frontend login           │
│  ├─ az identity (OIDC setup) │              │  ├─ frontend API:            │
│  └─ PowerShell (encryption)  │              │  │  ├─ dataset publish       │
│                              │              │  │  ├─ query publish         │
│  PowerShell (native JSON)    │  JSON body   │  │  ├─ vote accept           │
│  ├─ DatasetSpecification     │ ──────────►  │  │  ├─ query run             │
│  │  (with EncryptionSecrets) │              │  │  └─ results/audit         │
│  └─ QuerySpecification       │              │  │                            │
│                              │              │  └─ OIDC issuer              │
│  Your Storage Account        │              │                              │
│  Your Key Vault (Premium)    │              │                              │
│  Your Managed Identity       │              │                              │
└──────────────────────────────┘              └──────────────────────────────┘
```

- **Standard Azure CLI + PowerShell** handles data upload (`azcopy copy --cpk-by-value` with DEK), Key Vault key management (`az keyvault key import` for KEK, `az keyvault secret set` for wrapped DEK), identity setup (`az identity show`), and OIDC configuration. CPK uses **Server-Side Encryption with Customer-Provided Keys (SSE-CPK)** — data is uploaded as plaintext, and Azure Storage encrypts it server-side using the DEK provided via HTTP headers.
- **PowerShell** constructs the DatasetSpecification and QuerySpecification JSON bodies natively. For CPK, the DatasetSpecification includes `EncryptionSecrets` blocks with DEK/KEK references pointing to Key Vault. The structure was derived from the cleanroom extension source code (`CleanRoomSpecification` → `AccessPoint` → `PrivacyProxySettings` → `EncryptionSecrets` model in cleanroom v5.0.0).
- **`az managedcleanroom`** talks to Microsoft's managed service for collaboration lifecycle, publishing, voting, execution, and results.

> [!NOTE]
> 🔴 **ACCR team enhancement request:** If `az managedcleanroom frontend analytics dataset publish` accepted high-level parameters (`--storage-account`, `--encryption-mode CPK`, `--key-vault`, `--schema-format csv`, `--schema-fields "..."`, `--access-mode read`), even the native JSON construction would be unnecessary. This would simplify the workflow further.

### End-to-End Flow

```
  Owner (Woodgrove)              Northwind                        Woodgrove
  ─────────────────              ─────────                        ─────────
         │                            │                                │
  0. Install CLIs              0. Install CLIs                  0. Install CLIs
  1. Create collaboration
     ├─ add-collaborator ────────► email invitation
     ├─ add-collaborator ──────────────────────────────────────► email invitation
     └─ enable-workload
         │                     │                                │
         │               2. Accept invitation              2. Accept invitation
         │                     │                                │
         ▼                     ▼                                ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                    Managed Clean Room RP (ARM)                         │
  │  ┌───────────────────────────────────────────────────────────────┐     │
  │  │  Frontend Service (Analytics API)                             │     │
  │  │  ├─ POST /dataset publish     ◄── datasets from both parties │     │
  │  │  ├─ POST /query publish       ◄── query from Woodgrove       │     │
  │  │  ├─ POST /consent set         ◄── consent from each owner    │     │
  │  │  ├─ POST /vote accept         ◄── votes from both parties    │     │
  │  │  ├─ POST /query run           ◄── execution from Woodgrove   │     │
  │  │  ├─ GET  /query runresult     ◄── polling from Woodgrove     │     │
  │  │  └─ GET  /oidc/keys           ◄── JWKS for OIDC setup        │     │
  │  └───────────────────────────────────────────────────────────────┘     │
  │  ┌───────────────────────────────────────────────────────────────┐     │
  │  │  Consortium (CCF / CGS)                                       │     │
  │  │  Stores: datasets, queries, votes, consent, OIDC issuer       │     │
  │  └───────────────────────────────────────────────────────────────┘     │
  └─────────────────────────────────────────────────────────────────────────┘
         │                     │                                │
         │               ── each party does ──                  │
         │               3. Generate demo data                  │
         │               4. Prepare Azure resources             │
         │                  (incl. Key Vault Premium)           │
          │               5. Upload data with CPK              │
          │                  (azcopy --cpk-by-value)            │
          │                  + store DEK locally                 │
         │               6. OIDC + identity setup               │
         │                     │                                │
         │                     ▼                                ▼
  ┌────────────────────┐  ┌────────────────────┐  ┌─────────────────────────┐
  │ Northwind's Azure  │  │ Woodgrove's Azure  │  │ Clean Room (TEE)        │
  │ ├─ Storage Account │  │ ├─ Storage Account │  │                         │
  │ │  └─ CPK-enc CSV  │  │ │  ├─ CPK-enc CSV  │  │ 1. Hardware attestation │
  │ ├─ Key Vault (HSM) │  │ │  └─ output       │  │ 2. OIDC token from CCF  │
  │ │  ├─ KEK (RSA key)│  │ ├─ Key Vault (HSM) │  │ 3. Azure AD validates   │
  │ │  └─ wrappedDEK   │  │ │  ├─ KEK (RSA key)│  │ 4. SKR release KEK      │
  │ ├─ Managed Identity│  │ │  └─ wrappedDEK   │  │ 5. Unwrap DEK           │
  │ └─ OIDC Storage    │  │ ├─ Managed Identity│  │ 6. CPK headers to       │
  └────────────────────┘  │ └─ OIDC Storage    │  │    Storage (read/write) │
                          └────────────────────┘  │ 7. Run Spark SQL        │
                                                  │ 8. Write results (CPK)  │
                                                  └─────────────────────────┘
                                                               ▲
  7. Grant access (RBAC: Storage + KV Crypto                   │
     + KV Secrets + federated credential)                      │
  8. Publish datasets ─────────► Frontend                      │
     + consent enable            (incl. DEK/KEK secret refs)   │
  9. Publish query ─────────────► Frontend                     │
     + consent enable                                          │
 10. Vote accept ───────────────► Frontend                     │
                                                               │
 11. Run query ──────────────────► Frontend ──────► Clean Room
 12. View results ◄──────────────── Frontend ◄────── Clean Room
```

**How OIDC connects the clean room to your data (Steps 6, 7, and query execution):**

The clean room has no credentials of its own. Instead, each collaborator's **managed identity** is configured to trust tokens signed by the clean room's CCF instance. At runtime, the clean room proves its identity via hardware attestation, gets a signed JWT from CCF, and exchanges it for an Azure AD token that can access your storage and Key Vault.

```
  SETUP (you do this once)                          RUNTIME (Step 11: query execution)
  ────────────────────────                          ──────────────────────────────────

  Step 6: Create OIDC issuer                        Clean Room boots in TEE
  ┌──────────────────────────┐                      ┌──────────────────────────┐
  │ 1. Fetch JWKS (public    │                      │ 1. Intel SGX/SEV hardware│
  │    keys) from CCF:       │                      │    generates attestation │
  │    GET {frontend}/       │                      │    report: "I am genuine │
  │    collaborations/{id}/  │                      │    TEE running approved  │
  │    oidc/keys             │                      │    code"                 │
  │                          │                      │                          │
  │ 2. Create Azure Storage  │                      │ 2. Presents attestation  │
  │    static website and    │                      │    to CCF (consortium)   │
  │    upload JWKS + OpenID  │                      └────────────┬─────────────┘
  │    discovery document    │                                   │
  │                          │                                   ▼
  │    This creates a PUBLIC │                      ┌──────────────────────────┐
  │    URL that Azure AD can │                      │ 3. Sidecar calls CCF:    │
  │    reach to verify JWTs: │                      │    GET /token?tid={..}&  │
  │                          │                      │    sub={..}&aud={..}     │
  │    https://oidcXXX...    │                      │                          │
  │      .z13.web.core.      │                      │ 4. CCF validates         │
  │      windows.net/        │                      │    attestation, then     │
  │      oidc-XXXX           │                      │    reads back the issuer │
  │    ▲                     │                      │    URL YOU registered ───┤
  │    │ THIS is the         │                      │    in Step 6 from its    │
  │    │ "issuer URL"        │                      │    KV store. Sets it as  │
  │    │                     │                      │    the JWT "iss" claim.  │
  │ 3. Register this URL     │                      │                          │
  │    with CGS via REST     │                      │    Priority:             │
  │    POST setIssuerUrl.    │                      │    a. explicit iss param │
  │    ┌─────────────────┐   │                      │    b. per-tenant URL ◄───┤
  │    │ CCF stores it   │   │                      │       (this is ours)     │
  │    │ in its KV store │   │                      │    c. global gov URL     │
  │    │ keyed by your   │───┼──────────────────────│                          │
  │    │ tenant ID       │   │  same URL comes back │ 5. CCF signs JWT:        │
  │    └─────────────────┘   │  at runtime as the   │    iss: https://oidcXXX. │
  │                          │  "iss" claim          │      z13.web.core...    │
  └──────────────────────────┘                      │    sub: {contractId}-    │
                                                    │         {ownerId}        │
  Step 7: Grant access &                            │    aud: api://AzureAD    │
  link YOUR identity                                │         TokenExchange    │
  ┌──────────────────────────┐                      └────────────┬─────────────┘
  │                          │                                   │
  │ 1. RBAC: assign roles    │                                   ▼
  │    on your storage & KV  │                      ┌──────────────────────────┐
  │    to your managed       │                      │ 6. Sidecar calls Azure   │
  │    identity              │                      │    AD token endpoint:    │
  │    (az role assignment   │                      │    - client_id = YOUR    │
  │     create)              │                      │      managed identity's  │
  │                          │                      │      client ID (embedded │
  │ 2. Federated credential: │                      │      in dataset spec     │
  │    az identity           │                      │      from Step 6 → 8)   │
  │    federated-credential  │                      │    - assertion = the JWT  │
  │    create \              │                      │      from step 5 above   │
  │      --issuer $issuerUrl │                      │                          │
  │      --subject $subject  │                      │ 7. Azure AD looks up     │
  │      --audiences "api:// │                      │    that managed identity │
  │       AzureADTokenExchg" │                      │    and checks its        │
  │                          │                      │    federated credentials:│
  │    Where:                │                      │    a. Finds one with     │
  │    issuer = from Step 6  │                      │       matching issuer +  │
  │      (issuer-url.txt)    │                      │       subject + audience │
  │                          │                      │    b. Fetches JWKS from  │
  │    subject = must match  │◄─────────────────────│       the issuer URL ────┤
  │      what CCF puts as    │  Azure AD fetches    │       (your static      │
  │      "sub" in the JWT    │  /.well-known/       │        website)          │
  │    ▲                     │  openid-configuration│    c. Verifies JWT       │
  │    │ ⚠️ GAP: ideally     │  then /openid/v1/jwks│       signature with     │
  │    │ retrieved from the  │  from this URL       │       JWKS public key    │
  │    │ service via "Get    │                      │    d. Checks subject &   │
  │    │ Clean Room Details" │                      │       audience match ◄───┤
  │    │ but no CLI exposes  │                      │       sub in JWT must    │
  │    │ it yet. Script      │                      │       equal --subject    │
  │    │ computes it locally │                      │       in federated cred  │
  │    │ as {contractId}-    │                      │    e. Issues Azure AD    │
  │    │    {userId}         │                      │       access token AS    │
  │    │                     │                      │       your managed       │
  │    │                     │                      │       identity           │
  └──────────────────────────┘                      └────────────┬─────────────┘
                                                                 │
                                                                 ▼
                                                    ┌──────────────────────────┐
                                                    │ 8. Clean room now has    │
  Four things must match:                           │    an Azure AD token     │
  ┌──────────────────────────┐                      │    that IS your managed  │
  │ 1. issuer URL registered │                      │    identity. Uses it to: │
  │    with CGS (Step 6)     │                      │                          │
  │    = "iss" in the JWT    │                      │    ├─ Read your storage  │
  │    = --issuer in fed cred│                      │    │  (RBAC: Blob Data   │
  │                          │                      │    │   Contributor)       │
  │ 2. subject you set in    │                      │    ├─ Unwrap KEK from    │
  │    fed cred (Step 7)     │                      │    │  Key Vault (RBAC:   │
  │    = "sub" in the JWT    │                      │    │  Crypto Officer)     │
  │    (CCF computes this as │                      │    ├─ Read DEK secret    │
  │     {contractId}-        │                      │    │  (RBAC: Secrets     │
  │     {ownerId} internally)│                      │    │   User)             │
  │                          │                      │    └─ Decrypt data in    │
  │ 3. audience in fed cred  │                      │       TEE memory         │
  │    = "aud" in the JWT    │                      └──────────────────────────┘
  │    = api://AzureAD       │
  │      TokenExchange       │
  │                          │
  │ 4. JWKS public keys at   │
  │    the issuer URL must   │
  │    match CCF's signing   │
  │    private key           │
  └──────────────────────────┘
```

## How CPK Differs from SSE

| Aspect | SSE | CPK (this scenario) |
|--------|-----|-----|
| Who encrypts | Azure Storage (automatic) | Azure Storage (server-side, using your key via CPK headers) |
| Key Vault | Not needed for encryption | Required — stores KEK (HSM-backed) + wrapped DEK (secret) |
| Secret stores | Not needed | DEK store + KEK store (registered at Step 8) |
| Data prep (Step 5) | Plain upload via `az storage blob upload-batch` | `azcopy copy --cpk-by-value` (plaintext upload, server-side encryption) |
| Dataset publish (Step 8) | `--encryption-mode SSE` | `--encryption-mode CPK` + secret store params; publish-first flow (publish → fetch SKR policy → create KEK → wrap DEK) |
| Grant access RBAC (Step 7) | Storage only | Storage + Key Vault (Crypto User + Secrets User) |
| Clean room decryption | Reads data directly | SKR releases KEK → unwraps DEK → passes DEK via CPK headers to Azure Storage → transparent read/write |

```
CPK Data Flow (Server-Side Encryption with Customer-Provided Keys):

Upload (your machine):
  plaintext CSV ──► azcopy --cpk-by-value ──► Azure Blob Storage
                         ▲                    (encrypted at rest with your DEK)
                    CPK_ENCRYPTION_KEY env var
                    (base64 of 32-byte DEK)

Key setup (after publish):
  DEK (32 bytes, local) ──► RSA-OAEP-SHA256 wrap with KEK pub key ──► KV Secret
  KEK (RSA-2048, local) ──► az keyvault key import (HSM, exportable, SKR policy) ──► KV Key

Clean Room (in TEE):
  1. Presents OIDC token (from hardware attestation)
  2. Azure AD validates → grants managed identity token
  3. SKR releases KEK private key (attestation verified)
  4. Reads wrapped DEK from Key Vault secret
  5. Unwraps DEK using KEK private key (RSA-OAEP-SHA256)
  6. Passes DEK via CPK headers to Azure Storage (transparent decrypt on read)
  7. Processes Spark SQL query (data in plaintext in TEE memory)
  8. Writes results via CPK headers (transparent encrypt on write)
```

---

## Step 0: Prerequisites setup (all parties)

Install CLI extensions, log in, and set your configuration variables:

```powershell
# 1. Install managed cleanroom extension (public)
az extension add --name managedcleanroom

# 2. az cleanroom extension — NOT required.
#    Step 8 now constructs the DatasetSpecification JSON natively in PowerShell,
#    so the cleanroom CLI extension is no longer needed.
# curl -LO https://github.com/Azure/azure-cleanroom/releases/latest/download/cleanroom-5.0.0-py2.py3-none-any.whl
# az extension add --source ./cleanroom-5.0.0-py2.py3-none-any.whl --allow-preview true -y

# 3. Login to Azure
az login

# 4. If your account has access to multiple tenants, select the correct one
az login --tenant "<tenant-id>"

# 5. Set the target subscription (optional, if you have multiple)
az account set --subscription "<subscription-name-or-id>"

# 6. Verify you are in the correct tenant and subscription
az account show --query "{tenant:tenantId, subscription:name}" -o table
```

Now set the configuration variables for this walkthrough. Change the values to match your environment:

```powershell
# Collaboration
$COLLABORATION_NAME = "jstwitter-cpk-analytics"
$LOCATION           = "eastus2euap"

# Owner (creates the collaboration)
$OWNER_RESOURCE_GROUP = "jscleanroom-cpk-rg"

# Collaboration ARM resource ID (constructed from above; used by frontend commands)
$SUBSCRIPTION_ID  = (az account show --query id -o tsv)
$COLLABORATION_ID = "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$OWNER_RESOURCE_GROUP/providers/Microsoft.CleanRoom/collaborations/$COLLABORATION_NAME"

# Collaborator emails (for invitation)
$NORTHWIND_EMAIL = "jsnorthwind@outlook.com"
$WOODGROVE_EMAIL = "jswoodgrove@outlook.com"

# Northwind (data publisher)
$NORTHWIND_RESOURCE_GROUP = "jsnorthwind-cleanroom-cpk-rg"
$NORTHWIND_PERSONA        = "northwind"

# Woodgrove (data consumer / query runner)
$WOODGROVE_RESOURCE_GROUP = "jswoodgrove-cleanroom-cpk-rg"
$WOODGROVE_PERSONA        = "woodgrove"

# Dataset names (used in publish/query steps)
$NORTHWIND_DATASET_NAME = "northwind-input-csv"
$NORTHWIND_DATASTORE     = "northwinddata"
$WOODGROVE_DATASET_NAME  = "woodgrove-input-csv"
$WOODGROVE_DATASTORE     = "woodgrovedata"
$WOODGROVE_OUTPUT_DATASET = "woodgrove-output-csv"
$QUERY_NAME              = "mentions-analysis"

# Schema
$INPUT_SCHEMA  = "date:date,time:string,author:string,mentions:string"
$OUTPUT_SCHEMA = "author:string,Number_Of_Mentions:long"

# Output directory (generated metadata, temp files)
$OUT_DIR = "./generated"
```

> [!TIP]
> Copy both blocks above into your PowerShell session. All commands below reference these variables.

> [!NOTE]
> Every participant (owner, Northwind, Woodgrove) must run these steps in their own environment.

> [!IMPORTANT]
> **Multi-tenant users:** If your Microsoft account is a guest in multiple Azure AD tenants, you **must** specify the correct tenant via `az login --tenant`. The tenant ID is used during OIDC identity setup (Step 6) and federated credential creation (Step 7). Using the wrong tenant will cause access failures at query execution time.

---

> [!NOTE]
> **Reading this guide — Possibility 1 vs Possibility 2:**
> Some steps below show two approaches:
> - **Possibility 1 (Ideal)** is the **aspirational single-command UX** that the ACCR team should aim for. These commands don't exist yet — they represent what the developer experience _should_ look like.
> - **Possibility 2 (Current)** is the **working approach today**, using the provided scripts.
>
> **Step 8 (Publish datasets)** currently requires constructing DatasetSpecification JSON natively (with CPK encryption secrets). All P1/P2 blocks are UX improvements to `az managedcleanroom` itself.

## Step 1: Create the collaboration (Owner)

The clean room owner creates the collaboration and invites both parties. **This step is identical to SSE.**

> **Possibility 1 (Ideal)** — Single command that creates, invites, and enables workload:
> ```powershell
> az managedcleanroom collaboration create `
>     --collaboration-name $COLLABORATION_NAME `
>     --resource-group $OWNER_RESOURCE_GROUP `
>     --location $LOCATION `
>     --consortium-type ConfidentialACI `
>     --members "$NORTHWIND_EMAIL,$WOODGROVE_EMAIL" `
>     --workload-type analytics
> ```
> **Value add:** Today the script must make 3 separate CLI calls: create the collaboration, add each collaborator by email, and enable the analytics workload. In the ideal UX, a single command handles creation, invitation, and workload enablement — reducing 3 calls to 1.

**Possibility 2 (Current)** — Using the provided script:

```powershell
./scripts/01-setup-collaboration.ps1 `
    -collaborationName $COLLABORATION_NAME `
    -resourceGroup $OWNER_RESOURCE_GROUP `
    -collaboratorEmails @($NORTHWIND_EMAIL, $WOODGROVE_EMAIL) `
    -location $LOCATION
```

This script:
1. Gets the owner's identity (tenant ID + object ID) from the current `az login` session
2. Creates the collaboration (`az managedcleanroom collaboration create` with `--consortium-type ConfidentialACI` and `--user-identity`)
3. Adds each collaborator by email (`add-collaborator --email`)
4. Enables the analytics workload (`enable-workload --workload-type analytics`)
5. Outputs the **Collaboration ARM ID** and **Frontend Endpoint**

> [!IMPORTANT]
> Note the **Collaboration ARM ID** and **Frontend Endpoint** from the output — all collaborators need both for subsequent steps. The ARM ID is used as `$COLLABORATION_ID` and the Frontend Endpoint is the analytics API URL (`workloads[0].endpoint` — the CLI flattens `properties` via `client_flatten`).

> [!WARNING]
> **Resolved — Frontend endpoint discovery.** Source code analysis ([`Azure/azure-cli-extensions`](https://github.com/Azure/azure-cli-extensions/tree/main/src/managedcleanroom) → `aaz/latest/managedcleanroom/collaboration/_show.py`) confirms: `az managedcleanroom collaboration show` returns `workloads[]` where each workload has `endpoint`, `namespace`, and `workloadType` fields (the ARM `properties` are flattened by `client_flatten`). The frontend URL is at `workloads[0].endpoint` (for `workloadType: "analytics"`). Our `01-setup-collaboration.ps1` script extracts and displays this automatically.
>
> 🔴 **ACCR team: Please document the `collaboration show` response schema in CLI help text.** The `workloads[].endpoint` field is not mentioned in `--help`.

> [!WARNING]
> **Question — Why is `--user-identity` required?** The `collaboration create` command requires an explicit `--user-identity {tenant-id:...,object-id:...,account-type:...}` parameter, but this information is already available from the logged-in session — tenant ID via `az account show` and object ID via `az ad signed-in-user show`. Moreover, the tenant ID is already set at session level (via `az login --tenant`), making the `tenant-id` field in `--user-identity` redundant. Our script auto-derives it, but the CLI should do this internally. The only scenario where an explicit `--user-identity` might be needed is when creating a collaboration on behalf of a different identity (e.g., a service principal or another user) — but even then, it should be an **optional override** that defaults to the logged-in user. Additionally, there is no validation that the supplied identity matches the caller — supplying a different identity would register someone else as the owner, with errors only surfacing in later steps.
>
> 🔴 **ACCR team: Can `--user-identity` be made optional, defaulting to the logged-in user?**

> [!WARNING]
> **Confirmed gap — No collaborator management commands.** Source code analysis ([`Azure/azure-cli-extensions`](https://github.com/Azure/azure-cli-extensions/tree/main/src/managedcleanroom) → `aaz/latest/managedcleanroom/collaboration/`) confirms only `_add_collaborator.py` exists — there are no `list-collaborators`, `show-collaborator`, or `remove-collaborator` commands. This means there is no way to verify who has been added, check if a collaborator was added successfully, or remove a collaborator added by mistake. Our scripts must use try-and-catch to handle idempotent re-runs (detecting "already added" from error text) instead of a clean pre-check.
>
> 🔴 **ACCR team: Can `list-collaborators` and `remove-collaborator` commands be added?**

> [!CAUTION]
> **Untested — error patterns for idempotency are guessed.** This script assumes:
> - The error text from `add-collaborator` when re-adding matches `"already exists|already added|Conflict"`
> - The error text from `enable-workload` when re-enabling matches `"already enabled|already exists|Conflict"`
>
> The `.id` field from `collaboration show` and the `workloads[].endpoint` field have been confirmed from source code. Error patterns cannot be determined from CLI source code — the CLI uses standard `MgmtErrorFormat` and passes through whatever the ARM RP (`Microsoft.CleanRoom`) returns. Needs live validation.
>
> 🔴 **ACCR team: Please document the error responses for `add-collaborator` and `enable-workload` when called with duplicate inputs (e.g., re-adding the same collaborator).**

---

## Step 2: Accept invitation (each collaborator)

Each collaborator accepts their invitation to join the collaboration. **Identical to SSE.**

> **Possibility 1 (Ideal)** — Single command that configures, authenticates, and accepts:
> ```powershell
> az managedcleanroom frontend invitation accept `
>     --endpoint "<frontend-endpoint>" `
>     --collaboration-id $COLLABORATION_ID
> ```
> **Value add:** Today the script must configure the frontend endpoint, list pending invitations to discover the invitation ID, then accept it — 3 separate commands. In the ideal UX, a single command handles endpoint setup and acceptance (auto-discovering the pending invitation).

**Possibility 2 (Current)** — Using the provided script:

```powershell
# Run by each collaborator (Northwind and Woodgrove)
./scripts/02-accept-invitation.ps1 `
    -collaborationId $COLLABORATION_ID `
    -frontendEndpoint "<frontend-endpoint>"
```

This script:
1. Configures the frontend endpoint (`az managedcleanroom frontend configure --endpoint`)
2. Triggers interactive device code authentication (`az managedcleanroom frontend login`)
3. Lists pending invitations to discover the invitation ID (`az managedcleanroom frontend invitation list --collaboration-id`)
4. Parses the response to extract the invitation ID
5. Accepts the invitation (`az managedcleanroom frontend invitation accept --collaboration-id --invitation-id`)

> [!CAUTION]
> **Confirmed — response schema and idempotency validated from source code.** The CLI help does not document the `invitation list` response schema or the `invitation accept` idempotency behavior. These were discovered by reading the frontend service source code (`src/workloads/frontend/Controllers/CollaborationController.cs` on the `user/ashank/frontendAuth` branch):
> - `invitation list` returns `{ "invitations": [{ "invitationId": "..." }, ...] }` — a wrapper object, **not** a bare array ✅
> - Each invitation uses **`.invitationId`** (not `.id`) ✅
> - `invitation accept` is **inherently idempotent** — the controller runs a state machine (`Open` → `Accepted` → `Finalized`) and each transition catches Conflict/PreconditionFailed gracefully. Re-calling accept on a finalized invitation returns `200 OK` immediately ✅
> - `invitation list` returns invitations regardless of status (no status filter) — so re-runs will still find the invitation and accept will succeed as a no-op ✅
>
> 🔴 **ACCR team: Please document `invitation list` response schema and `invitation accept` idempotency behavior in the CLI help.**

> [!WARNING]
> **Note — `frontend login` vs `az login` for frontend authentication.** The frontend CLI supports two auth methods: (1) MSAL device code flow via `frontend login`, and (2) regular `az login` credentials. In production, where each collaborator uses a single corporate identity for both ARM and frontend, `az login` alone is sufficient — no separate `frontend login` needed. Our scripts use `frontend login` because we simulate multiple parties from one machine: `az login` is the corporate account (for Azure resources) while `frontend login` authenticates the personal account (matching the invited email). If you're running this with one identity per party, you can skip `frontend login`.

---

## Step 3: Generate demo data

> [!NOTE]
> **Demo only.** This step generates sample Twitter CSV data for testing. In a real scenario, each collaborator would already have their own data — skip this step and point Step 5 at your existing files.

Each collaborator downloads their own sample Twitter CSV dataset:

```powershell
# Run by Northwind
./demos/generate-data.ps1 -persona $NORTHWIND_PERSONA

# Run by Woodgrove
./demos/generate-data.ps1 -persona $WOODGROVE_PERSONA
```

This downloads CSV files from the [Azure Synapse samples repository](https://github.com/Azure-Samples/Synapse/tree/main/Data/Tweets):
- **Northwind** gets: `RahulPotharajuTweets.csv`, `raghurwiTweets.csv`, `MikeDoesBigDataTweets.csv`, `SQLCindyTweets.csv`
- **Woodgrove** gets: `BrigitMurtaughTweets.csv`, `FranmerMSTweets.csv`, `JeremyLiknessTweets.csv`, `mwinkleTweets.csv`

---

## Step 4: Prepare Azure resources (each collaborator)

Each collaborator provisions Azure resources. **Unlike SSE, CPK requires a Premium Key Vault** for storing encryption keys.

Using the provided script:

```powershell
# Run by Northwind
./scripts/04-prepare-resources.ps1 `
    -resourceGroup $NORTHWIND_RESOURCE_GROUP `
    -location $LOCATION `
    -outDir $OUT_DIR

# Run by Woodgrove
./scripts/04-prepare-resources.ps1 `
    -resourceGroup $WOODGROVE_RESOURCE_GROUP `
    -location $LOCATION `
    -outDir $OUT_DIR
```

This creates:
- A **resource group** (if not existing)
- A **storage account** for data (Azure Blob Storage)
- A **Key Vault (Premium SKU)** for KEK/DEK storage — required for CPK key management (HSM-backed keys with Secure Key Release)
- A **managed identity** (User-Assigned) for clean room access
- **RBAC assignments** for the logged-in user (Storage Blob Data Contributor, Key Vault Crypto Officer, Key Vault Secrets Officer)

The script runs 6+ Azure CLI commands (`az group create`, `az storage account create`, `az keyvault create --sku Premium`, `az identity create`, `az role assignment create` × 3).

---

## Step 5: Prepare data with encryption (each collaborator)

> [!IMPORTANT]
> **CPK is Server-Side Encryption, NOT Client-Side.** A common mistake is to manually encrypt
> files (e.g., AES-CBC) before uploading. This is **wrong** and results in double-encrypted
> data that the cleanroom cannot read. CPK mode means you upload **plaintext** files using
> `azcopy copy --cpk-by-value`, and Azure Storage encrypts them **server-side** using the
> DEK you provide via HTTP headers. At read time, the caller provides the same DEK via CPK
> headers and Azure Storage decrypts transparently.

> [!NOTE]
> In a real customer scenario, customers likely already have encrypted data in Azure Storage with keys in Key Vault. This step demonstrates the upload pipeline for the demo.

Each collaborator generates a DEK and uploads their data using azcopy with CPK encryption.

```powershell
# Run by Northwind
./scripts/05-prepare-data-cpk.ps1 `
    -resourceGroup $NORTHWIND_RESOURCE_GROUP `
    -persona $NORTHWIND_PERSONA `
    -dataDir "./demos/datasource/northwind/input/csv" `
    -outDir $OUT_DIR

# Run by Woodgrove
./scripts/05-prepare-data-cpk.ps1 `
    -resourceGroup $WOODGROVE_RESOURCE_GROUP `
    -persona $WOODGROVE_PERSONA `
    -dataDir "./demos/datasource/woodgrove/input/csv" `
    -outDir $OUT_DIR
```

This script uses **only standard Azure CLI + azcopy** (no `az cleanroom`):
1. Resolves the storage account URL and Key Vault URL (`az storage account show`, `az keyvault show`)
2. Creates blob containers for input data (and output for Woodgrove) (`az storage container create`)
3. Generates a random **32-byte DEK** (Data Encryption Key) — saved to `generated/datastores/keys/<dataset>-dek.bin`
4. Uploads **plaintext** data files using `azcopy copy --cpk-by-value` with env vars:
   - `CPK_ENCRYPTION_KEY` = base64(DEK)
   - `CPK_ENCRYPTION_KEY_SHA256` = base64(SHA256(DEK))
   - Azure Storage encrypts the data server-side using the DEK
5. Saves datastore metadata (schema, storage URL, container name, dataset suffix) to a JSON file for Step 8

> [!NOTE]
> **KEK creation and DEK wrapping happen in Step 8 (publish-first flow)**, not here.
> Step 5 only generates the DEK and uploads data. The KEK is created after publishing
> because the SKR release policy is fetched from the just-published dataset.

---

## Step 6: Set up identity & OIDC (each collaborator)

Each collaborator sets up OIDC issuer infrastructure and saves identity metadata.

> **Possibility 1 (Ideal)** — If the managed RP auto-provisions OIDC issuer:
> ```powershell
> # No separate step needed — issuer provisioned at collaboration create time,
> # identity registered automatically during invitation accept
> ```
> **Value add:** Today each collaborator must manually create an OIDC issuer (Azure Storage static website with JWKS + openid-configuration), then register it with the collaboration. In the ideal UX, the managed RP provisions OIDC infrastructure at collaboration create time and registers identities during invitation accept — eliminating this step entirely.

**Possibility 2 (Current)** — Using the provided script:

```powershell
# Run by Northwind
./scripts/06-setup-identity.ps1 `
    -resourceGroup $NORTHWIND_RESOURCE_GROUP `
    -persona $NORTHWIND_PERSONA `
    -collaborationId $COLLABORATION_ID `
    -frontendEndpoint $FRONTEND_ENDPOINT `
    -outDir $OUT_DIR

# Run by Woodgrove
./scripts/06-setup-identity.ps1 `
    -resourceGroup $WOODGROVE_RESOURCE_GROUP `
    -persona $WOODGROVE_PERSONA `
    -collaborationId $COLLABORATION_ID `
    -frontendEndpoint $FRONTEND_ENDPOINT `
    -outDir $OUT_DIR
```

This script uses **only standard Azure CLI** and `az managedcleanroom`:
1. **Creates an OIDC storage account** — a separate Azure Storage account with static website hosting enabled, used to serve the OpenID Connect discovery documents publicly
2. **Fetches OIDC issuer metadata** via `az managedcleanroom frontend oidc issuerinfo show` — returns `{ enabled, issuerUrl, tenantData: { issuerUrl, tenantId } }` (issuer metadata, **not** JWKS). Note: this response schema is not documented in `--help`; it was confirmed from source code ([`Azure/azure-cli-extensions`](https://github.com/Azure/azure-cli-extensions/tree/main/src/managedcleanroom) → `analytics_frontend_api/operations/_operations.py`). 🔴 **ACCR team: Please document the `oidc issuerinfo show` response schema in CLI help.**
3. **Fetches JWKS from the frontend** via direct REST call to `GET {frontendEndpoint}/collaborations/{id}/oidc/keys` — returns `{ keys: [{ kty, kid, alg, n, e, ... }] }`. No CLI command wraps this endpoint — confirmed from source code ([`Azure/azure-cli-extensions`](https://github.com/Azure/azure-cli-extensions/tree/main/src/managedcleanroom) — only `oidc issuerinfo show` exists).
   🔴 **ACCR team: Please add `az managedcleanroom frontend oidc keys show` CLI command to replace this direct REST call.**
4. **Uploads two discovery documents** to the static website's `$web` container:
   - `/.well-known/openid-configuration` — the OpenID discovery document that tells Azure AD where to find the JWKS
   - `/openid/v1/jwks` — the JWKS fetched from the frontend in step 3
5. **Registers the issuer URL** with CGS via direct REST call to `POST /collaborations/{id}/oidc/setIssuerUrl` — tells the clean room where to find the OIDC documents. No CLI command wraps this endpoint — confirmed from source code ([`Azure/azure-cli-extensions`](https://github.com/Azure/azure-cli-extensions/tree/main/src/managedcleanroom) — only `oidc issuerinfo show` exists).
   🔴 **ACCR team: Please add `az managedcleanroom frontend oidc set-issuer-url` CLI command to replace this direct REST call.**
6. **Reads the OIDC issuer URL** (the static website URL + container path) and saves it to a file for use in Step 7 (federated credential creation)
7. **Gets the managed identity's client ID and tenant ID** via `az identity show`
8. **Saves identity metadata to a JSON file** (client ID, tenant ID, issuer URL) — the identity registration with the collaboration happens implicitly when the DatasetSpecification (which embeds the identity) is published at Step 8

> [!CAUTION]
> **OIDC architecture — two possible approaches.** Source code analysis (`src/workloads/frontend/` on `user/ashank/frontendAuth` branch) and the [POC architecture docs](https://github.com/azure-core/azure-cleanroom/blob/develop/poc/managed-cleanroom/configure-workload.md) show two possible OIDC models for managed cleanroom:
>
> **Current script approach (Option B — user-hosted OIDC):**
> - Creates a static website to host OIDC discovery documents
> - Fetches JWKS from the frontend via **direct REST call** to `GET /collaborations/{id}/oidc/keys` (no CLI command wraps this endpoint yet)
> - Uploads `openid-configuration` and `jwks` to the static website
> - Registers the static website URL with CGS via **direct REST call** to `POST /collaborations/{id}/oidc/setIssuerUrl` (no CLI command wraps this endpoint yet)
>
> **Alternative approach (Option A — CCF-as-IDP):**
> - The `issuerUrl` from `oidc issuerinfo show` may already point to CCF's own OIDC endpoint
> - CCF serves `.well-known/openid-configuration` and JWKS at `/app/oidc/keys` directly
> - Users would only need to set the federated credential with this issuer URL — no storage account, no static website, no JWKS upload
> - If this is the intended model, Step 6 could be simplified significantly
>
> 🔴 **ACCR team: (1) Is Option A (CCF-as-IDP) or Option B (user-hosted) the correct approach? (2) Please add `az managedcleanroom frontend oidc keys show` CLI command to replace the direct REST call. (3) If Option A, the `issuerUrl` from `oidc issuerinfo show` should be documented as the federated credential issuer.**

> [!NOTE]
> **Why OIDC?** The clean room needs to prove (via hardware attestation) that it's running approved code before accessing your data. The OIDC issuer validates that attestation and issues a JWT token. Your managed identity has a federated credential that trusts this issuer, so Azure AD accepts the token and grants the clean room the RBAC permissions you assigned in Step 7.

---

## Step 7: Grant clean room access (each collaborator)

Each collaborator grants the clean room workload access. **Unlike SSE, CPK also requires Key Vault RBAC** (Crypto Officer + Secrets User) so the clean room can unwrap encryption keys.

Using the provided script:

```powershell
# Run by Northwind
./scripts/07-grant-access.ps1 `
    -resourceGroup $NORTHWIND_RESOURCE_GROUP `
    -collaborationId $COLLABORATION_ID `
    -contractId "Analytics" `
    -userId "<northwind-user-id>" `
    -setupKeyVault `
    -outDir $OUT_DIR

# Run by Woodgrove
./scripts/07-grant-access.ps1 `
    -resourceGroup $WOODGROVE_RESOURCE_GROUP `
    -collaborationId $COLLABORATION_ID `
    -contractId "Analytics" `
    -userId "<woodgrove-user-id>" `
    -setupKeyVault `
    -outDir $OUT_DIR
```

This script:
1. Reads the OIDC issuer URL from Step 6 output
2. Computes the federation subject string (`{contractId}-{userId}`)
3. Resolves the managed identity's principal ID (`az identity show`)
4. Resolves the storage account resource ID (`az storage account show`)
5. Assigns **Storage Blob Data Owner** on the storage account (`az role assignment create`)
6. Resolves the Key Vault resource ID (`az keyvault show`)
7. Assigns **Key Vault Crypto User** on the Key Vault (`az role assignment create`) — for KEK unwrap operations via SKR
8. Assigns **Key Vault Secrets User** on the Key Vault (`az role assignment create`) — for wrapped DEK retrieval
9. Creates a **federated credential** on the managed identity (`az identity federated-credential create --issuer --subject --audiences "api://AzureADTokenExchange"`)

> [!NOTE]
> The `userId` is the collaborator's object ID in the CCF governance. For MSA (personal Microsoft) accounts, this is the `oid` claim from the MSAL IdToken (JWT), **NOT** the Graph API object ID from `az ad signed-in-user show`. Extract it from the token:
> ```bash
> cat /tmp/msal-idtoken.txt | cut -d. -f2 | python3 -c "import sys,base64,json; b=sys.stdin.read(); print(json.loads(base64.urlsafe_b64decode(b+'=='))['oid'])"
> ```
> The `contractId` defaults to `"Analytics"` (capital A) — **this is case-sensitive**. The Spark agent uses `"Analytics"` as the contract ID in the federated credential subject (`Analytics-{ownerId}`). Using lowercase `"analytics"` will cause silent token exchange failures at query execution time.
>
> **Important**: For CPK mode, always pass the `-setupKeyVault` switch so the managed identity gets `Key Vault Crypto User` and `Key Vault Secrets User` RBAC roles. Without these, the cleanroom cannot release the KEK or read the wrapped DEK.

> [!CAUTION]
> **Partially confirmed — subject format and audience validated from source code, but retrieval is a gap.** Analysis of the Spark analytics agent (`src/workloads/analytics/cleanroom-spark-analytics-agent/Controllers/QueriesController.cs` on `user/ashank/frontendAuth` branch):
> - **Subject format** `{contractId}-{ownerId}` — confirmed at line 355: `string.Join("-", inputJob.ContractId, dataset.OwnerId)` ✅
> - **Audience** `api://AzureADTokenExchange` — confirmed at line 410: `var aud = "api://AzureADTokenExchange"` ✅
> - **RBAC roles** `Storage Blob Data Owner` + `Key Vault Crypto Officer` + `Key Vault Secrets User` — not confirmed from source (RBAC is configured on the Azure AD side, not in the clean room code). May need different role combinations for CPK key unwrapping ❓
>
> **Gap — `cleanroom_identity` is not exposed to users.** The [POC architecture docs](https://github.com/azure-core/azure-cleanroom/blob/develop/poc/managed-cleanroom/configure-workload.md) show that users should **retrieve** the `cleanroom_identity` from the service (via "Get Clean Room Details" → CGS returns `cleanroom_identity` → user sets federation). This `cleanroom_identity` is the `{contractId}-{ownerId}` subject computed internally by the analytics agent. However:
> - `az managedcleanroom frontend oidc issuerinfo show` returns `{ enabled, issuerUrl, tenantData }` — **no `cleanroom_identity` field**
> - No other CLI command or frontend API endpoint exposes this value
> - The `contractId` and `ownerId` are internal service values — users shouldn't have to guess them
> - Our script computes the subject locally (which matches the runtime format), but this is fragile — if the internal convention changes, the federation will break silently
>
> 🔴 **ACCR team: Please expose the federated credential subject (`cleanroom_identity`) via a CLI command or include it in the `oidc issuerinfo show` response, so users don't need to reverse-engineer the `{contractId}-{ownerId}` format.**
> If any of these don't match what the clean room actually presents at runtime, the token exchange will fail silently (Azure AD rejects the token). Will be validated once we can test end-to-end.

---

## Step 8: Publish datasets (each collaborator)

Each collaborator publishes their dataset metadata to the collaboration.

> **Possibility 1 (Ideal)** — Single command per dataset:
> ```powershell
> # Northwind publishes their input dataset (read-only)
> az managedcleanroom frontend analytics dataset publish `
>     --collaboration-id $COLLABORATION_ID `
>     --dataset-name $NORTHWIND_DATASET_NAME `
>     --storage-account $NORTHWIND_DATASTORE `
>     --encryption-mode CPK `
>     --key-vault "northwind-kv" `
>     --access-mode read `
>     --schema-format csv `
>     --schema-fields $INPUT_SCHEMA
>
> # Woodgrove publishes their input dataset (read-only)
> az managedcleanroom frontend analytics dataset publish `
>     --collaboration-id $COLLABORATION_ID `
>     --dataset-name $WOODGROVE_DATASET_NAME `
>     --storage-account $WOODGROVE_DATASTORE `
>     --encryption-mode CPK `
>     --key-vault "woodgrove-kv" `
>     --access-mode read `
>     --schema-format csv `
>     --schema-fields $INPUT_SCHEMA
>
> # Woodgrove also publishes the output dataset (write — query results land here)
> az managedcleanroom frontend analytics dataset publish `
>     --collaboration-id $COLLABORATION_ID `
>     --dataset-name $WOODGROVE_OUTPUT_DATASET `
>     --storage-account $WOODGROVE_DATASTORE `
>     --encryption-mode CPK `
>     --key-vault "woodgrove-kv" `
>     --access-mode write `
>     --schema-format csv `
>     --schema-fields $OUTPUT_SCHEMA
> ```
> **Value add:** Today publishing requires constructing a deeply nested DatasetSpecification JSON body natively in PowerShell, including CPK encryption secret chains (DEK/KEK). In the ideal UX, a single command accepts storage, encryption, and schema details directly — no JSON construction needed.

**Possibility 2 (Current)** — Using the provided script:

```powershell
# Run by Northwind
./scripts/08-publish-dataset-cpk.ps1 `
    -collaborationId $COLLABORATION_ID `
    -resourceGroup $NORTHWIND_RESOURCE_GROUP `
    -persona $NORTHWIND_PERSONA `
    -outDir $OUT_DIR

# Run by Woodgrove
./scripts/08-publish-dataset-cpk.ps1 `
    -collaborationId $COLLABORATION_ID `
    -resourceGroup $WOODGROVE_RESOURCE_GROUP `
    -persona $WOODGROVE_PERSONA `
    -outDir $OUT_DIR
```

This script constructs the **DatasetSpecification JSON** natively in PowerShell (no `az cleanroom` dependency) and uses a **publish-first flow**:

**Phase 1 — Publish datasets:**
1. Reads datastore metadata from Step 5 (`generated/datastores/{persona}-datastore-metadata.json` — storage URL, container, schema) and identity metadata from Step 6 (`generated/{resourceGroup}/identity-metadata.json` — client ID, tenant ID, OIDC issuer URL)
2. Builds the DatasetSpecification JSON — a nested structure containing schema, access policy, storage access point (`AccessPoint`), identity references, and protection settings (`PrivacyProxySettings`) **with CPK `EncryptionSecrets`** (DEK/KEK store references pointing to Key Vault). The structure was derived from the cleanroom extension source code (cleanroom v5.0.0 `CleanRoomSpecification` model)
3. Publishes via `az managedcleanroom frontend analytics dataset publish --body @file`

**Phase 2 — Create KEKs + wrap DEKs (per dataset):**
4. Fetches the **SKR release policy** from the just-published dataset via the frontend's `skr-policy` endpoint — this contains the collaboration's `ccePolicyHash` (attestation hash)
5. Generates an **RSA-2048 KEK** locally using Python `cryptography` library
6. Imports the KEK to Key Vault via `az keyvault key import` with `--exportable true --protection hsm --policy <skr-policy.json>` — NOT `az keyvault key create`
7. **Wraps the DEK** with the KEK's public key using **RSA-OAEP-SHA256** (client-side, Python `cryptography` lib) — NOT `az keyvault key encrypt` (server-side wrapping doesn't work for CPK flow)
8. Stores the **wrapped DEK** as a Key Vault secret (`az keyvault secret set`)

**Phase 3 — Enable consent:**
9. **Enables execution consent** on each published dataset via `az managedcleanroom frontend consent set --consent-action enable` — required for the clean room to access the data at query run time
10. Verifies the dataset state

> [!NOTE]
> **Why publish-first?** The SKR release policy contains the collaboration's attestation hash (`ccePolicyHash`). This is only available from a published dataset via the frontend's `skr-policy` endpoint. By publishing first (before creating the KEK), we don't need to hardcode or pre-fetch this hash.

> [!NOTE]
> **DatasetSpecification format verified against frontend source.** The JSON body matches `DatasetInputDetails` (`src/workloads/frontend/Models/CGS/DatasetInputDetails.cs`): `{ "data": { name, datasetSchema, datasetAccessPolicy, datasetAccessPoint } }`. The `AccessPoint.protection.encryptionSecrets` structure (DEK/KEK with `backingResource` pointing to Key Vault) matches the cleanroom `EncryptionSecrets` model. The frontend performs **no field-level validation** — it passes the AccessPoint through to CGS verbatim (`DatasetDocumentPublisher.cs:73-84`). Publishing will succeed, but the cleanroom runtime may fail if any derived values are wrong.
>
> **Specific runtime risks (see `.NOTES` in script for details):**
> - ~~`tokenIssuer.url` is set to `https://cgs/oidc` (v5.0.0 symbolic reference) — old samples used the actual OIDC issuer URL~~ **FIXED**: `issuerUrl`/`tokenIssuer.url` must be the **public OIDC URL** (e.g., `https://cleanroomoidc.z22.web.core.windows.net/{collaborationId}`), NOT `"https://cgs/oidc"`. Using the internal CGS hostname causes AAD to reject with `AADSTS700211: No matching federated identity record found for presented assertion issuer`. The script now reads from `generated/{rg}/issuer-url.txt`.
> - `backingResource.id` for DEK/KEK uses logical names (`$persona-dek-store`) — if CGS resolves these against a registry, they'd fail since we didn't register via `secretstore add`
> - `protection.configuration` uses Python-style single-quoted dict — if CGS parses as JSON, use double quotes
> - `store.id` uses the datastore name — if CGS expects the ARM resource ID, change to `storeId` from metadata
>
> 🔴 **ACCR team: Please document the expected `dataset publish --body` JSON schema in the CLI help, so users don't need to read source code to verify compatibility.**

> [!CAUTION]
> **Untested — idempotency relies on `dataset show` behavior.** The script checks if a dataset is already published by calling `dataset show` and assuming it returns a non-zero exit code when the dataset doesn't exist, and a JSON response when it does. If `dataset show` behaves differently (e.g., returns an empty response or a 200 with an error object), the idempotency check may not work correctly.

**Northwind** publishes 1 dataset:
- `northwind-input-csv` — their input data (read-only access)

**Woodgrove** publishes 2 datasets:
- `woodgrove-input-csv` — their input data (read-only access)
- `woodgrove-output-csv` — the output dataset where query results are written (write access). Woodgrove owns the output because they are the query proposer.

---

## Step 9: Publish query (Woodgrove)

Woodgrove proposes a Spark SQL query that combines both datasets. The query itself is identical to SSE — encryption mode doesn't affect the query.

```powershell
./scripts/09-publish-query.ps1 `
    -collaborationId $COLLABORATION_ID `
    -queryName $QUERY_NAME `
    -queryDir "./demos/query/woodgrove/query1" `
    -publisherInputDataset $NORTHWIND_DATASET_NAME `
    -consumerInputDataset $WOODGROVE_DATASET_NAME `
    -outputDataset $WOODGROVE_OUTPUT_DATASET `
    -outDir $OUT_DIR
```

This script uses **only PowerShell + `az managedcleanroom`**:
1. Reads query segment files (`segment1.txt`, `segment2.txt`, `segment3.txt`) directly from disk
2. Builds the query specification JSON in PowerShell (segments with execution sequence, input/output dataset mappings)
3. Checks if the query is already published via `query show` (skips publish if so — idempotent)
4. Publishes via `az managedcleanroom frontend analytics query publish --body @file`
5. **Enables execution consent** on the query via `az managedcleanroom frontend consent set --consent-action enable` — the query publisher enables consent at publish time; this can be done before other collaborators vote
6. Verifies the query state

> [!CAUTION]
> **Untested** — idempotency relies on `query show` behavior. The script assumes `query show` returns a non-zero exit code when a query doesn't exist, and a JSON response when it does. If `query show` behaves differently (e.g., returns an empty response or a 200 with an error object), the idempotency check may not work correctly. Will be validated once we can test end-to-end.

> [!CAUTION]
> **Confirmed — query body JSON structure.** Validated against the frontend service source code
> (`src/workloads/frontend/Models/CGS/` in the `azure-core/azure-cleanroom` repo):
> - `queryData.segments[]` — each segment has `executionSequence` (int), `data` (string), and optional `preConditions`/`postFilters` arrays
> - `inputDatasets[]` — each entry has `view` (string) and `datasetDocumentId` (string)
> - `outputDataset` — same shape as `inputDatasets[]` entries: `{view, datasetDocumentId}`
> - `contractId` — not part of the client-facing schema (managed internally by the server in `CreateUserDocument.cs`)
>
> 🔴 **ACCR team: Please document the `query publish --body` JSON schema in the CLI help, including segment fields and dataset input structure.**

> [!WARNING]
> **CLI gap** — `az managedcleanroom` has no commands for building query segments or generating the query body. However, `az cleanroom` (non-public) **does** have `spark-sql publish --prepare-only` which generates a compatible query body (`src/tools/azure-cli-extension/cleanroom/azext_cleanroom/collaboration_cmd.py:694`). Its output includes `{ contractId, queryData, inputDatasets, outputDataset }` — note the extra `contractId` field which the frontend ignores (not part of `QueryInputDetails`). Our scripts hand-craft the JSON instead (confirmed correct against source — see CAUTION above).
>
> 🔴 **ACCR team: Can `az managedcleanroom` add equivalent query body generation commands (e.g., `query segment add`, `query publish --prepare-only`)?**

The query runs three SQL segments:
1. `CREATE OR REPLACE TEMP VIEW publisher_view AS SELECT * FROM publisher_data`
2. `CREATE OR REPLACE TEMP VIEW consumer_view AS SELECT * FROM consumer_data`
3. `SELECT author, COUNT(*) AS Number_Of_Mentions, SUM(mentions) AS Restricted_Sum FROM (SELECT * FROM publisher_view UNION ALL SELECT * FROM consumer_view) ...`

---

## Step 10: Vote on query (each collaborator)

**Both** collaborators must vote to approve the query before it can execute.

```powershell
# Run by Northwind
./scripts/10-vote-query.ps1 `
    -collaborationId $COLLABORATION_ID `
    -queryName $QUERY_NAME

# Run by Woodgrove (the proposer also votes)
./scripts/10-vote-query.ps1 `
    -collaborationId $COLLABORATION_ID `
    -queryName $QUERY_NAME
```

The script performs 2 CLI calls per collaborator:
1. Votes to accept the query (`az managedcleanroom frontend analytics query vote accept`) — no `--body` needed per CLI help; idempotent, skips if already voted
2. Verifies query state via `query show`

> [!CAUTION]
> **Partially confirmed — vote idempotency still uncertain.** Source code analysis (`src/workloads/frontend/Controllers/CollaborationController.cs:202-222` on `user/ashank/frontendAuth` branch):
> - `VoteRequest` has a `proposalId` field, but it's **optional** in the OpenAPI schema (`proposalId?: string`) — our script correctly omits it ✅
> - The controller calls `VoteDocumentProposalAsync` which posts to CGS at `/userdocuments/{documentId}/vote_accept` — no try-catch for Conflict/duplicate votes ❓
> - Unlike `invitation accept` (which has a state machine with Conflict handling), vote accept has **no duplicate-vote handling** — if CGS rejects a re-vote, the error propagates unhandled
> - The script's `Invoke-AzIdempotent` error-text matching (`"already voted|accepted|approved"`) is the only safety net — but the actual CGS error message for duplicate votes is still unknown
>
> Needs live testing to confirm whether CGS silently accepts or rejects duplicate votes.
>
> 🔴 **ACCR team: Please document `vote accept` idempotency behavior (what happens on duplicate votes) and the `VoteRequest` body schema in the CLI help.**

> [!WARNING]
> **CLI gap** — `vote accept --help` does not document whether voting twice returns an error or is silently accepted. The [POC architecture docs](https://github.com/azure-core/azure-cleanroom/blob/develop/poc/managed-cleanroom/spark/README.md) confirm that document approvers accept/reject directly — no `proposalId` is needed, consistent with the CLI help showing `--body` is optional.

After both votes, the query state changes from "Proposed" to "Accepted".

---

## Step 11: Run query (Woodgrove)

Execute the approved query in the confidential clean room:

> **Possibility 1 (Ideal)** — Single command:
> ```powershell
> az managedcleanroom frontend analytics query run `
>     --collaboration-id $COLLABORATION_ID `
>     --query-name $QUERY_NAME `
>     --wait
> ```
> **Value add:** Today the script must submit the query, extract a job ID from the response, then poll `runresult show` in a loop until completion. In the ideal UX, a single command with `--wait` blocks until the job completes — no polling loop needed.

**Possibility 2 (Current)** — Using the provided script (submits + polls for completion):

```powershell
./scripts/11-run-query.ps1 `
    -collaborationId $COLLABORATION_ID `
    -queryName $QUERY_NAME
```

The script:
1. Submits the query run via `az managedcleanroom frontend analytics query run`
2. Polls for completion using `az managedcleanroom frontend analytics query runresult show` every 15 seconds (configurable via `-pollIntervalSeconds`)
3. Times out after 30 minutes (configurable via `-timeoutMinutes`)
4. Displays the final status (COMPLETED or FAILED)

> [!NOTE]
> This step is intentionally **not idempotent** — each invocation submits a new query run. Re-running the script will start a fresh execution, which is valid (you may want to re-run a query with updated data).

> [!CAUTION]
> **Confirmed — response schemas validated from source code.** Analysis of the frontend service (`src/workloads/frontend/` on `user/ashank/frontendAuth` branch):
> - **`query run` response** (`QueryRunOutput` in OpenAPI schema): field is `jobId` (string) — our script's primary extraction `$runResponse.jobId` is correct ✅
> - **`runresult show` response** (`QueryRunResult` in `Models/CGS/QueryRunResult.cs`): nesting is `status.applicationState.state` — our script's primary path `$result.status.applicationState.state` is correct ✅
> - **State values** (from OpenAPI enum): `"SUBMITTED"`, `"RUNNING"`, `"COMPLETED"`, `"FAILED"`, `"SUBMISSION_FAILED"`, `"PENDING_RERUN"`, `"INVALIDATING"`, `"SUCCEEDING"`, `"FAILING"`, `"SUSPENDING"`, `"SUSPENDED"`, `"RESUMING"`, `"UNKNOWN"` — our script checks `"COMPLETED"` and `"FAILED"`/`"SUBMISSION_FAILED"` which are correct ✅
>
> 🔴 **ACCR team: Please document `query run` and `runresult show` response schemas in the CLI help, including the `jobId` field, `status.applicationState.state` nesting, and valid state enum values.**

---

## Step 12: View results

View run history and audit events:

```powershell
./scripts/12-view-results.ps1 `
    -collaborationId $COLLABORATION_ID `
    -queryName $QUERY_NAME
```

The script calls two native CLI commands directly — no gap between ideal and current UX:

```powershell
az managedcleanroom frontend analytics query runhistory list --collaboration-id $COLLABORATION_ID --document-id $QUERY_NAME
az managedcleanroom frontend analytics auditevent list --collaboration-id $COLLABORATION_ID
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
    --collaboration-id $COLLABORATION_ID `
    --document-id $QUERY_NAME `
    --body $runBody
```

The frontend API accepts `startDate` and `endDate` as ISO 8601 strings in the run request body. Only data within the specified range is loaded into the clean room.

> [!NOTE]
> Date filtering is applied at the data-loading stage — the clean room only mounts partitions matching the range. This is more efficient than filtering in the SQL query itself.

### Privacy controls

The sample query includes privacy-preserving mechanisms built into the query segments:

- **Pre-conditions** (`segment1.txt`, `segment2.txt`): Each segment includes a `minRowCount` check. If either dataset has fewer rows than the threshold, the query aborts — preventing analysis on too-small datasets where individual records could be re-identified.

- **Post-filters** (`segment3.txt`): The final aggregation query filters output to only include groups with `Number_Of_Mentions >= 2`, ensuring no single-author results are exposed.

These controls are defined in the query segments under `demos/query/woodgrove/query1/`. You can customise the thresholds by editing the segment files before publishing the query (Step 9).

### Audit events

View all audit events for the collaboration directly via the CLI:

```powershell
az managedcleanroom frontend analytics auditevent list `
    --collaboration-id $COLLABORATION_ID
```

Events include query execution start/completion, input/output row counts, dataset access, and failures. The audit log is maintained in the CCF ledger and is tamper-proof.

---

## Appendix: CLI commands per step

| Step | Description | CLI Extension | CLI Commands Used |
|------|-------------|---------------|-------------------|
| 0 | Prerequisites | `managedcleanroom` | `az extension add`, `az login` |
| 1 | Create collaboration | `managedcleanroom` | `az account show`, `az ad signed-in-user show`, `az managedcleanroom collaboration create`, `add-collaborator` (×2), `enable-workload` |
| 2 | Accept invitation | `managedcleanroom` | `az managedcleanroom frontend configure`, `login`, `invitation list`, `invitation accept` |
| 3 | Generate demo data | _(none)_ | PowerShell only — downloads CSV files from GitHub |
| 4 | Prepare resources | _(standard az)_ | `az group create`, `az storage account create`, `az keyvault create --sku Premium`, `az identity create`, `az role assignment create` (×3) |
| 5 | Prepare data (CPK) | _(standard az)_ | `az storage account show`, `az keyvault show`, `az storage container create`, `azcopy copy --cpk-by-value` (upload plaintext with DEK via CPK headers) |
| 6 | Identity & OIDC | `managedcleanroom` + _(standard az)_ + _(REST)_ | `az storage account create`, `az storage blob service-properties update`, `az role assignment create`, `az managedcleanroom frontend oidc issuerinfo show`, `GET /collaborations/{id}/oidc/keys` _(direct REST — no CLI)_, `az storage blob upload` (×2), `POST /collaborations/{id}/oidc/setIssuerUrl` _(direct REST — no CLI)_, `az identity show` |
| 7 | Grant access (CPK) | _(standard az)_ | `az identity show`, `az storage account show`, `az keyvault show`, `az role assignment create` (×3: Storage + KV Crypto User + KV Secrets User), `az identity federated-credential create`. **Note**: Use `-setupKeyVault` switch for CPK mode |
| **8** | **Publish datasets** | `managedcleanroom` | PowerShell `New-DatasetBody` (builds DatasetSpecification JSON natively with CPK `EncryptionSecrets`), `az managedcleanroom frontend analytics dataset publish --body @file`, fetch SKR policy, Python `create-kek.py` (RSA-2048 local gen + `az keyvault key import --exportable --protection hsm --policy`), Python `generate-wrapped-dek.py` (RSA-OAEP-SHA256 client-side wrap), `az keyvault secret set` (wrapped DEK), `az managedcleanroom frontend consent set --consent-action enable` (per dataset) |
| 9 | Publish query | `managedcleanroom` | PowerShell JSON assembly, `az managedcleanroom frontend analytics query publish --body @file`, `az managedcleanroom frontend consent set --consent-action enable` |
| 10 | Vote on query | `managedcleanroom` | `az managedcleanroom frontend analytics query show`, `query vote accept --body` |
| 11 | Run query | `managedcleanroom` | `az managedcleanroom frontend analytics query run`, `query runresult show` (polling) |
| 12 | View results | `managedcleanroom` | `az managedcleanroom frontend analytics query runhistory list`, `auditevent list` |
| 13 | Run status | _(REST)_ | `GET /collaborations/{id}/analytics/runs/{jobId}` via `frontend-helpers.ps1` — polls for COMPLETED/FAILED |
| 14 | Run history | _(REST)_ | `GET /collaborations/{id}/analytics/queries/{docId}/runs` via `frontend-helpers.ps1` — standalone with 404 handling |
| 15 | Audit events | _(REST)_ | `GET /collaborations/{id}/analytics/auditevents` via `frontend-helpers.ps1` — standalone with `value[]` wrapper parsing |

> **No `az cleanroom` dependency.** All steps use standard Azure CLI or the publicly available `az managedcleanroom` extension. Step 8 constructs the DatasetSpecification JSON (with CPK encryption secrets) natively in PowerShell.
