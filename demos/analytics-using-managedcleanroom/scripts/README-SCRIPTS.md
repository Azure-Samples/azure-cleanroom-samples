# Big Data Analytics — Fast Setup Scripts (alternate scripting solution)

This guide is an **alternate, scripting-first path** for setting up a managed
clean-room analytics collaboration and running queries.

For the step-by-step variants of the same scenario, see
[README-API.md](../README-API.md) (REST via `Invoke-RestMethod` + `az rest`) and
[README-CLI.md](../README-CLI.md) (`az managedcleanroom`). This doc wraps those same
operations into per-persona orchestrators.

- **Owner** = Woodgrove (creates the collaboration, publishes + runs queries)
- **Collaborator** = Northwind (joins, contributes data, approves queries)

Run every command from the `demos/analytics-using-managedcleanroom/` directory,
including commands that invoke scripts in subdirectories.

Choose one scenario:

| Scenario | Setup participants | Query | Approval |
|---|---|---|---|
| **Single-owner (default)** | Woodgrove only | `query1`: both input views use Woodgrove's dataset | Publishing casts Woodgrove's vote; no Northwind vote |
| **Cross-party** | Woodgrove and Northwind, in separate authenticated terminals | `query2`: Northwind and Woodgrove datasets | Northwind votes after Woodgrove publishes |

---

> Prerequisites: Azure CLI 2.75+, PowerShell 7+, `az login` per persona, and the
> quota noted in [README-API.md](../README-API.md) Step 1.1. Acquire a per-persona
> frontend token first (README Step 1.5) or set `$env:CLEANROOM_FRONTEND_TOKEN`.
> Each terminal must use its own participant's token. No `managedcleanroom`
> extension is required by this scripting path.

---

## Phase 1 — Owner creates the collaboration (control plane)

```powershell
$collabName = "collab1"
$deployParams = @{
    resourceGroup = "cr-collab-rg"
    collaborationName = $collabName
    location = "westus"
    resourceLocation = "westus"
    aksSku = "Standard_D4ds_v5"
    nodePoolSize = 3
}
# Cross-party only: uncomment and replace with Northwind's actual email.
# $deployParams.additionalCollaborators = @("<northwind-email>")
.\scripts\bicep\deploy-managed-cleanroom.ps1 @deployParams
```

This is Bicep (declarative collaboration resource) + the ARM *action* steps
(`enableWorkload`, `addCollaborator`). Runtime ~35 min. See [scripts/bicep/](bicep/).

---

## Phase 2 — Each participant resolves the collaboration and joins

Run this before fetching OIDC information or publishing datasets. For the
single-owner scenario only Woodgrove needs Phases 2–4. For the cross-party
scenario both participants run them in their own terminals.

```powershell
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$persona   = "woodgrove"          # or "northwind"
$personaRg = "cr-e2e-$persona-rg"
$collabName = "collab1"           # name chosen by the owner in Phase 1
$suffix = "-v1"                  # use a new suffix for a new dataset iteration

. .\scripts\frontend\Invoke-Frontend.ps1
$fe = Get-FrontendContext -Persona $persona
$collabId = Resolve-CollaborationId -Context $fe -CollaborationName $collabName
.\scripts\frontend\03-accept-invitation.ps1 -Persona $persona -CollaborationId $collabId

# Use this participant's JWT oid for the federated credential.
$payload = $fe.Token.Split('.')[1].Replace('-', '+').Replace('_', '/')
$payload += '=' * ((4 - $payload.Length % 4) % 4)
$claims = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($payload)) | ConvertFrom-Json
$personaOid = $claims.oid
if (-not $personaOid) { throw "The frontend token must contain an oid claim." }
```

The acceptance helper returns without accepting again if the participant is
already active (including the owner).

---

## Phase 3 — Each participant provisions resources, data and OIDC

```powershell
.\scripts\04-prepare-resources.ps1 -resourceGroup $personaRg -persona $persona -location westus
.\demos\generate-data.ps1 -persona $persona
.\scripts\05-prepare-data.ps1 -resourceGroup $personaRg -variant sse -persona $persona `
    -dataDir ".\generated\datasource\$persona\csv" -datasetSuffix $suffix
.\scripts\frontend\05-fetch-jwks.ps1 -Persona $persona -CollaborationId $collabId `
    -outDir "generated\$personaRg"

$oidcParams = @{
    resourceGroup = $personaRg
    persona = $persona
    collaborationId = $collabId
    JwksFile = "generated\$personaRg\jwks.json"
}
# If tenant policy requires a whitelisted OIDC account, set its approved URL:
# $oidcParams.OidcStorageUrl = "https://<account>.z22.web.core.windows.net"
.\scripts\06-setup-oidc-storage.ps1 @oidcParams
.\scripts\frontend\05-set-issuer-url.ps1 -Persona $persona -CollaborationId $collabId `
    -outDir "generated\$personaRg"
.\scripts\07-grant-access.ps1 -resourceGroup $personaRg -collaborationId $collabId `
    -contractId "Analytics" -userId $personaOid -EncryptionMode SSE

if ($persona -eq "woodgrove") {
    .\scripts\08-build-dataset-body.ps1 -resourceGroup $personaRg -persona $persona `
        -subdirectory "2025-09-01"
} else {
    .\scripts\08-build-dataset-body.ps1 -resourceGroup $personaRg -persona $persona
}
```

For CPK encryption instead, follow the CPK upload, access, and key-preparation
steps in [README-API.md](../README-API.md). Use a distinct dataset suffix for
each iteration and a separate output directory for each query run.

---

## Phase 4 — Each participant publishes datasets

Publish all query inputs and the owner's output dataset **before** Phase 5.
For cross-party queries, wait until both participants have completed this phase.

```powershell
.\scripts\frontend\06-publish-dataset.ps1 -Persona $persona -CollaborationId $collabId `
    -DocumentId "$persona-input-csv$suffix" -BodyFile "generated\publish\$persona-input-dataset.json"
if ($persona -eq "woodgrove") {
    .\scripts\frontend\06-publish-dataset.ps1 -Persona $persona -CollaborationId $collabId `
        -DocumentId "woodgrove-output-csv$suffix" -BodyFile "generated\publish\woodgrove-output-dataset.json"
}
```

---

## Phase 5 — Owner builds and publishes the chosen query

**Option A — Single-owner query1** (no Northwind data or vote):

```powershell
$queryName = "query1$suffix"
.\scripts\09-build-query-body.ps1 -queryName $queryName `
    -queryDir ".\demos\query\woodgrove\query1" `
    -publisherInputDataset "woodgrove-input-csv$suffix" `
    -consumerInputDataset "woodgrove-input-csv$suffix" `
    -outputDataset "woodgrove-output-csv$suffix"
```

**Option B — Cross-party query2** (use Northwind's exact published dataset ID;
their suffix may differ):

```powershell
$northwindDataset = "<northwind-input-csv-suffix>"
$queryName = "query2$suffix"
.\scripts\09-build-query-body.ps1 -queryName $queryName `
    -queryDir ".\demos\query\woodgrove\query2" `
    -publisherInputDataset $northwindDataset `
    -consumerInputDataset "woodgrove-input-csv$suffix" `
    -outputDataset "woodgrove-output-csv$suffix"
```

After choosing **one** option, publish its body:

```powershell
.\scripts\frontend\07-publish-query.ps1 -Persona woodgrove -CollaborationId $collabId `
    -QueryName $queryName -BodyFile "generated\publish\$queryName.json"
```

Publishing casts the publisher's accept vote; remaining affected collaborators
approve in Phase 6 before execution.

---

## Phase 6 — Northwind approves query2 (cross-party only)

Skip this phase for single-owner query1. For query2, send its exact name to
Northwind, who runs this **after publication**, using their existing terminal:

```powershell
$queryName = "<query2-name-from-woodgrove>"
.\scripts\frontend\run-collaborator.ps1 -Persona northwind -CollaborationId $collabId `
    -QueryName $queryName -SkipAccept
```

No dataset is published here: it was already published before the query was
proposed. The helper fetches the proposal ID and casts Northwind's vote.

---

## Phase 7 — Owner runs the query and downloads output

In Woodgrove's terminal, verify that the chosen query is accepted, then run:

```powershell
$queryInfo = Invoke-Frontend -Context $fe -Path "$collabId/analytics/queries/$queryName"
if ($queryInfo.state -ne "Accepted") { throw "Wait for the required query approvals before running." }

# Run + wait for completion + print run history/audit
.\scripts\frontend\run-query.ps1 -Persona woodgrove -CollaborationId $collabId `
    -QueryName $queryName -ScaleSku small

# Copy the job ID printed by the orchestrator; isolate this run's output.
$jobId = "<job-id-printed-above>"
.\scripts\11-download-output.ps1 -resourceGroup $personaRg -datasetSuffix $suffix `
    -JobId $jobId -OutputDir ".\generated\output\SSE-$jobId"
```

Re-run an accepted query by repeating Phase 7 — no re-setup needed.

Both `run-query.ps1` and `09-run-query.ps1` accept `-ScaleSku small` (default),
`medium`, or `large`; this controls the Spark execution profile, not the AKS
node-pool SKU. They also accept `-StartDate` / `-EndDate`. Check the quota and
profile details in README-API.md Step 09 before selecting a larger profile.

A failed or submission-failed run causes the monitor and orchestrator to throw.
If submission returns no job ID, check run history before retrying rather than
assuming no run was scheduled.

---

## Cheat sheet

| Persona | Phase | Commands |
|---------|-------|--------------|
| Owner | 1 Create | `deploy-managed-cleanroom.ps1` |
| Each participant | 2 Join | `Resolve-CollaborationId` → `03-accept-invitation.ps1` |
| Each participant | 3 Resources + OIDC | Resource, upload, OIDC, access and body-building helpers |
| Each participant | 4 Publish datasets | `06-publish-dataset.ps1` (Woodgrove publishes input + output) |
| Owner | 5 Publish query | `09-build-query-body.ps1` → `07-publish-query.ps1` |
| Northwind | 6 Approve query2 only | `run-collaborator.ps1 -SkipAccept` |
| Owner | 7 Execute + download | `run-query.ps1` → `11-download-output.ps1` |

### Local request previews versus service dry runs

Only the frontend step/orchestrator scripts and
`scripts/bicep/deploy-managed-cleanroom.ps1` support `-DryRun`. Their switch
prints planned requests instead of sending them; the ARM wrappers still read
the current Azure CLI account. Resource provisioning, upload, OIDC storage,
access, body-building, key-preparation, download and Grafana helpers do **not**
support this switch. Body builders write local files even though they do not
call the service.

A local preview does not validate authentication, dataset contents, approval
state or effective `skuSettings`; the run helpers return only a `<job-id>`
placeholder. For an actual service-side dry run, use the
`dryRun: true` request body in README-API.md Step 09; that sends a frontend
request but does not execute the Spark query.
