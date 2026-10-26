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

> **Destructive clean setup:** If the named collaboration already exists, this
> script deletes it, waits until ARM reports it absent, and creates a fresh one.
> It does not reuse or resume the old collaboration. Its existing datasets,
> queries, governance state, and managed infrastructure are not retained.
> A deletion error or timeout stops setup before creation.
> Use `-DryRun` to preview, or `manage-collaboration.ps1` for actions that should
> keep an existing collaboration.

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
ARM uses `2026-09-30-preview`; frontend calls use `2026-03-01-preview`.
The only currently supported sizing is `Standard_D4ds_v5` with **3 nodes**.
Both the script and Bicep reject larger configurations until support is expanded.
Quota and regional capacity remain prerequisites, not guarantees made by these scripts.

### Timeouts

`-TimeoutMinutes` defaults to **60 minutes per lifecycle operation**, not for the
entire script. Delete, create, enable/readiness, and each collaborator addition
have separate budgets. A complete clean setup can therefore take over an hour.
The budget starts before submitting each operation; creation uses
`az deployment group create --no-wait` and polls both deployment and collaboration
state, so it does not hide a blocking creation wait before starting the timer.
Recovery tracks the ARM operation URL, not `provisioningState`, which may remain
`Succeeded` while recovery is still in progress.

The polling deadline is checked between Azure CLI requests; it is not a hard
process-kill timeout on an individual network request. On timeout, Azure work
may still be running: inspect its status before retrying. Fast setup stops before
creation if deletion fails or times out. Query monitoring has its own independent
60-minute default and does not cancel or resubmit a query on timeout.

### Optional target resource configuration

To pass additional creation-time settings such as public IP tags, add a
`targetResourceConfiguration` hashtable to `$deployParams` before invoking the
deployment script:

```powershell
$deployParams.targetResourceConfiguration = @{
    iPTagConfiguration = @{
        type = "FirstPartyUsage"
        value = "/AzureCleanRoomsNonProd"
    }
}
.\scripts\bicep\deploy-managed-cleanroom.ps1 @deployParams -DryRun
```

Use only IP-tag values approved for your subscription and scenario. These are
networking IP tags, not ordinary ARM resource tags. Use ARM property names,
including the case-sensitive `iPTagConfiguration` spelling, not CLI shorthand
such as `i-p-tag-configuration`.

The script merges this object with `aksSku` and `nodePoolSize`. If those fields
also appear in the object, the object values take precedence; they must still
be `Standard_D4ds_v5` and `3`. With the object omitted, existing
behavior is unchanged (`Standard_D4ds_v5`, 3 nodes by default).

`-DryRun` prints the effective configuration without deploying; it reads the
current Azure CLI account unless `-Subscription` is supplied. Remove `-DryRun`
to create the collaboration.
The script passes the object through a temporary JSON file, which is removed
even if deployment fails. Direct Bicep deployments can also supply the optional
`targetResourceConfiguration` object parameter.

These settings apply to the freshly created collaboration, including when
replacing an existing one. The script always performs delete -> wait for absence
-> create -> enable Analytics -> wait for health -> add additional collaborators.
If no collaboration exists, it goes directly to creation.
`-DeleteExistingCollab` remains accepted for compatibility but is already true
by default; explicitly passing `-DeleteExistingCollab:$false` is rejected.
After recreation, repeat Phases 2 onward using the new frontend collaboration UUID.

For creation-time collaborators, pass `-Collaborators @("<email>")`.
`-AdditionalCollaborators` adds participants after the new collaboration is ready.
To add participants without deleting an existing collaboration, use the
standalone management commands below instead of rerunning fast setup.

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
$EncryptionMode = "SSE"         # SSE or CPK
$iteration = 1                 # increase for each new dataset iteration
$suffix = if ($EncryptionMode -eq "CPK") { "-cpk-v$iteration" } else { "-v$iteration" }

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
$variant = $EncryptionMode.ToLowerInvariant()
.\scripts\05-prepare-data.ps1 -resourceGroup $personaRg -variant $variant -persona $persona `
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
    -contractId "Analytics" -userId $personaOid -EncryptionMode $EncryptionMode

if ($persona -eq "woodgrove") {
    .\scripts\08-build-dataset-body.ps1 -resourceGroup $personaRg -persona $persona `
        -subdirectory "2025-09-01"
} else {
    .\scripts\08-build-dataset-body.ps1 -resourceGroup $personaRg -persona $persona
}
```

For CPK, install AzCopy first. The upload helper generates the DEKs and uploads
with Azure Storage CPK encryption; do not manually encrypt the input files.
Access setup grants the workload its Key Vault permissions.
Prepare the KEKs **after publishing** in Phase 4, not before.
Use a distinct dataset suffix for each iteration and a separate output directory
for each query run. See README-API.md Appendix C for the key-release architecture.

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

if ($EncryptionMode -eq "CPK") {
    .\scripts\08-prepare-dataset-keys.ps1 -resourceGroup $personaRg -persona $persona `
        -collaborationId $collabId -frontendEndpoint $fe.Frontend
}
```

Key preparation uses the explicit `-TokenFile` when supplied, then the frontend
environment tokens, then the per-persona temp token file. If you used a custom
token-file location for other steps, pass that same file here.
Publish all datasets before retrieving their SKR policies.

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
$scaleSku = "small"
$jobId = .\scripts\frontend\run-query.ps1 -Persona woodgrove -CollaborationId $collabId `
    -QueryName $queryName -ScaleSku $scaleSku -TimeoutMinutes 60

# The orchestrator returns the job ID; isolate this run's output.
.\scripts\11-download-output.ps1 -resourceGroup $personaRg -datasetSuffix $suffix `
    -JobId $jobId -OutputDir ".\generated\output\$EncryptionMode-$jobId"
```

Re-run an accepted query by repeating Phase 7 — no re-setup needed.

Both `run-query.ps1` and `09-run-query.ps1` accept `-ScaleSku small` (default),
`medium`, or `large`; this controls the Spark execution profile, not the AKS
node-pool SKU. They also accept `-StartDate` / `-EndDate`. Check the quota and
profile details in README-API.md Step 09 before selecting a larger profile.
`large` remains preview-only, not production-ready. For date filtering:

```powershell
$jobId = .\scripts\frontend\run-query.ps1 -Persona woodgrove -CollaborationId $collabId `
    -QueryName $queryName -ScaleSku medium -StartDate "2025-09-01" -EndDate "2025-09-02"
```

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

The frontend step/orchestrator scripts, Bicep deployment/management wrappers,
and `common/add-app-collaborator.ps1` support `-DryRun`. Their switch prints
planned requests instead of sending them; ARM wrappers read the current Azure
CLI account unless `-Subscription` is supplied. Resource provisioning, upload, OIDC storage,
access, body-building, key-preparation, download and Grafana helpers do **not**
support this switch. Body builders write local files even though they do not
call the service.

A local preview does not validate authentication, dataset contents, approval
state or effective `skuSettings`; the run helpers return only a `<job-id>`
placeholder. For an actual service-side dry run, both query runners accept `-ServiceDryRun`:

```powershell
$preview = .\scripts\frontend\run-query.ps1 -Persona woodgrove -CollaborationId $collabId `
    -QueryName $queryName -ScaleSku medium -ServiceDryRun
$preview.skuSettings | ConvertTo-Json -Depth 10
```

This sends `dryRun: true` but does not publish, execute, monitor, or fetch results.
It cannot be combined with `-DryRun` or the orchestrator's `-BodyFile`.

## Inspect, diagnose, cancel, and change consent

```powershell
$readArgs = @{ Persona = $persona; CollaborationId = $collabId }
.\scripts\frontend\get-frontend-data.ps1 @readArgs -Resource Collaborators
.\scripts\frontend\get-frontend-data.ps1 @readArgs -Resource Datasets
.\scripts\frontend\get-frontend-data.ps1 @readArgs -Resource Dataset `
    -DocumentId "$persona-input-csv$suffix"
.\scripts\frontend\get-frontend-data.ps1 @readArgs -Resource Queries
.\scripts\frontend\get-frontend-data.ps1 @readArgs -Resource Query -QueryName $queryName

# One detailed snapshot, even for a stuck/failed run:
$result = .\scripts\frontend\10-monitor-query.ps1 @readArgs -JobId $jobId -Once
$result.events | Format-Table type, reason, message, count -Wrap
$result.status.applicationState | ConvertTo-Json -Depth 5

# Cancel only a persistent failure/unwanted run, not a single early FailedMount.
.\scripts\frontend\09-cancel-query.ps1 @readArgs -QueryName $queryName -JobId $jobId

# Enable or disable consent without republishing.
.\scripts\frontend\06-set-consent.ps1 @readArgs `
    -DocumentId "$persona-input-csv$suffix" -ConsentAction disable
```

The reader also supports `Collaborations` (`-ActiveOnly`), `Invitations`
(`-PendingOnly`), `OidcKeys`, `Run`, `RunHistory`, and `AuditEvents`.
Polling prints new/updated events, stops on failure/cancellation, and times out
after 60 minutes by default. A timeout **does not cancel or resubmit** the run.
The quota/capacity/network troubleshooting tables in the main guides still apply;
a generic executor failure alone does not establish a capacity problem.

## Manage an existing collaboration and retrieve diagnostics

These are ARM operations, using the selected Azure subscription or an explicit
`-Subscription`. Mutations wait by default; `-NoWait` submits only.

```powershell
$armArgs = @{ ResourceGroup = $collabRg; CollaborationName = $collabName }
$collab = .\scripts\bicep\manage-collaboration.ps1 @armArgs -Action Show
$collab.properties.health | ConvertTo-Json -Depth 10
.\scripts\bicep\manage-collaboration.ps1 @armArgs -Action EnableWorkload
.\scripts\bicep\manage-collaboration.ps1 @armArgs -Action AddCollaborator `
    -Collaborators @("<northwind-email>")

.\scripts\frontend\12-get-readonly-kubeconfig.ps1 @armArgs -OutFile ".\readonly.kubeconfig"
.\scripts\12-open-grafana-dashboard.ps1 -KubeConfigPath ".\readonly.kubeconfig"
```

Use recovery only as a last resort; deletion permanently removes the collaboration
and its associated resources. Neither is an automatic response to a query failure:

```powershell
.\scripts\bicep\manage-collaboration.ps1 @armArgs -Action Recover -ForceRecover
# Delete only (does NOT recreate):
.\scripts\bicep\manage-collaboration.ps1 @armArgs -Action Delete
```

## SPN participants

Acquire the SPN token with `common/get-sp-token-sni.ps1` as described in the main
guides and extract its JWT `oid` for the FIC subject. ARM operations use the
current Azure CLI login; setting a frontend token does not change that login.
For collaborator registration, use all three identity fields:

```powershell
$app = @{
    userIdentifier = "<app-client-id>"
    objectId = "<enterprise-application-object-id>"
    tenantId = "<tenant-id>"
}
.\scripts\bicep\manage-collaboration.ps1 @armArgs -Action AddCollaborator -Collaborators @($app)
# The same object is accepted by deploy-managed-cleanroom.ps1:
# -Collaborators @($app) at creation, or -AdditionalCollaborators @($app).
```

`common/add-app-collaborator.ps1` also accepts `-AppClientId`, `-ObjectId` and
`-TenantId`. SPNs auto-activate and need no invitation acceptance.

## Offline validation

These target-configuration tests mock Azure CLI; they do not deploy or run queries:

```powershell
.\scripts\bicep\tests\test-target-resource-configuration.ps1
```
