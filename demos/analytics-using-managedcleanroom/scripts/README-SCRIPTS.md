# Big Data Analytics — Fast Setup Scripts (alternate scripting solution)

This guide is an **alternate, scripting-first path**: opinionated PowerShell
orchestrators that give each persona the **minimum uber-level commands** to stand
up a managed clean-room analytics collaboration. The goal: after these steps, the
**only** remaining action is running queries — and that is scripted too.

For the step-by-step variants of the same scenario, see
[README-API.md](../README-API.md) (REST via `Invoke-RestMethod` + `az rest`) and
[README-CLI.md](../README-CLI.md) (`az managedcleanroom`). This doc wraps those same
operations into per-persona orchestrators.

- **Owner** = Woodgrove (creates the collaboration, publishes + runs queries)
- **Collaborator** = Northwind (joins, contributes data, approves queries)

All paths are relative to `demos/analytics-using-managedcleanroom/`.

---

> Prerequisites: Azure CLI 2.75+, PowerShell 7+, `az login` per persona, and the
> quota noted in [README-API.md](../README-API.md) Step 1.1. Acquire a per-persona
> frontend token first (README Step 1.5) or set `$env:CLEANROOM_FRONTEND_TOKEN`.

---

## Phase 1 — Owner creates the collaboration (control plane)

```powershell
# Deploy collaboration + enable Analytics workload + invite Northwind (one command).
./scripts/bicep/deploy-managed-cleanroom.ps1 `
    -resourceGroup cr-collab-rg -collaborationName collab1 `
    -resourceLocation westus `
    -additionalCollaborators northwind@contoso.com
```

This is Bicep (declarative collaboration resource) + the ARM *action* steps
(`enableWorkload`, `addCollaborator`). Runtime ~35 min. See [scripts/bicep/](bicep/).

---

## Phase 2 — Each persona provisions resources + OIDC (setup)

Run per persona (identical commands, only `$persona` differs). These are the
existing helper scripts; they create storage/KV/MI and wire OIDC.

```powershell
$persona   = "woodgrove"          # or "northwind"
$personaRg = "cr-e2e-$persona-rg"
$collabId  = "<frontend-collaboration-uuid>"   # from Phase 3 for collaborators

./scripts/04-prepare-resources.ps1 -resourceGroup $personaRg -persona $persona -location westus
./demos/generate-data.ps1          -persona $persona
./scripts/05-prepare-data.ps1      -resourceGroup $personaRg -variant sse -persona $persona -dataDir "./generated/datasource/$persona/csv" -datasetSuffix "-v1"
# OIDC (needs $collabId from Phase 3)
./scripts/frontend/05-fetch-jwks.ps1 -Persona $persona -CollaborationId $collabId -outDir "generated/$personaRg"
./scripts/06-setup-oidc-storage.ps1  -resourceGroup $personaRg -persona $persona -collaborationId $collabId -JwksFile "generated/$personaRg/jwks.json"
./scripts/frontend/05-set-issuer-url.ps1 -Persona $persona -CollaborationId $collabId -outDir "generated/$personaRg"
./scripts/07-grant-access.ps1        -resourceGroup $personaRg -collaborationId $collabId -contractId "Analytics" -userId $personaOid
./scripts/08-build-dataset-body.ps1  -resourceGroup $personaRg -persona $persona
```

---

## Phase 3 — Collaborator joins + contributes + approves (one command)

Once the owner has published a query, each collaborator runs a single
orchestrator to accept the invite, (optionally) publish their dataset, and vote:

```powershell
# Northwind: accept invite + publish input dataset + approve the query
./scripts/frontend/run-collaborator.ps1 `
    -Persona northwind -QueryName query1-v1 `
    -DatasetDocumentId northwind-input-csv-v1 `
    -DatasetBodyFile generated/publish/northwind-input-dataset.json
```

If already onboarded and the query is published, this is literally:

```powershell
./scripts/frontend/run-collaborator.ps1 -Persona northwind -QueryName query1-v1
```

See [scripts/frontend/run-collaborator.ps1](frontend/run-collaborator.ps1).

---

## Phase 4 — Owner publishes the query (once)

```powershell
# Build the query body (existing helper), then publish
./scripts/09-build-query-body.ps1 -queryName query1-v1 `
    -queryDir "./demos/query/woodgrove/query1" `
    -publisherInputDataset woodgrove-input-csv-v1 `
    -consumerInputDataset  woodgrove-input-csv-v1 `
    -outputDataset         woodgrove-output-csv-v1

./scripts/frontend/run-query.ps1 -Persona woodgrove -QueryName query1-v1 `
    -BodyFile generated/publish/query1-v1.json -SkipMonitor -SkipResults
```

`-SkipMonitor -SkipResults` because collaborators still need to approve (Phase 3)
before the query can actually run.

---

## Phase 5 — Run the query (scripted, repeatable)

After the query is approved by all collaborators, running it is one command:

```powershell
# Run + wait for results + print run history/audit
./scripts/frontend/run-query.ps1 -Persona woodgrove -QueryName query1-v1

# Download the output CSVs
./scripts/11-download-output.ps1 -resourceGroup cr-e2e-woodgrove-rg -datasetSuffix "-v1"
```

Re-run any time by repeating Phase 5 — no re-setup needed.

---

## Cheat sheet

| Persona | Phase | Uber command |
|---------|-------|--------------|
| Owner | 1 Create | `./scripts/bicep/deploy-managed-cleanroom.ps1 ...` |
| Both  | 2 Provision + OIDC | `04-prepare-resources` → OIDC helpers |
| Collaborator | 3 Join + approve | `./scripts/frontend/run-collaborator.ps1 -Persona <p> -QueryName <q>` |
| Owner | 4 Publish query | `./scripts/frontend/run-query.ps1 -Persona woodgrove -QueryName <q> -BodyFile <f> -SkipMonitor -SkipResults` |
| Owner | 5 Run query | `./scripts/frontend/run-query.ps1 -Persona woodgrove -QueryName <q>` |

> Every script supports `-DryRun` to print the planned requests without calling
> the service — useful for validating a run before executing it.
