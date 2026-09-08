# Frontend (dataplane) workflow scripts — Steps 03–12

These scripts wrap the **frontend REST API** calls that README-API.md
(`../../README-API.md`) shows inline for Steps 03–12. They complement the
existing resource-provisioning helpers in `../` and the Bicep control-plane
deployment in `../bicep/`.

## Where this fits

| Phase | Surface | Tooling |
|-------|---------|---------|
| Create collaboration, enable workload, add collaborator (Steps 02) | ARM control plane (`management.azure.com`, `2026-04-30-preview`) | `../bicep/managed-cleanroom.bicep` + `../bicep/deploy-managed-cleanroom.ps1` |
| Provision storage/KV/MI, OIDC storage, data upload (Steps 04–06 helpers) | ARM + storage | `../04-*`, `../05-*`, `../06-*`, `../07-*`, `../08-*`, `../09-*` |
| **Accept, publish, approve, run, monitor, results (Steps 03–12 REST)** | **Frontend / dataplane** (`...cleanroom.cloudapp.azure.net`, `2026-03-01-preview`) | **the scripts in this folder** |

Bicep cannot express Steps 03–12 — they are runtime governance/dataplane
operations, not resource provisioning. See `../../README-API.md` Appendix F.

## Prerequisites

1. The collaboration is created and the Analytics workload is enabled and
   healthy (run `../../bicep/deploy-managed-cleanroom.ps1` first).
2. Each collaborator is authenticated and has a **frontend token** available.
   Token resolution order used by `Invoke-Frontend.ps1`:
   1. `$env:CLEANROOM_FRONTEND_TOKEN` (SPN / CI)
   2. `-TokenFile <path>`
   3. per-persona temp file `msal-idtoken-<persona>.txt`

   Acquire a per-persona token (README Step 1.5):

   ```powershell
   az account get-access-token --resource "https://management.azure.com/" `
       --query accessToken -o tsv |
       Out-File (Join-Path $env:TEMP "msal-idtoken-woodgrove.txt") -NoNewline
   ```

## Files

| Script | Step | Runs as |
|--------|------|---------|
| `Invoke-Frontend.ps1` | — | shared helper (dot-sourced) |
| `run-collaborator.ps1` | 03 + 06 + 08 | EACH collaborator (orchestrator) |
| `03-accept-invitation.ps1` | 03 | EACH collaborator |
| `05-fetch-jwks.ps1` | 5.1 | EACH collaborator |
| `05-set-issuer-url.ps1` | 5.3 | EACH collaborator |
| `06-publish-dataset.ps1` | 6.2 / 6.3 | EACH (output = Woodgrove) |
| `07-publish-query.ps1` | 7.2 | Woodgrove |
| `08-approve-query.ps1` | 08 | EACH collaborator |
| `09-run-query.ps1` | 09 | Woodgrove |
| `10-monitor-query.ps1` | 10 | ANY |
| `11-results-audit.ps1` | 11.1 / 11.2 | Woodgrove |
| `12-get-readonly-kubeconfig.ps1` | 12.1 | Owner (ARM action) |

> `12-get-readonly-kubeconfig.ps1` is the one exception: it is an ARM *action*
> (`getReadonlyKubeConfig`), so it uses `az rest` against ARM and takes the
> resource group + collaboration name, not the frontend context.

## Quick start — collaborator orchestrator

When the query is **already published** by the owner, a collaborator can join and
approve in one command with `run-collaborator.ps1` (chains 03 → optional 06 → 08):

```powershell
cd scripts/frontend
# Northwind: accept invite + vote on an already-published query
./run-collaborator.ps1 -Persona northwind -QueryName query1-v1

# Also publish a dataset before voting
./run-collaborator.ps1 -Persona northwind -QueryName query2-v1 `
    -DatasetDocumentId northwind-input-csv-v1 `
    -DatasetBodyFile generated/publish/northwind-input-dataset.json
```

It resolves the collaboration (by `-CollaborationId`, `-CollaborationName`, or the
first visible one), tolerates an already-accepted invite, and forwards `-DryRun`.
Resource provisioning + OIDC (Steps 04–05) are prerequisites and are not run here.

## Per-persona runbook (individual steps)

Set common variables in each collaborator's terminal:

```powershell
$persona = "northwind"          # or "woodgrove"
$personaRg = "cr-e2e-$persona-rg"
$collabId = "<frontend-collaboration-uuid>"   # printed by 03-accept-invitation.ps1
cd scripts/frontend
```

### Northwind (publisher — input only)

```powershell
./03-accept-invitation.ps1 -Persona $persona
./05-fetch-jwks.ps1        -Persona $persona -CollaborationId $collabId -outDir "generated/$personaRg"
# (run ../06-setup-oidc-storage.ps1 and ../07-grant-access.ps1 per README Step 5)
./05-set-issuer-url.ps1    -Persona $persona -CollaborationId $collabId -outDir "generated/$personaRg"
# (run ../08-build-dataset-body.ps1 per README Step 6.1)
./06-publish-dataset.ps1   -Persona $persona -CollaborationId $collabId `
    -DocumentId "northwind-input-csv-v1" -BodyFile "generated/publish/northwind-input-dataset.json"
./08-approve-query.ps1     -Persona $persona -CollaborationId $collabId -QueryName "query2-v1"
```

### Woodgrove (owner — input + output, publishes/runs query)

```powershell
$persona = "woodgrove"
./03-accept-invitation.ps1 -Persona $persona
./05-fetch-jwks.ps1        -Persona $persona -CollaborationId $collabId -outDir "generated/$personaRg"
./05-set-issuer-url.ps1    -Persona $persona -CollaborationId $collabId -outDir "generated/$personaRg"
./06-publish-dataset.ps1   -Persona $persona -CollaborationId $collabId `
    -DocumentId "woodgrove-input-csv-v1"  -BodyFile "generated/publish/woodgrove-input-dataset.json"
./06-publish-dataset.ps1   -Persona $persona -CollaborationId $collabId `
    -DocumentId "woodgrove-output-csv-v1" -BodyFile "generated/publish/woodgrove-output-dataset.json"
# (build query body via ../09-build-query-body.ps1 per README Step 7.1)
./07-publish-query.ps1     -Persona $persona -CollaborationId $collabId `
    -QueryName "query1-v1" -BodyFile "generated/publish/query1-v1.json"
./08-approve-query.ps1     -Persona $persona -CollaborationId $collabId -QueryName "query1-v1"
$jobId = (./09-run-query.ps1 -Persona $persona -CollaborationId $collabId -QueryName "query1-v1")
./10-monitor-query.ps1     -Persona $persona -CollaborationId $collabId -JobId $jobId
./11-results-audit.ps1     -Persona $persona -CollaborationId $collabId -QueryName "query1-v1"
# then download output via ../11-download-output.ps1 (README Step 11.3)
./12-get-readonly-kubeconfig.ps1 -resourceGroup <collabRg> -collaborationName <collabName>
```

## Notes

- All scripts accept optional `-Frontend` and `-TokenFile` overrides.
- `06-publish-dataset.ps1 -DisableConsent` toggles execution consent off.
- `09-run-query.ps1` accepts `-StartDate` / `-EndDate` for date-range filtering.
- `10-monitor-query.ps1` dumps collaboration `health` on any non-`COMPLETED` run.
