# Frontend (dataplane) workflow scripts — Steps 03–12

These scripts wrap the **frontend REST API** calls that README-API.md
(`../../README-API.md`) shows inline for Steps 03–12. They complement the
existing resource-provisioning helpers in `../` and the Bicep control-plane
deployment in `../bicep/`.

## Where this fits

| Phase | Surface | Tooling |
|-------|---------|---------|
| Create collaboration, enable workload, add collaborator (Steps 02) | ARM control plane (`management.azure.com`, `2026-09-30-preview`) | `../bicep/managed-cleanroom.bicep` + `../bicep/deploy-managed-cleanroom.ps1` |
| Provision storage/KV/MI, OIDC storage, data upload (Steps 04–06 helpers) | ARM + storage | `../04-*`, `../05-*`, `../06-*`, `../07-*`, `../08-*`, `../09-*` |
| **Accept, publish, approve, run, monitor, results (Steps 03–12 REST)** | **Frontend / dataplane** (`...cleanroom.cloudapp.azure.net`, `2026-03-01-preview`) | **the scripts in this folder** |

Bicep cannot express Steps 03–12 — they are runtime governance/dataplane
operations, not resource provisioning. See `../../README-API.md` Appendix F.

Run every command below from **`demos/analytics-using-managedcleanroom/`**.
Do not change into `scripts/frontend`: generated metadata and body paths are
relative to the demo root. For a complete ordered setup, follow
[README-SCRIPTS.md](../README-SCRIPTS.md).

## Prerequisites

1. The collaboration is created and the Analytics workload is enabled and
   healthy (run `.\scripts\bicep\deploy-managed-cleanroom.ps1` first).
2. Each collaborator is authenticated and has a **frontend token** available.
   Token resolution order used by `Invoke-Frontend.ps1`:
   1. `-TokenFile <path>` (explicit override)
   2. `$env:CLEANROOM_FRONTEND_TOKEN` (SPN / CI)
   3. `$env:MANAGEDCLEANROOM_ACCESS_TOKEN` (CLI session)
   4. per-persona temp file `msal-idtoken-<persona>.txt`

   Acquire a per-persona token and extract its `oid` as `$personaOid`
   (README Step 1.5). For the corporate-account token option:

   ```powershell
   $persona = "woodgrove"   # use "northwind" in Northwind's terminal
   az account get-access-token --resource "https://management.azure.com/" `
       --query accessToken -o tsv |
       Out-File (Join-Path ([System.IO.Path]::GetTempPath()) "msal-idtoken-$persona.txt") -NoNewline
   ```

## Files

| Script | Step | Runs as |
|--------|------|---------|
| `Invoke-Frontend.ps1` | — | shared helper (dot-sourced) |
| `get-frontend-data.ps1` | 03–11 | ANY authorized collaborator; lists/shows resources |
| `run-collaborator.ps1` | 03 + 06 + 08 | EACH collaborator (orchestrator) |
| `run-query.ps1` | 07 + 09–11 | Owner; returns the job ID or service-preview response |
| `03-accept-invitation.ps1` | 03 | EACH collaborator |
| `05-fetch-jwks.ps1` | 5.1 | EACH collaborator |
| `05-set-issuer-url.ps1` | 5.3 | EACH collaborator |
| `06-publish-dataset.ps1` | 6.2 / 6.3 | EACH (output = Woodgrove) |
| `06-set-consent.ps1` | 06 | Document owner; enable/disable without republishing |
| `07-publish-query.ps1` | 7.2 | Woodgrove |
| `08-approve-query.ps1` | 08 | EACH collaborator |
| `09-run-query.ps1` | 09 | Woodgrove |
| `09-cancel-query.ps1` | 09 | Woodgrove; full job ID required |
| `10-monitor-query.ps1` | 10 | ANY; bounded polling or `-Once` |
| `11-results-audit.ps1` | 11.1 / 11.2 | Woodgrove |
| `12-get-readonly-kubeconfig.ps1` | 12.1 | Owner (ARM action) |

> `12-get-readonly-kubeconfig.ps1` is the one exception: it is an ARM *action*
> (`getReadonlyKubeConfig`), so it uses `az rest` against ARM and takes the
> resource group + collaboration name, not the frontend context.

## Quick start — collaborator orchestrator

For a cross-party query, Northwind must first join, configure access and publish
its input dataset. Woodgrove then publishes `query2` referencing that dataset.
Northwind can now approve the **already-published** query:

```powershell
$collabId = "<frontend-collaboration-uuid>"
$queryName = "<query2-name-from-woodgrove>"
.\scripts\frontend\run-collaborator.ps1 -Persona northwind -CollaborationId $collabId `
    -QueryName $queryName -SkipAccept
```

Single-owner `query1` binds Woodgrove's input twice; it needs neither Northwind
data nor a Northwind vote. Publication casts Woodgrove's accept vote.

Pass `-CollaborationId` or `-CollaborationName` explicitly. The orchestrators
auto-select only when exactly one collaboration is visible; ambiguous names
or multiple visible collaborations produce an error. Resource provisioning and
OIDC are prerequisites and are not run by the orchestrator. Do not defer a
query's input-dataset publication until its approval step.

## Per-persona runbook (individual steps)

Set common variables in each participant's terminal. Choose the persona
**before** deriving `$personaRg` and resolving the collaboration:

```powershell
$persona = "northwind"          # or "woodgrove"
$personaRg = "cr-e2e-$persona-rg"
$collabName = "<collaboration-name>"
. .\scripts\frontend\Invoke-Frontend.ps1
$fe = Get-FrontendContext -Persona $persona
$collabId = Resolve-CollaborationId -Context $fe -CollaborationName $collabName
.\scripts\frontend\03-accept-invitation.ps1 -Persona $persona -CollaborationId $collabId
```

Next run README Step 04 to provision resources and upload data, and Step 05
to configure OIDC and grants. The frontend calls within Step 05 can use:

```powershell
.\scripts\frontend\05-fetch-jwks.ps1 -Persona $persona -CollaborationId $collabId `
    -outDir "generated\$personaRg"
# Run .\scripts\06-setup-oidc-storage.ps1 with this JWKS file (README Step 5.2).
.\scripts\frontend\05-set-issuer-url.ps1 -Persona $persona -CollaborationId $collabId `
    -outDir "generated\$personaRg"
# Then run .\scripts\07-grant-access.ps1 (README Step 5.4).
```

Build the dataset bodies with `.\scripts\08-build-dataset-body.ps1` as in
README Step 6.1, retaining Woodgrove's optional `2025-09-01` subdirectory.
The examples below use suffix `-v1`; replace it consistently if your upload
used a different suffix. Prepare keys after publishing if using CPK.

### Northwind (publisher — input only)

```powershell
.\scripts\frontend\06-publish-dataset.ps1 -Persona northwind -CollaborationId $collabId `
    -DocumentId "northwind-input-csv-v1" -BodyFile "generated\publish\northwind-input-dataset.json"
```

Send this exact dataset ID to Woodgrove and wait for the cross-party query to
be published. Then approve the query name supplied by Woodgrove:

```powershell
$queryName = "<query2-name-from-woodgrove>"
.\scripts\frontend\08-approve-query.ps1 -Persona northwind -CollaborationId $collabId -QueryName $queryName
```

### Woodgrove (owner — input + output, publishes/runs query)

Use Woodgrove's own terminal, token and resource metadata from the common steps.
Publish both datasets:

```powershell
.\scripts\frontend\06-publish-dataset.ps1 -Persona woodgrove -CollaborationId $collabId `
    -DocumentId "woodgrove-input-csv-v1" -BodyFile "generated\publish\woodgrove-input-dataset.json"
.\scripts\frontend\06-publish-dataset.ps1 -Persona woodgrove -CollaborationId $collabId `
    -DocumentId "woodgrove-output-csv-v1" -BodyFile "generated\publish\woodgrove-output-dataset.json"
```

Build **either** the single-owner `query1-v1` body or the cross-party `query2-v1`
body with `.\scripts\09-build-query-body.ps1` (README Step 7.1). Query2 requires
Northwind's dataset to have been published already. Publish the chosen body:

```powershell
$queryName = "query1-v1"   # or "query2-v1" if you built the cross-party body
.\scripts\frontend\07-publish-query.ps1 -Persona woodgrove -CollaborationId $collabId `
    -QueryName $queryName -BodyFile "generated\publish\$queryName.json"
```

For query2, wait for Northwind's vote before continuing. Query1 needs no extra
vote. Check approval and execute:

```powershell
$queryInfo = Invoke-Frontend -Context $fe -Path "$collabId/analytics/queries/$queryName"
if ($queryInfo.state -ne "Accepted") { throw "Wait for the required query approvals before running." }
$jobId = .\scripts\frontend\09-run-query.ps1 -Persona woodgrove -CollaborationId $collabId `
    -QueryName $queryName -ScaleSku small
.\scripts\frontend\10-monitor-query.ps1 -Persona woodgrove -CollaborationId $collabId -JobId $jobId
.\scripts\frontend\11-results-audit.ps1 -Persona woodgrove -CollaborationId $collabId -QueryName $queryName
.\scripts\11-download-output.ps1 -resourceGroup $personaRg -datasetSuffix "-v1" `
    -JobId $jobId -OutputDir ".\generated\output\$jobId"
.\scripts\frontend\12-get-readonly-kubeconfig.ps1 -resourceGroup "<collabRg>" -collaborationName "<collabName>"
```

## Notes

- Frontend scripts accept optional `-Frontend` and `-TokenFile` overrides;
  the ARM-only kubeconfig helper does not. An explicit token file takes
  precedence over environment tokens. The default frontend is the non-attested
  West US endpoint with CA-issued TLS; certificate verification remains enabled.
- `06-publish-dataset.ps1 -DisableConsent` disables execution consent after
  publishing. Use `06-set-consent.ps1` to change consent without republishing.
- `09-run-query.ps1` and `run-query.ps1` accept `-ScaleSku small` (default),
  `medium`, or `large`, and `-StartDate` / `-EndDate` for date-range filtering.
  The Spark execution profile is separate from the collaboration's AKS SKU.
- Both run helpers stop if the frontend response has no job ID. Check run
  history before retrying: the service may already have scheduled the query.
- `10-monitor-query.ps1` prints new/updated events while waiting and throws on
  failure, cancellation, or timeout after a best-effort health lookup. Default
  timeout is 60 minutes (`-TimeoutMinutes`); `-Once` returns one detailed result,
  including failures, without waiting. Timeout does not cancel or resubmit.
- `-DryRun` on these frontend scripts previews requests locally; it does not
  send the service-side `dryRun: true` request or validate `skuSettings`.
  Run helpers return `<job-id>` for a local preview, not an actual submitted run.
  The kubeconfig helper reads the current Azure CLI account unless
  `-Subscription` is supplied.

## Additional operations

See [Fast Setup Scripts](../README-SCRIPTS.md) for the complete examples:

| Capability | Script / option |
|---|---|
| Effective `skuSettings` without executing | Either query runner with `-ServiceDryRun`; accepts `-ScaleSku`, `-StartDate`, `-EndDate` |
| Lists and individual resource inspection | `get-frontend-data.ps1 -Resource <resource>` |
| One detailed run result, including events | `10-monitor-query.ps1 -JobId <id> -Once` |
| Cancel a persistent failure/unwanted run | `09-cancel-query.ps1 -QueryName <name> -JobId <full-job-id>` |
| Consent without republishing | `06-set-consent.ps1 -DocumentId <id> -ConsentAction` with `enable` or `disable` |
| ARM show/health, enable, add, recover, delete-only | `../bicep/manage-collaboration.ps1 -Action <action>` |

The reader supports `Collaborations` (`-ActiveOnly`), `Invitations`
(`-PendingOnly`), `Collaborators`, `OidcKeys`, `Datasets`, `Dataset`, `Queries`,
`Query`, `Run`, `RunHistory`, and `AuditEvents`. Supply `-CollaborationId` except
when listing collaborations; individual documents need `-DocumentId`
(`-QueryName` is an alias), and runs need `-JobId`.

`-ServiceDryRun` cannot be combined with `-DryRun` or the orchestrator's
`-BodyFile`. Real execution returns the job ID; diagnostics go to the host.
`large` remains preview-only. Inspect detailed events before cancelling;
one early `FailedMount` is not a reason to cancel.
