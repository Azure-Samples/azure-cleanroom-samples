# Customer TSG Index

Troubleshooting guide for Azure Confidential Clean Room — Analytics workload.
Issues extracted from the [REST API guide](../../README-API.md) and [CLI guide](../../README-CLI.md).

---

## Prerequisites & Setup

| # | Issue | Error / Symptom | Cause | Fix | Step |
|---|---|---|---|---|---|
| 1 | Quota insufficient | Collaboration creation fails, or a query cannot acquire resources | Collaboration creation requires 12 Ddsv5 vCPUs for the supported `Standard_D4ds_v5` three-node AKS configuration. The consortium also requires 4 Confidential ACI vCPUs, 2 Confidential container groups, 1 Standard ACI vCPU, and 1 Standard container group. Query execution separately requires Confidential ACI vCPU and container-group quota for the planned concurrency. | Verify the AKS and consortium-provisioning ACI quotas before creating the collaboration. Use the capacity-planning table to calculate the additional Confidential ACI quota required for query execution in `$resourceLocation`. | Step 01 |
| 2 | RP role assignment required | ARM operations fail due to missing permissions | RP App requires User Access Administrator on the subscription | `az role assignment create --assignee "d76bde86-0387-4db5-af46-51a9e31e6666" --role "User Access Administrator" --scope "/subscriptions/$subscription"` | Step 01 |
| 3 | Python 3.13 tuple error (CLI only) | CLI commands fail with tuple error | CLI extension bug in older versions | Upgrade to `managedcleanroom` extension `1.0.0b10` | Any CLI step |

## Identity & Federated Credentials

| # | Issue | Error / Symptom | Cause | Fix | Step |
|---|---|---|---|---|---|
| 4 | Wrong OID used | `SPARK_JOB_FAILED: ExitCode 1` or `AADSTS700211` | Used `az ad signed-in-user show --query id` instead of JWT `oid` (differ for MSA accounts) | Extract `oid` from JWT payload (Step 1.5.2). Delete/recreate FIC with correct subject `Analytics-{oid}`. | Step 01 / 05 |
| 5 | Wrong contractId casing | Federated credential subject mismatch at query runtime | Used lowercase `analytics` instead of `Analytics` | `contractId` must be `"Analytics"` (capital A) in `07-grant-access.ps1` | Step 05 |
| 6 | No matching federated identity record | `AADSTS700211: No matching federated identity record` | Wrong issuer URL or stale FIC | Republish dataset; delete and recreate FIC with correct issuer URL | Step 05 / 06 |

## Collaboration & Workload

| # | Issue | Error / Symptom | Cause | Fix | Step |
|---|---|---|---|---|---|
| 7 | Health state not Ok | `healthState` is not `Ok` after enabling workload | Pods not ready; infrastructure provisioning issue | Poll `healthState` and inspect `healthIssues` array for pod/container failures | Step 02 |
| 8 | ContractNotFound | `ContractNotFound`; frontend errors on all operations | Stale CCF endpoint | Create a new collaboration, or use Force Recover as last resort | Any |
| 9 | Unresponsive collaboration | All frontend operations fail | Internal state corruption | **API**: `az rest --method POST --url ".../recover" --body '{"forceRecover":true}'`. **CLI**: `az managedcleanroom collaboration recover --force-recover $true` | Any |

## Connectivity & Certificates

| # | Issue | Error / Symptom | Cause | Fix | Step |
|---|---|---|---|---|---|
| 10 | SSL certificate verify failed | `SSL certificate verify failed` | Wrong or stale frontend endpoint | Configure the non-attested endpoint explicitly and keep TLS verification enabled | Any frontend call |
| 11 | NSG blocking AKS endpoint | Query execution times out | Tenant NSGs block inbound port 443 to AKS | Contact ACCR team with `tenantId` to whitelist tenant | Step 09 |
| 12 | AVNM blocking connectivity | Query execution times out | Azure Virtual Network Manager tenant policy blocks port 443 | Tenant admin must create AVNM rule to allow port 443 from internet | Step 09 |

## Query Execution

Start with the detailed run result, not run history. Run history is a summary
and can omit `events[]`.

First, list the detailed query events to see execution progress, warnings, and
failures. Then inspect the application state and error:

```powershell
$result.events |
    Select-Object type, reason, message, firstTimestamp, lastTimestamp, count |
    Format-Table -Wrap

$result.status.applicationState | ConvertTo-Json -Depth 5
```

If the run events indicate a collaboration-wide workload or endpoint problem,
check collaboration health:

```powershell
# CLI guide
az managedcleanroom collaboration show `
    --collaboration-name $collabName `
    --resource-group $collabRg `
    --query "health"

# REST API guide
az rest --method GET --resource $armEndpoint `
    --url "$collabArmUrl`?api-version=$armApiVersion" `
    | ConvertFrom-Json | ForEach-Object { $_.properties.health } |
    ConvertTo-Json -Depth 5
```

- `healthState: Ok`: the collaboration workload is healthy; continue with the
  individual run's error and events.
- `healthState: Error`: preserve `healthIssues` and the run ID when contacting
  support.
- `AnalyticsEndpointUnreachable`: verify NSG/AVNM port 443 requirements. If
  networking is correct and the issue persists, contact support.

| # | Issue | Error / Symptom | Cause | Fix | Step |
|---|---|---|---|---|---|
| 13 | Pending rerun | `PENDING_RERUN` | Normal scheduling state | Keep polling; it transitions to `SUBMITTED` automatically. | Step 10 |
| 14 | Transient driver mount warning | One early `FailedMount` for `spark-drv-...-conf-map`, followed by progress | Driver pod was created immediately before its Spark configuration map | No action. Do not cancel or resubmit the run. | Step 10 |
| 15 | Confidential ACI placement failure | During query execution, `FailedCreatePodSandBox: resource is not available in the location ... Resource requested: N CPU M GB`; executors remain in `Init`/`PENDING` | Confidential ACI platform capacity is unavailable for the requested query container-group size in the region; this can occur even when subscription quota is sufficient and is distinct from collaboration-creation quota. The warning is surfaced as a query-pod warning in detailed `events[]`. | If the warning persists, use [Cancel a Query](../../README-CLI.md#cancel-a-query), reduce concurrent runs, verify Confidential ACI quota, and retry later. If a stuck run has no useful warning, check collaboration health and preserve the run ID. If the issue continues, contact the Azure Container Instances (ACI) team with the region, requested CPU/memory, AKS-VN2 resource ID, and event message. | Step 09 / 10 |
| 16 | Collaboration scheduling capacity exhausted | `FailedScheduling: all schedulable Spark nodes are at their per-node pod limit` | Concurrent runs exceed the scheduling capacity of the supported `Standard_D4ds_v5` three-node configuration | Reduce concurrent runs. Upcoming updates will enable higher capacity, but an existing collaboration cannot be resized; create a new collaboration after the required configuration becomes supported. | Step 01 / 10 |
| 17 | Executor starvation | Driver `ExitCode: 11`; all executors fail; `Initial job has not accepted any resources` repeats | Executors cannot acquire Confidential ACI capacity | Check `events[]` for `FailedCreatePodSandBox`, then follow issue 15. | Step 10 |
| 18 | Dataset identity failure | `SPARK_JOB_FAILED: ExitCode 1` with `AADSTS700211: No matching federated identity record` | Incorrect federated-credential subject or stale dataset issuer | Recreate the FIC with subject `Analytics-{token oid}` and republish the dataset if its issuer is stale. Never substitute `az ad signed-in-user show --query id` for the token `oid`. | Step 05 / 06 / 10 |
| 19 | Insufficient shuffle/spill space | `java.io.IOException: No space left on device` during a wide join or shuffle | The selected `scaleSku` is too small for the workload | Retry with `medium` when using `small`. `large` is preview-only and is not ready for production use. | Step 09 / 10 |
| 20 | Executor start-then-crash | Executor reaches `Running`, then fails with `ExitCode: 1` and no useful message | Unknown runtime failure after successful placement; this alone does not prove a capacity shortage | Do not classify it as capacity unless `FailedCreatePodSandBox` is also present. Retry once; if it recurs, preserve the run ID and events and contact support. | Step 10 |
| 21 | Container or image failure | Persistent `Failed`/`BackOff`, image-pull, container-crash, or pod-init errors | Collaboration workload or service infrastructure failure | Check collaboration health. If the issue persists, preserve the run ID and `healthIssues` and contact support. | Step 10 |
| 22 | Query submission timeout | `AnalyticsRequestFailed`, 100-second timeout, or no run appears in run history | Analytics endpoint unreachable, unhealthy AKS workload, or NSG/AVNM blocking port 443 | Check run history before resubmitting, then check collaboration health and networking. Preserve the run ID, if created, and contact support when health remains in `Error`. | Step 09 / 10 |
| 23 | Already voted / Conflict | `Already voted` or `Conflict` on vote | Idempotent vote — already voted | Safe to ignore only when the query already records that collaborator's vote. Do not ignore unrelated conflicts. | Step 08 |

A single early warning is not necessarily fatal. Treat it as blocking when the
query stops making progress and the same warning persists, or when the run
reaches `FAILED` or `SUBMISSION_FAILED`.

For any unresolved query-execution issue, contact the ACCR team and provide:

1. Collaboration resource ID.
2. Run ID.
3. Run events.
4. `healthIssues`, when available.
5. [Read-only kubeconfig](../../README-CLI.md#121-get-readonly-kubeconfig)
   through an approved secure support channel.

## Dataset & Schema

| # | Issue | Error / Symptom | Cause | Fix | Step |
|---|---|---|---|---|---|
| 24 | Schema incompatible | `is_schema_compatible: Missing field` | Output `allowedFields` missing query output columns | Ensure output dataset `allowedFields` includes all columns the query produces | Step 06 / 07 |
| 25 | CPK data corruption | Decryption failures in CPK mode | User manually encrypted files before upload | CPK is server-side encryption — upload plaintext via `azcopy copy --cpk-by-value` | Step 04 |

## Frontend API

| # | Issue | Error / Symptom | Cause | Fix | Step |
|---|---|---|---|---|---|
| 26 | 404 Not Found on frontend | `404 Not Found` | Using ARM resource ID instead of frontend UUID | **API**: Use UUID from `Invoke-Frontend -Path ""`. **CLI**: Use UUID from `frontend collaboration list` | Step 03+ |
| 27 | BOM encoding in body JSON | ARM API rejects body JSON | PowerShell `Out-File` adds BOM | Use `[System.IO.File]::WriteAllText()` instead of `Out-File` | Step 02 |

## SPN / App-Based Authentication

| # | Issue | Error / Symptom | Cause | Fix | Step |
|---|---|---|---|---|---|
| 28 | Certificate not registered | `AADSTS700027: certificate not registered` | Using `az login` cert auth instead of MSAL SNI | Use Python MSAL with `public_certificate` via `get-sp-token-sni.ps1` | SPN auth |
| 29 | Credential lifetime error | `Credential lifetime exceeds max value` | Certificate lifetime too long | Use OneCert + `trustedCertificateSubjects` in app manifest | SPN auth |
| 30 | Invalid collaborator identifier | `InvalidCollaboratorIdentifier` | Missing `--object-id` and `--tenant-id` | Add `--object-id` (Enterprise App, not app reg) and `--tenant-id` to `add-collaborator` | SPN setup |
