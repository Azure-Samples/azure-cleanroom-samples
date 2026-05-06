# Customer TSG Index

Troubleshooting guide for Azure Confidential Clean Room — Analytics workload.
Issues extracted from the [REST API guide](../../README-API.md) and [CLI guide](../../README-CLI.md).

---

## Prerequisites & Setup

| # | Issue | Error / Symptom | Cause | Fix | Step |
|---|---|---|---|---|---|
| 1 | Quota insufficient | Collaboration creation or query execution fails | Subscription lacks minimum vCPU quota in the `$resourceLocation` region | Ensure at least 8 vCPUs Ddsv5 (AKS) + 6 vCPUs Confidential ACI. Add 4 Ddsv5 per additional concurrent query. | Step 01 |
| 2 | RP role assignment required | ARM operations fail due to missing permissions | RP App requires User Access Administrator on the subscription | `az role assignment create --assignee "d76bde86-0387-4db5-af46-51a9e31e6666" --role "User Access Administrator" --scope "/subscriptions/$subscription"` | Step 01 |
| 3 | Python 3.13 tuple error (CLI only) | CLI commands fail with tuple error | CLI extension bug in older versions | Upgrade to `managedcleanroom` extension v1.0.0b5+ | Any CLI step |

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
| 10 | SSL certificate verify failed | `SSL certificate verify failed` | EUAP endpoint cert mismatch | **API**: `-SkipCertificateCheck` (in `Invoke-Frontend`). **CLI**: `$env:AZURE_CLI_DISABLE_CONNECTION_VERIFICATION = "1"` | Any frontend call |
| 11 | NSG blocking AKS endpoint | Query execution times out | Tenant NSGs block inbound port 443 to AKS | Contact ACCR team with `tenantId` to whitelist tenant | Step 09 |
| 12 | AVNM blocking connectivity | Query execution times out | Azure Virtual Network Manager tenant policy blocks port 443 | Tenant admin must create AVNM rule to allow port 443 from internet | Step 09 |

## Query Execution

| # | Issue | Error / Symptom | Cause | Fix | Step |
|---|---|---|---|---|---|
| 13 | Spark job failed | `SPARK_JOB_FAILED: ExitCode 1` | Federated credential subject mismatch | Delete/recreate FIC with correct `Analytics-{oid}` subject | Step 09 / 10 |
| 14 | Query fails or times out | `FAILED` / `SUBMISSION_FAILED` or stuck in `SUBMITTED`/`RUNNING` | CACI capacity shortage, executor pods stuck, container crashes | Check `properties.health.healthIssues` for pod-level failures | Step 10 |
| 15 | PENDING_RERUN state | Query shows `PENDING_RERUN` | Normal scheduling behavior | Keep polling — transitions to `SUBMITTED` automatically | Step 10 |
| 16 | Already voted / Conflict | `Already voted` or `Conflict` on vote | Idempotent vote — already voted | Safe to ignore | Step 08 |

## Dataset & Schema

| # | Issue | Error / Symptom | Cause | Fix | Step |
|---|---|---|---|---|---|
| 17 | Schema incompatible | `is_schema_compatible: Missing field` | Output `allowedFields` missing query output columns | Ensure output dataset `allowedFields` includes all columns the query produces | Step 06 / 07 |
| 18 | CPK data corruption | Decryption failures in CPK mode | User manually encrypted files before upload | CPK is server-side encryption — upload plaintext via `azcopy copy --cpk-by-value` | Step 04 |

## Frontend API

| # | Issue | Error / Symptom | Cause | Fix | Step |
|---|---|---|---|---|---|
| 19 | 404 Not Found on frontend | `404 Not Found` | Using ARM resource ID instead of frontend UUID | **API**: Use UUID from `Invoke-Frontend -Path ""`. **CLI**: Use UUID from `frontend collaboration list` | Step 03+ |
| 20 | BOM encoding in body JSON | ARM API rejects body JSON | PowerShell `Out-File` adds BOM | Use `[System.IO.File]::WriteAllText()` instead of `Out-File` | Step 02 |

## SPN / App-Based Authentication

| # | Issue | Error / Symptom | Cause | Fix | Step |
|---|---|---|---|---|---|
| 21 | Certificate not registered | `AADSTS700027: certificate not registered` | Using `az login` cert auth instead of MSAL SNI | Use Python MSAL with `public_certificate` via `get-sp-token-sni.ps1` | SPN auth |
| 22 | Credential lifetime error | `Credential lifetime exceeds max value` | Certificate lifetime too long | Use OneCert + `trustedCertificateSubjects` in app manifest | SPN auth |
| 23 | Invalid collaborator identifier | `InvalidCollaboratorIdentifier` | Missing `--object-id` and `--tenant-id` | Add `--object-id` (Enterprise App, not app reg) and `--tenant-id` to `add-collaborator` | SPN setup |
