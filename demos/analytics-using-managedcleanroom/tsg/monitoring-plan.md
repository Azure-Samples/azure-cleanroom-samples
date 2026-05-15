# ACCR Comprehensive Monitoring, Alerting, Diagnostics & Debugging Plan

## Architecture Overview

ACCR has **3 AME-hosted services** and **per-customer collaboration deployments**:

| Component | Type | AKS Cluster | Namespace |
|---|---|---|---|
| **RP (CleanRoomService)** | AME Service | aks-prodcbn | cleanroom-ns |
| **Frontend Service** | AME Service | aks-frontend-prodcdm | frontendns |
| **Consortium Manager** | AME Service | consortiummanager-aks-prodcdm | consortiummanager-ns |
| **Collaboration AKS** | Customer Deployment | {name}-aks (per collaboration) | analytics, cleanroom-system, etc. |
| **CCF Consortium (CACI)** | Customer Deployment | N/A (ACI container group) | N/A |

**3 Personas**: Customer, Product Group, CSS (Support)

**Regions**: All monitoring applies to both **Canary** (eastus2euap / centraluseuap) and **West US (PROD)**.

**Telemetry Pipeline**: Geneva (Logs + Metrics) → Kusto. This pipeline is being built and will be the primary source for engineering debugging, alerting, monitors, and business reporting.

---

## 1. AME Service Monitoring (RP, Frontend, Consortium Manager)

### 1.1 Platform Metrics (Azure Monitor — automatic, no code change)

These are auto-collected by Azure for each AKS cluster hosting the services:

| Metric | Alert Threshold | Persona | Status |
|---|---|---|---|
| Node CPU % | > 80% sustained 5 min | Product Group | Planned |
| Node Memory Working Set % | > 85% sustained 5 min | Product Group | Planned |
| Cluster Node Count | < expected node count | Product Group | Planned |
| Pod Restart Count | > 3 in 15 min per pod | Product Group | Planned |
| Pod Status (NotReady/Failed) | Any | Product Group | Planned |
| API Server Request Latency | p99 > 5s | Product Group | Planned |
| API Server Availability | < 99.9% | Product Group | Planned |

**Implementation**: Azure Monitor Diagnostic Settings on each AME AKS cluster → Log Analytics workspace.

### 1.2 Synthetic Runners (Endpoint Availability & Validation)

Synthetic monitors to verify endpoint availability and end-to-end functionality for all 3 AME services:

| Synthetic | Target | Description | Frequency | Status |
|---|---|---|---|---|
| RP endpoint availability | ARM endpoint | ARM GET on known collaboration resource | 5 min | Planned |
| Frontend endpoint availability | Frontend Service (`/checkConnectionsInit`) | GET — no auth required | 5 min | Planned |
| Consortium Manager endpoint availability | Consortium Manager | Health endpoint | 5 min | Planned |
| BVT (Build Verification Tests) | PROD deployment | Regular automated E2E tests on PROD: create collaboration, enableWorkload, run query, validate output, delete. Alerts on failure. | Scheduled (e.g., daily) | Planned |
| Longhaul tests | PROD deployment | Long-running collaboration with periodic query execution to validate stability, resource leaks, and performance degradation over time. Alerts on health degradation or query failures. | Continuous | Planned |

### 1.3 Application-Level Metrics & Telemetry

#### 1.2.1 Geneva/MDM Metrics (AME Standard)

| Metric | Source Service | Description | Status |
|---|---|---|---|
| `CollaborationCreate.Duration` | RP Worker | Time from PUT to provisioningState=Succeeded (~25-35 min) | Planned |
| `CollaborationCreate.Success/Failure` | RP Worker | Count by result, failure reason | Planned |
| `EnableWorkload.Duration` | RP Worker | Time for enableWorkload operation (~7-8 min) | Planned |
| `EnableWorkload.Success/Failure` | RP Worker | Count by workloadType, failure reason | Planned |
| `DeleteCollaboration.Duration` | RP Worker | Full delete lifecycle time | Planned |
| `SyncOps.OperationDuration` | SyncOps Worker | Per-task duration (CreateCluster, CCF, MOBO, etc.) | Planned |
| `SyncOps.TaskFailure` | SyncOps Worker | By task name, exception type | Planned |
| `Frontend.RequestLatency` | Frontend Service | p50/p95/p99 of frontend API calls | Planned |
| `Frontend.RequestCount` | Frontend Service | By endpoint, status code | Planned |
| `Frontend.AuthFailure` | Frontend Service | By auth type (user/SPN/MI), error code | Planned |
| `Frontend.PublishDataset.Duration` | Frontend Service | Time to publish dataset | Planned |
| `Frontend.PublishDataset.Success/Failure` | Frontend Service | Count by result, error reason | Planned |
| `Frontend.PublishQuery.Duration` | Frontend Service | Time to publish query | Planned |
| `Frontend.PublishQuery.Success/Failure` | Frontend Service | Count by result, error reason | Planned |
| `Frontend.SubmitQuery.Duration` | Frontend Service | Time to submit analytics query for execution | Planned |
| `Frontend.SubmitQuery.Success/Failure` | Frontend Service | Count by result, error reason | Planned |
| `Frontend.QueryExecution.Duration` | Frontend Service | End-to-end query execution time (SUBMITTED → COMPLETED) | Planned |
| `Frontend.QueryExecution.Success/Failure` | Frontend Service | Count by final state (COMPLETED/FAILED/SUBMISSION_FAILED) | Planned |
| `Frontend.VoteOnProposal.Duration` | Frontend Service | Time to vote on governance proposal | Planned |
| `Frontend.VoteOnProposal.Success/Failure` | Frontend Service | Count by result, error reason | Planned |
| `Frontend.AddCollaborator.Duration` | Frontend Service | Time to add collaborator via frontend | Planned |
| `Frontend.AddCollaborator.Success/Failure` | Frontend Service | Count by result, error reason | Planned |
| `Frontend.ListAuditEvents.Duration` | Frontend Service | Time to retrieve audit events from CCF ledger | Planned |
| `Frontend.DownloadOutput.Duration` | Frontend Service | Time to download query output | Planned |
| `Frontend.DownloadOutput.Success/Failure` | Frontend Service | Count by result, error reason | Planned |
| `ConsortiumManager.OperationDuration` | Consortium Manager | CCF network operations | Planned |
| `ConsortiumManager.CCFHealth` | Consortium Manager | CCF node health checks | Planned |

#### 1.2.2 Structured Logging (Geneva/Kusto) — Status: TBD

Emit structured logs from all 3 services using the following schema (TBD — exact schema to be finalized):

```
{
  "timestamp": "2026-05-15T10:30:00Z",
  "service": "RP|Frontend|ConsortiumManager",
  "operationId": "<ARM operation ID>",
  "collaborationName": "ashank110",
  "subscriptionId": "a7f2c3cd-...",
  "tenantId": "8d32de66-...",
  "resourceGroup": "ashank-collab3",
  "taskName": "CreateClusterTask",
  "workflowName": "CreateClusterWorkflow",
  "level": "Information|Warning|Error",
  "message": "...",
  "exceptionType": "ArmException",
  "durationMs": 1500,
  "correlationId": "<guid>"
}
```

#### 1.2.3 Application Insights (Optional, for Frontend Service) — Status: TBD/NotPlanned

For the customer-facing Frontend Service, Application Insights provides:
- Dependency tracking (Frontend → CCF, Frontend → Analytics endpoint)
- Request/response tracing with correlation IDs
- Failed dependency alerts (e.g., analytics endpoint unreachable — the NRMS NSG issue)
- Live Metrics Stream for real-time debugging

### 1.3 Resource Logs (Diagnostic Settings)

Enable Diagnostic Settings on all 3 AME AKS clusters:

| Category | Purpose | Log Table | Status |
|---|---|---|---|
| `kube-audit-admin` | API server mutations (not get/list) | AKSAuditAdmin | TBD |
| `kube-apiserver` | API server logs | AKSControlPlane | TBD |
| `kube-controller-manager` | Controller manager events | AKSControlPlane | TBD |
| `kube-scheduler` | Scheduling decisions | AKSControlPlane | TBD |
| `guard` | AAD/RBAC auth events | AKSControlPlane | TBD |
| `cluster-autoscaler` | Node scaling events | AKSControlPlane | TBD |

**Cost optimization**: Use resource-specific mode, configure AKSAudit as Basic logs tier. Skip `kube-audit` (very high volume).

### 1.4 Container Insights — Status: Planned

Enable Container Insights on AME AKS clusters for:
- ContainerLogV2 (stdout/stderr from all pods)
- KubeEvents (pod creation, deletion, failures)
- KubePodInventory (pod state tracking)
- Perf (CPU/memory per container)

**Key containers to monitor:**

| Deployment | Containers | Critical Signals | Status |
|---|---|---|---|
| cleanroom-worker | cleanroomworker | Task execution, workflow state | Planned |
| cleanroom-syncopsworker | cleanroomsyncopsworker, cleanroominternalapi, ccf-provider-client, cleanroom-client, cgs-client, cluster-provider-client | Long-running ops, OOM (cluster-provider-client had 200Mi→512Mi fix), sidecar health | Planned |
| cleanroomapi | cleanroomapi | ARM API handling | Planned |
| frontend-service | frontend-service, cgs-client, skr, ccr-proxy | Frontend API, attestation, TLS | Planned |
| consortiummanagerservice | ccf-consortium-manager, ccr-proxy, skr | CCF network management | Planned |

---

## 2. Customer Collaboration Monitoring

### 2.1 RP-Side Collaboration Health Monitoring

The RP already has a `cleanroommonitoringworker` that monitors collaborations. Enhance it with:

#### 2.1.1 Health State Monitoring (existing pattern)

Current: `healthState` = `Ok` | `Error`, `healthIssues` array with pod/container failures.

**Enhancements needed:**

| Monitor | Current State | Gap | Action | Status |
|---|---|---|---|---|
| Pod health (all namespaces incl. analytics) | Monitors all necessary pods including analytics namespace for failures/errors | Filters out Succeeded pods (completed Spark jobs) | — | Available |
| CCF node health | Via RP monitoring (ccf-provider) | Auto-triggers consortium recovery when nodes are down/need replacement | — | Available |

#### 2.1.2 Provisioning Lifecycle Telemetry

Track every workflow task's duration and success/failure:

```
CreateCollaborationWorkflow:
  ├── CreateCollaborationPreflight     (~1 min)
  ├── CreateManagedResourceGroup       (~2-5 min, MOBO)
  ├── CreateCluster                    (~15 min)
  ├── UpdateClusterDenyAssignments     (~1 min)
  ├── EnableClusterObservability       (~6 min)
  ├── CreateConsortium                 (~15 min, CCF/CACI)
  ├── PrepareConsortium                (~2 min)
  ├── AddMembershipMapping             (~1 min)
  └── CreateCollaborationFinalize      (~1 min)

EnableWorkloadWorkflow:
  ├── EnableWorkloadPreflight          (~1 min)
  ├── GenerateContract                 (~1 min)
  ├── CreateSparkService               (~5 min)
  ├── UpdateWorkloadNsgRules           (~1 min)
  ├── SetDeploymentInfo                (~1 min)
  └── EnableWorkloadFinalize           (~1 min)
```

Emit duration + result for each task → Kusto table for tracking regressions.

#### 2.1.3 Collaboration ARM Resource Monitoring

| Signal | Source | Description | Status |
|---|---|---|---|
| Activity Log (customer RG) | Azure Monitor (auto) | ARM CRUD operations on all 4 resources in customer RG: `Microsoft.CleanRoom/Collaborations`, `Microsoft.CleanRoom/Consortiums`, `Microsoft.Resources/moboBrokers` (×2). Provisioning failures on sub-resources (AKS, CACI, etc.) surface as failures on the parent ARM resource. Customers can create Activity Log alerts on these. | Available |
| Activity Log (MRGs) | Azure Monitor (auto) | ARM operations on resources in managed RGs: **Collab MRG** (`CleanRoom_Collaborations_{name}`): AKS cluster, VNet, Public IP, NAT Gateway, UAMI, Private DNS. **Consortium MRG** (`CleanRoom_consortiums_ccf-{name}`): CACI container groups (×3), Storage Account, UAMI, Key Vault. Deny assignments allow `*/read`, so Activity Log (`Microsoft.Insights/eventtypes/values/read`) is accessible to customers. | Available |

### 2.2 Customer-Facing Monitoring

> Customer Alerts/Detection includes AzureMonitor for ARM resource events, Frontend APIs for ledger events, and collaboration health monitoring bubbled from the RP.

#### 2.2.1 RP (ARM) APIs

| API | Purpose | Notes | Status |
|---|---|---|---|
| ARM GET → `properties.health.healthState` | Health alerting | Customer can poll the ARM GET endpoint on a schedule (e.g., Azure Logic App, Azure Function, or their own monitoring) and alert when `healthState != Ok` | Available |
| POST `.../getReadonlyKubeConfig` | Readonly kubeconfig | Returns a readonly kubeconfig for the collaboration AKS cluster. Customer can use `kubectl` to inspect pods, logs, events across all namespaces including analytics. Also enables port-forwarding to Grafana dashboards for detailed Spark query execution monitoring. | Available |

#### 2.2.2 Frontend Service APIs

> **Note**: The Frontend Service is not an ARM endpoint — it is a separate HTTPS endpoint. Azure Monitor does not track Frontend API calls. Customers must implement their own monitoring (polling, alerting) on these APIs. Product Group telemetry for Frontend operations is tracked via Geneva metrics (see Section 1.2.1).

| API | Purpose | Notes | Status |
|---|---|---|---|
| `GET /collaborations/{id}` | Collaboration status | Returns collaboration user status | Available |
| `GET /{collabId}/analytics/runs/{jobId}` | Query execution state | SUBMITTED → RUNNING → COMPLETED | Available |
| `GET /{collabId}/analytics/queries/{name}/runs` | Run history | Total rows read/written, duration | Available |
| `GET /{collabId}/analytics/auditevents` | Audit trail | Ledger events from CCF | Available |

**Gaps for Customers:**

| Gap | Recommended Pattern | Priority | Status | Notes |
|---|---|---|---|---|
| No portal experience | Azure portal Insights blade for Collaborations | Medium | TBD | |

### 2.3 Collaboration AKS Cluster Observability

Each collaboration AKS cluster already deploys Prometheus (the EnableClusterObservability task). Extend with:

| Component | Purpose | Status |
|---|---|---|
| Prometheus (prometheus-server) | Cluster + pod metrics | Available |
| Grafana dashboards (`cleanroom-spark-grafana` in `telemetry` ns) | Spark query execution, resource usage, and logs visualization. Customer accesses via readonly kubeconfig + port-forward. | Available |
| Grafana dashboard enhancements | Add AKS diagnostics, additional cluster metrics, query run history and execution stats (rows read/written, duration, success/failure) to Grafana dashboards | Planned |

**Challenge**: Deny assignments on MOBO RGs block most management operations. Only RP MI, agentpool MI, and cluster MI are excluded.

**Approach for confidential logs**: Route collaboration cluster logs through the RP's monitoring infrastructure (Geneva sink in cluster-provider-client), since direct Log Analytics workspace access is blocked by deny assignments.

---

## 3. Alerting Strategy

> Product Group uses Geneva, Kusto, Alerts, and ICM for detection. Customer uses AzureMonitor for ARM resource events and Frontend APIs for ledger events.

### 3.1 Product Group Alerts (ICM)

| Alert | Source | Condition | Status |
|---|---|---|---|
| **RP Pod CrashLoop** | Container Insights | Pod restart count > 3 in 15 min | Planned |
| **RP Worker OOM** | Container Insights | OOMKilled event (cluster-provider-client had this, fixed to 512Mi) | Planned |
| **Collaboration Create Failure Rate** | Geneva metric | Failure rate > 10% in 1 hour | Planned |
| **Collaboration Create P95 Duration** | Geneva metric | P95 > 45 min (normal ~30 min) | Planned |
| **EnableWorkload Failure Rate** | Geneva metric | Failure rate > 10% in 1 hour | Planned |
| **Frontend Service 5xx Rate** | Geneva metric | 5xx rate > 5% in 5 min | Planned |
| **Frontend Unavailable** | Synthetic monitor | /checkConnectionsInit fails for > 5 min | Planned |
| **Consortium Manager Unavailable** | Synthetic monitor | Health endpoint fails for > 5 min | Planned |
| **MOBO MRG Creation Failure** | RP Worker logs | InternalError from MOBO (known intermittent) | Planned |
| **RBAC Propagation Timeout** | RP Worker logs | > 12 iterations (3 min) without success | Planned |
| **CACI Capacity Shortage** | Cluster Provider | FailedCreatePodSandBox per region | Planned |
| **VN2 Node Registration Failure** | Cluster Provider | WaitForVN2NodeReady timeout | Planned |
| **CCF Node Unhealthy** | Consortium Manager | CCF node health check fails | Planned |
| **Attestation Failure** | ccr-proxy logs | VerifySnpAttestationFailed (tag mismatch) | Planned |
| **Cosmos DB Throttling** | Geneva/Cosmos metrics | 429 responses > 10 in 5 min | Planned |

### 3.2 Customer Alerts (Azure Monitor, customer-configurable)

Expose these as Azure Monitor metrics on the `Microsoft.CleanRoom/Collaborations` resource:

| Metric | Customer Action | Status |
|---|---|---|
| `CollaborationHealthState` (0=Ok, 1=Error) | Create metric alert for degraded health | TBD | Health is already available via ARM GET API on the RP |
| `QueryExecution` | Query execution failure | NotPlanned | RP does not have query execution visibility — query execution is handled by the Frontend Service |

Expose as resource logs (Diagnostic Settings on Collaboration resource):

| Category | Data | Status |
|---|---|---|
| `CollaborationOperations` | Create, delete, enableWorkload, addCollaborator events | Planned |
| `QueryExecution` | Query submit, state changes, completion/failure | Planned |
| `AuditEvents` | CCF ledger events (who did what, when) | Planned |
| `HealthEvents` | Health state changes with details | Planned |

### 3.3 CSS (Support) Alerts

> **CSS Access**: CSS can get access to engineering Kusto via security group membership. Same Geneva → Kusto pipeline, same queries.

| Alert | Source | Purpose | Status |
|---|---|---|---|
| Customer collaboration stuck in Provisioning > 60 min | RP Worker telemetry | Proactive customer outreach | Planned |
| Customer query stuck in SUBMITTED > 15 min | Frontend Service telemetry and RP monitoring | CACI capacity or scheduling issue | Planned |
| Customer enableWorkload failure | RP Worker telemetry | Quota / capacity issue | Planned |
| Multiple collaborations failing in same region | Geneva aggregation | Regional platform issue | Planned |

---

## 4. Diagnostics & Debugging

> Product Group uses Geneva, Kusto (including confidential logs), and customer collaboration kubeconfig. Customer and CSS both use the [Customer TSG](customer/index.md). CSS can get access to Product Group Kusto via security group membership.

### 4.1 Product Group Debugging Workflow

```
Issue Reported
    │
    ├─── 1. Check RP worker logs/metrics in Kusto
    │        → Match operationId from ARM operation
    │        → Query RP worker logs for collaboration lifecycle events
    │
    ├─── 2. Check syncops worker logs in Kusto (for long-running ops)
    │        → CreateCluster, UpdateCluster, CCF operations
    │        → Filter by collaborationName, taskName, workflowName
    │
    ├─── 3. Check sidecar logs in Kusto (syncops pod has 6 containers)
    │        ccf-provider-client     → CCF/CACI operations
    │        cluster-provider-client → AKS cluster operations
    │        cgs-client              → Contract governance
    │        cleanroom-client        → Clean room operations
    │        cleanroominternalapi    → Internal API
    │
    ├─── 4. Check frontend service logs/metrics in Kusto
    │        → Frontend API errors, auth issues
    │
    ├─── 5. Check consortium manager logs in Kusto
    │        → CCF network operations
    │
    ├─── 6. Check customer collaboration cluster
    │        → Use readonly kubeconfig (via getReadonlyKubeConfig API)
    │        → Check pods across analytics namespace, cleanroom-system
    │        → Access Grafana dashboards for Spark execution details
    │
    ├─── 7. Check Cosmos DB (membership/state)
    │        Account: cleanroomdbgbl-{env}
    │        DB: ServiceDatabase, Container: userscontainer
    │        → Use AAD token auth (local auth disabled)
    │
    └─── 8. Check ARM Activity Log
             → ARM operation status, MOBO operations
```

### 4.2 Kusto Queries (Geneva → Kusto Pipeline) — Status: TBD

> **Note**: The Geneva (Logs + Metrics) → Kusto pipeline is being built. Once available, these queries (and similar alerts/monitors) will be defined on Geneva/Kusto. The table names, schemas, and exact queries below are placeholders and will be finalized when the pipeline is operational.

```kql
// TBD — queries will be defined once Geneva → Kusto pipeline is operational
// Examples of planned query patterns:

// Failed collaboration creations in last 24h
// CleanRoomOperations
// | where TimeGenerated > ago(24h)
// | where OperationType == "CreateCollaboration"
// | where Result == "Failed"
// | summarize Count=count(), FailureReasons=make_set(FailureReason) by SubscriptionId, TenantId

// Task duration percentiles
// CleanRoomOperations
// | where TimeGenerated > ago(7d)
// | summarize p50=percentile(DurationMs, 50), p95=percentile(DurationMs, 95) by TaskName

// MOBO failure rate by region
// CleanRoomOperations
// | where TaskName == "CreateManagedResourceGroupTask"
// | summarize Total=count(), Failed=countif(Result == "Failed") by Region

// Frontend error rate by endpoint
// FrontendServiceRequests
// | summarize Total=count(), Errors=countif(StatusCode >= 500) by Endpoint

// Customer collaborations with health issues
// CollaborationHealth
// | where HealthState == "Error"
// | project TimeGenerated, CollaborationName, SubscriptionId, HealthIssues
```

### 4.3 CSS Debugging Workflow

CSS should use the [Customer TSG](customer/index.md) plus these additional tools:

| Step | Tool | What to Check | Status |
|---|---|---|---|
| 1 | Azure Portal → Collaboration resource | provisioningState, healthState, healthIssues | Available |
| 2 | Activity Log on collaboration RG | ARM operation history, error details | Available |
| 3 | Customer TSG issue matrix | Map symptom to issue # (1-23) | Available |
| 4 | Kusto (via security group membership) | Query CleanRoomOperations for operation history | Planned |
| 5 | Escalate to Engineering | Provide: subscriptionId, collaborationName, operationId, timeframe | Available |

### 4.4 Common Failure Patterns (from operational history)

| Pattern | Root Cause | Detection | Auto-Mitigation | Status |
|---|---|---|---|---|
| MOBO InternalError | MOBO service intermittent failure | ARM InternalError on MRG creation | Recovery handler checks MRG existence (canaryConsort9 fix) | Available (fixed) |
| RBAC propagation race | Azure RBAC eventual consistency | 403 on role assignment operations | WaitForRbacPropagation (12×15s) checking all 3 roles | Available (fixed) |
| VN2 K8s version mismatch | AKS default K8s > VN2 kubelet version | VN2 node fails to register | Pin K8s version (1.33), WaitForVN2NodeReady | Available (fixed) |
| NRMS NSG blocking | NRMS policy auto-applies restrictive NSG | Frontend→Analytics timeout | UpdateWorkloadNsgRulesTask adds AllowInternetInboundHttps | Available (fixed) |
| Consortium null endpoints | Treating ARM resource existence as success | PrepareConsortiumTask NPE | Check provisioningState == Succeeded (canaryConsort9 fix) | Available (fixed) |
| cluster-provider-client OOM | 200Mi memory limit too low | OOMKilled event | Increased to 512Mi | Available (fixed) |
| Prometheus NFS blocked by deny | Observability installed before deny assignment | prometheus-server pod stuck | Reordered workflow: CreateCluster → DenyAssignment → EnableObservability | Available (fixed) |
| Helm upgrade kills operations | Pod restart during active provisioning | Syncops operation aborted | Never deploy during active provisioning; maxSurge:0 | Available (documented) |

---

## 5. Business & Adoption Telemetry

> Reports include Business (Adoption/Usage) and Product Group (Performance/Errors/Health). Telemetry and Dev-Ops/Tools are TBD.

### 5.1 Business Metrics (Engineering Reports) — Status: TBD

> **Note**: Business metrics will be sourced from the Geneva → Kusto pipeline once operational. Exact queries and dashboards TBD.

| Metric | Source | Frequency | Status |
|---|---|---|---|
| Active collaborations (by subscription, tenant, region) | Cosmos DB | Daily | TBD |
| New collaborations created (count, trend) | RP Worker telemetry | Daily | TBD |
| Collaborations deleted | RP Worker telemetry | Daily | TBD |
| Unique tenants/subscriptions using ACCR | ARM activity log | Weekly | TBD |
| Query execution count | Frontend Service telemetry | Daily | TBD |
| Data volume processed (rows read/written) | Frontend Service telemetry | Daily | TBD |
| Average collaboration lifetime | Cosmos DB + ARM | Weekly | TBD |
| Regional distribution | RP Worker telemetry | Weekly | TBD |
| Workload types enabled | RP Worker telemetry | Weekly | TBD |

### 5.2 Engineering Health Reports — Status: TBD

> **Note**: Engineering reports will be sourced from the Geneva → Kusto pipeline once operational. Exact queries and dashboards TBD.

| Report | Metrics | Frequency | Status |
|---|---|---|---|
| Service Availability | RP/Frontend/CM uptime, error rates | Hourly | TBD |
| Provisioning Performance | Create/EnableWorkload p50/p95/p99 durations | Daily | TBD |
| Failure Analysis | Failure rate by task, by region, by error type | Daily | TBD |
| CACI Capacity | FailedCreatePodSandBox rate by region | Daily | TBD |
| Resource Usage | AKS node count, CPU/memory utilization across AME clusters | Daily | TBD |
| Dependency Health | MOBO success rate, ARM latency, CCF availability | Daily | TBD |

---

## 6. Auto-Healing & Self-Recovery

> Customer has auto-recovery of Consortium. Product Group has auto-upgrades (TBD).

### 6.1 Existing

| Mechanism | Scope | Status |
|---|---|---|
| Consortium auto-recovery | CCF node restart | Available |
| MOBO failure recovery handler | MRG creation | Available |
| Retry on HttpRequestWithStatusException 403 | RP Worker | Available |
| Retry on CosmosException 408/429/503 | RP Worker | Available |

### 6.2 Recommended Additions

| Mechanism | Scope | Description | Status |
|---|---|---|---|
| Auto-retry failed RBAC propagation | RP Worker | Increase wait with exponential backoff, 20 iterations | Planned |
| Auto-recover stuck Provisioning state | RP monitoring worker | If Provisioning > 60 min, check if underlying resources exist and finalize | TBD/NotPlanned |
| Auto-cleanup orphaned MRGs | RP monitoring worker | Periodic scan for CleanRoom_* RGs without parent collaboration | TBD/NotPlanned |
| Auto-upgrade cluster images | Per collaboration | Coordinated rolling update of collaboration AKS cluster images | TBD/NotPlanned |
| Health state recovery | RP monitoring worker | Auto-restart unhealthy pods in collaboration cluster | TBD/NotPlanned |

---

## 7. Reference: Azure Standard Patterns Used

| Pattern | Azure Standard | ACCR Implementation | Status |
|---|---|---|---|
| Platform Metrics | Azure Monitor auto-collected | AKS metrics for all clusters | Planned |
| Resource Logs | Diagnostic Settings → Log Analytics | AKS control plane logs, future: Collaboration resource logs | Planned |
| Activity Log | Automatic for ARM resources | Already collected for Collaboration CRUD operations | Available |
| Container Insights | AKS add-on | ContainerLogV2, KubeEvents, Perf for AME clusters | Planned |
| Prometheus + Grafana | Managed Prometheus + Azure Managed Grafana | Prometheus deployed in collaboration clusters; Grafana TBD | Available (Prometheus) / TBD (Grafana) |
| Resource Health | Azure Resource Health | Add for Collaboration resource type | TBD/NotPlanned |
| Recommended Alerts | Alert rule templates (like AKS has) | Create for Collaboration resource type | Planned |
| Geneva/MDM | AME standard for 1P services | Custom metrics for all 3 services | Planned |
| Geneva → Kusto Pipeline | AME standard for log analysis | Logs + Metrics → Kusto for debugging, alerts, monitors | Planned |
| ICM | AME standard for incidents | Alert → ICM routing | Planned |
| Application Insights | OpenTelemetry-based APM | Optional for Frontend Service | TBD/NotPlanned |
