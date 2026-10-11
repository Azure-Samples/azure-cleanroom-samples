# ACCR Monitoring & Observability — Detailed Specs

This folder contains detailed implementation specs for each area of the [ACCR Monitoring Plan](../monitoring-plan.md).

## Architecture Context

ACCR has **3 AME-hosted services** and **per-customer collaboration deployments**:

| Component | Type | AKS Cluster (Test) | Namespace |
|-----------|------|-------------------|-----------|
| RP (CleanRoomService) | AME Service | `aks-testwestus` | `cleanroom-ns` |
| Frontend Service | AME Service | `aks-frontend-testwestus` | `frontendns` |
| Consortium Manager | AME Service | `consortiummanager-aks-testwestus` | `consortiummanager-ns` |
| Collaboration AKS | Customer Deployment | `{name}-aks` (per collaboration) | `analytics`, `cleanroom-system`, `telemetry` |
| CCF Consortium (CACI) | Customer Deployment | N/A (ACI container group) | N/A |

**Telemetry Pipeline:** App → OTLP gRPC → ILB:4317 → Geneva DaemonSet → Geneva Storage → Kusto → Dgrep/Dashboards

## Sections

| # | Section | Folder | Description |
|---|---------|--------|-------------|
| 1 | [AME Service Monitoring](1-ame-service-monitoring/) | `1-ame-service-monitoring/` | Platform metrics, synthetic runners, app telemetry, resource logs, container insights for RP/Frontend/CM |
| 2 | [Customer Collaboration Monitoring](2-customer-collaboration-monitoring/) | `2-customer-collaboration-monitoring/` | RP-side health, customer-facing APIs, collaboration AKS observability |
| 3 | [Alerting Strategy](3-alerting-strategy/) | `3-alerting-strategy/` | PG alerts (ICM), customer alerts (Azure Monitor), CSS alerts |
| 4 | [Diagnostics & Debugging](4-diagnostics-debugging/) | `4-diagnostics-debugging/` | PG debugging workflow, Kusto queries, CSS workflow, failure patterns |
| 5 | [Business Telemetry](5-business-telemetry/) | `5-business-telemetry/` | Adoption metrics, engineering health reports |
| 6 | [Auto-Healing](6-auto-healing/) | `6-auto-healing/` | Existing self-recovery, planned additions |
| 7 | [Reference](7-reference/) | `7-reference/` | Azure standard patterns used |

## Dependencies

```
Geneva OTLP Pipeline (PR #15773998) ──► Geneva → Kusto (Vaiddhe/Yash) ──► Kusto Queries/Alerts
                                                                           ├── Section 3 (Alerting)
                                                                           ├── Section 4 (Diagnostics)
                                                                           └── Section 5 (Business)

App Code Instrumentation (Section 1.3) ──► Metrics + Structured Logs ──► All downstream sections
```

## Personas

| Persona | What They Need | Primary Sections |
|---------|---------------|-----------------|
| Product Group (Engineering) | Detect issues, debug, measure performance | 1, 3.1, 4.1, 5.2 |
| Customers | Collaboration health, query status | 2.2, 3.2 |
| CSS (Support) | Triage tickets without escalating | 3.3, 4.3 |
