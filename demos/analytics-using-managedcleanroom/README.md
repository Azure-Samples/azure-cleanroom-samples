# Azure Confidential Clean Rooms — Managed Analytics

Azure Confidential Clean Rooms (ACCR) enable multiple parties to collaborate on
sensitive data without exposing it to each other. The **managed analytics**
workload uses Spark SQL to join, aggregate, and analyze datasets
inside a hardware-attested clean room — all orchestrated through Azure Resource
Manager and a secure frontend service.

## Samples

| Guide | Description |
|---|---|
| [REST API (az rest + Invoke-RestMethod)](README-API.md) | End-to-end walkthrough using `az rest` for ARM operations and `Invoke-RestMethod` for frontend operations |
| [CLI (az managedcleanroom)](README-CLI.md) | End-to-end walkthrough using the `managedcleanroom` Azure CLI extension |

## Troubleshooting

See the [TSG index](tsg/index.md) for diagnostics, debugging, and known issues.
