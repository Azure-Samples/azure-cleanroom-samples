# Azure Confidential Clean Rooms — Samples

[Azure Confidential Clean Rooms (ACCR)](https://learn.microsoft.com/en-us/azure/confidential-computing/confidential-clean-rooms)
enable multiple parties to collaboratively analyze sensitive data without
exposing it to each other. Data is processed inside a hardware-attested
confidential environment orchestrated through Azure Resource Manager, ensuring
that no party — including Microsoft — can access the raw data.

The **managed** flavour of ACCR handles infrastructure provisioning and policy enforcement automatically so that participants can focus
on defining datasets and queries rather than managing clean-room infrastructure.

## Samples

### Managed Analytics (Spark SQL)

Run big-data analytics across multi-party datasets using Spark SQL inside a
managed clean room. Two end-to-end walkthroughs are available:

| Guide | Description |
|---|---|
| [Azure CLI (`az managedcleanroom`)](demos/analytics-using-managedcleanroom/README-CLI.md) | Walkthrough using the `managedcleanroom` Azure CLI extension |
| [REST API (`az rest` + `Invoke-RestMethod`)](demos/analytics-using-managedcleanroom/README-API.md) | Walkthrough using `az rest` for ARM operations and `Invoke-RestMethod` for frontend operations |

For troubleshooting, see the [TSG](demos/analytics-using-managedcleanroom/tsg/index.md).
