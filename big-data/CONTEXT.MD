## Overview
Azure Confidential Clean Rooms (ACCR) enable secure sharing of privacy-sensitive data by collaborating customer organizations by using confidential computing and Trusted Execution Environments (TEEs) to protect data throughout its lifecycle. This is in preview today and the sample scenarios for the same are available [here](../README.md).

We’re now launching an improved version of ACCR focused on big data analytics, which may be useful for customers across all industries, but particularly those in marketing and advertising who may want to analyze large amounts of privacy-sensitive data across different organizations securely using SQL queries. This version allows Spark SQL queries to be executed in a distributed, confidential compute based Apache Spark engine (OSS Spark Engine offering, integrated into the ACCR), thus enabling big data analytics in a fully secure and verifiable manner.

Based on the feedback we’ve received from preview, we’ve also simplified the clean room setup/onboarding, membership and contract management in this version by handling these automatically from the ACCR end. The following are some of the key aspects of this version:
- Spark Engine Integration
- Voting on the queries before execution
- User based collaborations instead of certificates
- Support for Non-Azure collaborators
- Verifiability / Audit
- Option for encryption of datasets
- Privacy

## Personas Involved
### Clean room owner [CO], who:
- Sets up the clean room – hence needs an Azure subscription
### Clean room user [CU], who:
- Can share data and/or queries to the clean room collaboration
- Needs to have a Microsoft (work/school or personal) account to authenticate themselves to use the clean room

## Phased Approach

|                      | Release-1 [Now]                                                                 | Release-2                                                                 | Preview-1                        | Preview-2                        |
|----------------------|----------------------------------------------------------------------------------|---------------------------------------------------------------------------|----------------------------------|----------------------------------|
| **Capabilities**     | - Clean room setup, Member Invitation/Acceptance<br>- Azure Blob Storage & AWS S3 as data sources/sinks for input/output<br>- Ability to propose and approve Spark SQL queries<br>- Ability to execute Spark SQL queries in a confidential manner | All Release-1 capabilities and<br>- Input k-min & output k-min privacy-preserving functions<br>- Support for custom parameters in queries | All Release-2 capabilities       | All Release-2 capabilities       |
| **Deployment/Management** | PS Scripts and Local Docker Containers                                         | PS Scripts and Local Docker Containers                                    | Azure CLI / PS Scripts + Frontend API (without UI) using cleanroom RP | Azure Portal + Frontend UI and/or Azure CLI/PS Scripts |

## Sample Scenario
Please see [here](README.md) for the steps to run clean rooms for big data analytics using Spark.
