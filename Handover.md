# Azure Managed Cleanroom Agentic Experience
## Project Handover Document (Cleaned)

**Author:** Shreya Sangwa  
**Date:** July 2026  
**Primary Goal:** Enable an agentic-first Azure Managed Cleanroom workflow through Azure MCP tooling and skills.

---

# 1. Executive Summary

The implementation foundation is complete:

- Managed Cleanroom MCP toolset implemented in Azure MCP server codebase.
- Management plane and data plane SDK dependencies integrated.
- Unit and live-test paths established.
- Skills authored and validated locally.

Primary remaining work for takeover:

1. Publish SDK packages (management + data plane).
2. Complete review and merge workflow for the 3 PRs listed below.
3. Raise and merge skills PR in the destination skills repo.
4. Replace local/dev SDK references with published package versions.
5. Run final end-to-end validation through MCP client flow.

Implementation context at handover:

- Upstream repo: https://github.com/microsoft/mcp
- Working fork: https://github.com/ShreyaSangwa/mcp
- Working branch: `ManagedCleanrooms`
- Snapshot commit in local workspace: `9067e39bd`

---

# 2. Current Status

## 2.1 Completed

- Management plane SDK generated, validated, integrated.
- Data plane SDK generated, validated, integrated.
- Managed Cleanroom tool area implemented with commands and tests.
- Skills authored and validated locally.

## 2.2 Pending

- SDK publication and package finalization.
- PR review resolution and merge completion.
- Skills repository PR submission and merge.
- Full E2E validation after merge and package finalization.

---

# 3. Repositories and Deliverables

## 3.1 SDKs

| Area | Repository | Package | Status |
|---|---|---|---|
| Management plane | https://github.com/ShreyaSangwa/azure-sdk-for-net/tree/mgmt_sdk/sdk/cleanroom | `Azure.ResourceManager.CleanRoom` | Publishing pending |
| Data plane | https://github.com/azure-core/azure-cleanroom/tree/sdk_dataplane/src/sdk/clients/frontend-client | `Azure.Cleanroom.Analytics.Frontend.Client` | Publishing pending |

Notes:

- SDK sample integration PR link provided: https://github.com/Azure-Samples/azure-cleanroom-samples/pull/98

## 3.2 Azure MCP Server Toolset

- Repository: https://github.com/microsoft/mcp
- Toolset path: `tools/Azure.Mcp.Tools.ManagedCleanroom`

## 3.3 Skills

- Current authored skill source path:
  `tools/Azure.Mcp.Tools.ManagedCleanroom/skills/managedcleanroom-agent-support/SKILL.md`
- Destination skills repository: https://github.com/ShreyaSangwa/GitHub-Copilot-for-Azure/tree/feature/azure-managed-cleanroom-skill/plugin/skills/azure-managed-cleanroom
- Skills PR (open): https://github.com/ShreyaSangwa/GitHub-Copilot-for-Azure/pull/1

---

# 4. Open PRs

## 4.1 PR List

1. PR 1 (upstream, under review): https://github.com/microsoft/mcp/pull/2882  
   Scope: Initial Managed Cleanroom toolset and first command surface.
2. PR 2 (fork): https://github.com/ShreyaSangwa/mcp/pull/2  
   Scope: Adds two `get` commands (management plane + data plane).
3. PR 3 (fork): https://github.com/ShreyaSangwa/mcp/pull/3  
   Scope: Full Managed Cleanroom command surface.
4. PR 4 (skills repo, open): https://github.com/ShreyaSangwa/GitHub-Copilot-for-Azure/pull/1  
   Scope: Adds `azure-managed-cleanroom` skill, references, helper scripts, and tests.

## 4.2 PR Closure Checklist (applies to all three)

- Address reviewer comments.
- Re-run local build.
- Re-run unit tests.
- Re-run live tests when impacted.
- Update PR description with validation evidence.
- Resolve CI issues.
- Get approval and merge.

## 4.3 Merge Strategy Note

PR 2 and PR 3 are on fork branches while PR 1 is upstream. Before final merge, decide one of these approaches:

1. Cherry-pick scoped changes from PR 2/PR 3 onto the upstream PR branch.
2. Recreate a single consolidated upstream PR if review complexity is lower.

---

# 5. Quick Start for New Owner

## 5.1 Prerequisites

- Git
- .NET SDK used by Azure MCP server
- Azure CLI
- VS Code with MCP-capable client/extension
- Access to required Azure subscription/tenant/resource groups
- Access to SDK source repos and MCP repos

## 5.2 Clone

```bash
git clone https://github.com/Azure/azure-sdk-for-net.git
git clone https://github.com/microsoft/mcp.git
```

## 5.3 Authenticate

```bash
az login
az account set --subscription <subscription-id>
```

## 5.4 Build and Test

```bash
cd mcp
dotnet build Microsoft.Mcp.slnx

# Full test pass (long-running)
dotnet test Microsoft.Mcp.slnx

# Targeted Managed Cleanroom tests
dotnet test tools/Azure.Mcp.Tools.ManagedCleanroom/tests/Azure.Mcp.Tools.ManagedCleanroom.Tests/Azure.Mcp.Tools.ManagedCleanroom.Tests.csproj
```

## 5.4.1 After Making Command Changes

Use this sequence when you modify files under `tools/Azure.Mcp.Tools.ManagedCleanroom/src/Commands`.

```bash
cd mcp

# 1) Fast rebuild of the Managed Cleanroom toolset only
dotnet build tools/Azure.Mcp.Tools.ManagedCleanroom/src/Azure.Mcp.Tools.ManagedCleanroom.csproj

# 2) Rebuild Azure MCP server to ensure command registration and wiring still compile
dotnet build servers/Azure.Mcp.Server/src/Azure.Mcp.Server.csproj

# 3) Run the full Managed Cleanroom test project
dotnet test tools/Azure.Mcp.Tools.ManagedCleanroom/tests/Azure.Mcp.Tools.ManagedCleanroom.Tests/Azure.Mcp.Tools.ManagedCleanroom.Tests.csproj

# 4) Optional: run only command-focused tests for faster iteration
dotnet test tools/Azure.Mcp.Tools.ManagedCleanroom/tests/Azure.Mcp.Tools.ManagedCleanroom.Tests/Azure.Mcp.Tools.ManagedCleanroom.Tests.csproj --filter "FullyQualifiedName~ManagedCleanroomCommandTests"

# 5) Final pre-PR validation across the solution
dotnet build Microsoft.Mcp.slnx
```

## 5.5 Validate in MCP Client

Confirm:

- Managed Cleanroom tools are discoverable.
- Commands execute correctly.
- Responses are returned correctly.
- Skills route workflow correctly.

---

# 6. Execution Plan

## Phase 1: Publish SDKs

1. Confirm latest API spec.
2. Regenerate SDK if needed.
3. Build/test SDKs.
4. Publish packages.
5. Confirm package availability.

## Phase 2: Update MCP Dependencies

1. Replace local/dev SDK references with published package versions.
2. Restore/build/test.
3. Run affected live tests.

## Phase 3: Complete PR Merges

1. Resolve review comments.
2. Attach validation evidence.
3. Merge in agreed order.

## Phase 4: Skills PR

1. Confirm destination skills repo.
2. Skills PR raised: https://github.com/ShreyaSangwa/GitHub-Copilot-for-Azure/pull/1.
3. Address review comments.
4. Merge.

## Phase 5: Final E2E Validation

Run at least one full natural-language workflow:

- Dataset publish
- Query run
- Run status retrieval
- Audit event verification

---

# 7. Risks and Known Gotchas

## 7.1 Risks

- SDK publication delay blocks dependency finalization.
- PR review/merge sequencing can cause scope drift.
- Skills repo destination is still unconfirmed.

---

# 8. Validation and Evidence

Before requesting final review:

- Build passes.
- Unit tests pass.
- Live tests pass (or justified exceptions documented).
- PR descriptions include command/test evidence.
- No unrelated file changes.

Live test recording guide:

- https://github.com/microsoft/mcp/blob/main/docs/recorded-tests.md

---

# 9. Takeover Checklist

- [ ] Confirm SDK source branches and PR links.
- [ ] Confirm destination skills repository.
- [ ] Confirm merge strategy for PR 1/2/3.
- [ ] Publish SDK packages.
- [ ] Update MCP package references.
- [ ] Build + unit test + live test pass.
- [ ] Merge MCP PRs.
- [ ] Merge skills PR (#1 in GitHub-Copilot-for-Azure).
- [ ] Complete final E2E validation.
- [ ] Update this document with final package versions, owners, and merged PR links.

---

# 10. Final References

- ACCR PM spec: https://microsoftapc-my.sharepoint.com/personal/dejv_microsoft_com/Documents/TrustedVM/ACCR%20Agentic%20Experience.docx?d=wb0e8fe160c064adb8ef676e491e75747&nav=eyJjIjo3NDU2MTM5MjF9&e=3e2lGw-qEG3Z7tzdC5PcQ&at=17&CT=1779341493102&OR=OWA-NT-Mail&CID=34a81c02-0119-74e3-6445-ff023811a810&isSPOFile=1
- Engineering plan: https://github.com/azure-core/azure-cleanroom/blob/user/vaidmishra/readings/docs/azure-mcp-server-dev-spec.md
- Samples repo with SDK integration: https://github.com/ShreyaSangwa/azure-cleanroom-samples/tree/sample_branch/packages
- Dataplane SDK repo: https://github.com/azure-core/azure-cleanroom/tree/sdk_dataplane/src/sdk/clients/frontend-client
- Dataplane/management sample PR link: https://github.com/Azure-Samples/azure-cleanroom-samples/pull/98
- Management-plane SDK repo: https://github.com/ShreyaSangwa/azure-sdk-for-net/tree/mgmt_sdk/sdk/cleanroom
- Azure MCP repo: https://github.com/microsoft/mcp
- Working fork: https://github.com/ShreyaSangwa/mcp
- MCP server toolset path in fork: https://github.com/ShreyaSangwa/mcp/tree/ManagedCleanrooms/tools/Azure.Mcp.Tools.ManagedCleanroom
- Skills repo path: https://github.com/ShreyaSangwa/GitHub-Copilot-for-Azure/tree/feature/azure-managed-cleanroom-skill/plugin/skills/azure-managed-cleanroom
- Skills PR (open): https://github.com/ShreyaSangwa/GitHub-Copilot-for-Azure/pull/1
- Toolset docs: https://github.com/microsoft/mcp/tree/main/tools/Azure.Mcp.Tools.ManagedCleanroom/docs
- Recorded tests guide: https://github.com/microsoft/mcp/blob/main/docs/recorded-tests.md
- Contribution guide: https://github.com/microsoft/mcp/blob/main/CONTRIBUTING.md

---

# 11. Handover Note

The solution is functionally in place; remaining work is publication, merge completion, and final operationalization. Prioritize SDK publish + PR merge + skills PR + E2E validation in that order.
