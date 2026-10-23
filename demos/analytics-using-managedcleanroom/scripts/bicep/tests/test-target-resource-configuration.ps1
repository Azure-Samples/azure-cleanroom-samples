$ErrorActionPreference = "Stop"
$deploymentScript = Join-Path $PSScriptRoot "../deploy-managed-cleanroom.ps1"
$testDirectory = Join-Path ([System.IO.Path]::GetTempPath()) "cleanroom target config $([guid]::NewGuid())"
$previousExitCode = $global:LASTEXITCODE

function Assert-Equal {
    param($Actual, $Expected, [string]$Message)
    if ($Actual -cne $Expected) {
        throw "${Message}: expected '$Expected', got '$Actual'."
    }
}

function Invoke-Scenario {
    param([hashtable]$Parameters, [string]$FailureMode = "", [switch]$Existing)
    $state = [pscustomobject]@{
        Calls = [System.Collections.Generic.List[string]]::new()
        Configuration = $null
        ParameterFile = $null
        Deployed = $false
        Error = $null
        Output = @()
    }

    function az {
        $command = $args -join " "
        $state.Calls.Add($command)
        $global:LASTEXITCODE = 0
        switch -Wildcard ($command) {
            "account show *" { '{"id":"offline-test-subscription"}'; return }
            "group create *" { return }
            "provider register *" { return }
            "deployment group create *" {
                $parameter = @($args | Where-Object { $_ -like "targetResourceConfiguration=@*" })
                Assert-Equal $parameter.Count 1 "Expected one JSON object parameter"
                $state.ParameterFile = $parameter[0].Substring("targetResourceConfiguration=@".Length)
                Assert-Equal (Test-Path -LiteralPath $state.ParameterFile) $true "Parameter file exists during deployment"
                $state.Configuration = Get-Content -LiteralPath $state.ParameterFile -Raw | ConvertFrom-Json
                if ($FailureMode -eq "ExitCode") { $global:LASTEXITCODE = 1; return }
                if ($FailureMode -eq "Throw") { throw "Simulated deployment exception" }
                $state.Deployed = $true
                return
            }
            "rest --method GET *" {
                if ($Existing -or $state.Deployed) {
                    '{"id":"/offline/collaboration","properties":{"workloads":[{"workloadType":"Analytics","endpoint":"https://example.test"}],"health":{"healthState":"Ok"}}}'
                } else {
                    '{}'
                }
                return
            }
            "rest --method POST */enableWorkload?*" { return }
            default { throw "Unexpected Azure CLI call: $command" }
        }
    }

    try {
        $state.Output = @(& (Join-Path $testDirectory "deploy-managed-cleanroom.ps1") `
            -resourceGroup offline-rg -collaborationName offline-collab @Parameters 6>&1)
    } catch {
        $state.Error = $_
    }
    return $state
}

try {
    New-Item -ItemType Directory -Path $testDirectory | Out-Null
    Copy-Item -LiteralPath $deploymentScript -Destination $testDirectory
    $cases = @(
        @{ Name = "defaults"; Parameters = @{}; Sku = "Standard_D4ds_v5"; Nodes = 3 },
        @{ Name = "legacy sizing"; Parameters = @{ aksSku = "Standard_D16ds_v5"; nodePoolSize = 6 }; Sku = "Standard_D16ds_v5"; Nodes = 6 },
        @{ Name = "empty object"; Parameters = @{ targetResourceConfiguration = @{} }; Sku = "Standard_D4ds_v5"; Nodes = 3 },
        @{ Name = "IP tags with legacy sizing"; Parameters = @{ aksSku = "Standard_D8ds_v5"; nodePoolSize = 4; targetResourceConfiguration = @{ iPTagConfiguration = @{ type = "FirstPartyUsage"; value = "/AzureCleanRoomsNonProd" } } }; Sku = "Standard_D8ds_v5"; Nodes = 4 },
        @{ Name = "object overrides legacy sizing"; Parameters = @{ aksSku = "Standard_D4ds_v5"; nodePoolSize = 3; targetResourceConfiguration = @{ aksSku = "Standard_D32ds_v5"; nodePoolSize = 10 } }; Sku = "Standard_D32ds_v5"; Nodes = 10 },
        @{ Name = "partial object retains defaults"; Parameters = @{ targetResourceConfiguration = @{ nodePoolSize = 5 } }; Sku = "Standard_D4ds_v5"; Nodes = 5 }
    )
    foreach ($case in $cases) {
        $parameters = $case.Parameters.Clone()
        $parameters.dryRun = $true
        $state = Invoke-Scenario -Parameters $parameters
        if ($state.Error) { throw $state.Error }
        Assert-Equal $state.Calls.Count 1 "$($case.Name): only account lookup in dry run"
        $prefix = "[DRY-RUN]   targetResourceConfiguration="
        $line = @($state.Output | Where-Object { "$($_)".StartsWith($prefix) })
        Assert-Equal $line.Count 1 "$($case.Name): effective configuration printed"
        $configuration = "$($line[0])".Substring($prefix.Length) | ConvertFrom-Json
        Assert-Equal $configuration.aksSku $case.Sku "$($case.Name): SKU"
        Assert-Equal $configuration.nodePoolSize $case.Nodes "$($case.Name): node count"
        if ($case.Parameters.targetResourceConfiguration.iPTagConfiguration) {
            Assert-Equal $configuration.iPTagConfiguration.value "/AzureCleanRoomsNonProd" "IP tag in dry-run output"
        }
        Write-Host "PASS: $($case.Name)"
    }

    foreach ($invalid in @(
        @{ aksSku = "UnsupportedSku" },
        @{ aksSku = $null },
        @{ nodePoolSize = 2 },
        @{ nodePoolSize = 11 },
        @{ nodePoolSize = $null }
    )) {
        $state = Invoke-Scenario -Parameters @{ targetResourceConfiguration = $invalid; dryRun = $true }
        if (-not $state.Error) { throw "Invalid configuration was accepted: $($invalid | ConvertTo-Json -Compress)" }
        Assert-Equal $state.Calls.Count 0 "Invalid sizing rejected before Azure CLI calls"
        Write-Host "PASS: invalid configuration rejected: $($invalid | ConvertTo-Json -Compress)"
    }

    $tagValue = '/tag with spaces, "quotes" and \slashes'
    $configuration = @{ aksSku = "Standard_D8ds_v5"; nodePoolSize = 5; iPTagConfiguration = @{ type = "FirstPartyUsage"; value = $tagValue } }
    foreach ($failureMode in @("", "ExitCode", "Throw")) {
        $state = Invoke-Scenario -Parameters @{ targetResourceConfiguration = $configuration } -FailureMode $failureMode
        if ($failureMode -eq "") {
            if ($state.Error) { throw $state.Error }
            Assert-Equal $state.Deployed $true "Successful deployment"
        } else {
            if (-not $state.Error) { throw "Deployment failure was not propagated: $failureMode" }
            $expectedMessage = if ($failureMode -eq "ExitCode") { "Bicep deployment failed*" } else { "Simulated deployment exception*" }
            if ($state.Error.Exception.Message -notlike $expectedMessage) { throw $state.Error }
        }
        Assert-Equal $state.Configuration.aksSku "Standard_D8ds_v5" "SKU forwarded in JSON file"
        Assert-Equal $state.Configuration.nodePoolSize 5 "Node count forwarded in JSON file"
        Assert-Equal $state.Configuration.iPTagConfiguration.type "FirstPartyUsage" "IP tag type forwarded in JSON file"
        Assert-Equal $state.Configuration.iPTagConfiguration.value $tagValue "Special characters round-trip in JSON file"
        Assert-Equal (Test-Path -LiteralPath $state.ParameterFile) $false "Temporary file removed after deployment"
        Assert-Equal $configuration.Count 3 "Caller configuration unchanged"
        Write-Host "PASS: JSON forwarding and cleanup (failure mode: '$failureMode')"
    }

    $state = Invoke-Scenario -Parameters @{ targetResourceConfiguration = $configuration } -Existing
    if ($state.Error) { throw $state.Error }
    Assert-Equal $state.Deployed $false "Existing collaboration skipped"
    Assert-Equal $state.ParameterFile $null "No parameter file for skipped deployment"
    Write-Host "PASS: existing collaboration skipped"
    Write-Host "All 15 offline target-resource-configuration tests passed."
}
finally {
    Remove-Item -LiteralPath $testDirectory -Recurse -Force -ErrorAction SilentlyContinue
    $global:LASTEXITCODE = $previousExitCode
}