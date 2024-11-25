function Verify-Template {
    param (
        [Parameter(Mandatory = $true)]
        [string]$deploymentTemplatePath
    )
    #https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
    $ErrorActionPreference = 'Stop'
    $PSNativeCommandUseErrorActionPreference = $true

    Import-Module $PSScriptRoot/../common/common.psm1
    Import-Module $PSScriptRoot/../azure-helpers/azure-helpers.psm1 -Force -DisableNameChecking

    $repo = "Azure/azure-cleanroom-private"
    $workflowPath = ".github/workflows/release.yml"
    $workflowRef = "refs/heads/release"
    $commitId = ""

    $deploymentTemplate = Get-Content $deploymentTemplatePath | ConvertFrom-Json

    $resources = $deploymentTemplate.resources | Where-Object { $_.type -eq "Microsoft.ContainerInstance/containerGroups"}

    if ($null -eq $resources) {
        Write-Log Error "No container groups found in the deployment template."
        exit 1
    }

    $containerImages = $resources.properties.containers |`
        ForEach-Object { $_.properties.image } |`
        Select-Object -Unique

    $containerImages += $resources.properties.initContainers |`
        ForEach-Object { $_.properties.image } |`
        Select-Object -Unique

    foreach ($image in $containerImages) {
        Write-Log Verbose "Verifying image '$image'..."
        $attestationResult = gh attestation verify `
            "oci://$image" `
            --repo $repo --format json | ConvertFrom-Json
        CheckLastExitCode

        $imagePath = $image.Split("@")[0]
        $attestation = $attestationResult | Where-Object {$_.verificationResult.statement.subject.name -eq $imagePath}

        if ($commitId -eq "") {
            $commitId = $attestation.verificationResult.statement.predicate.buildDefinition.resolvedDependencies.digest.gitCommit
        }
        else {
            if ($commitId -ne $attestation.verificationResult.statement.predicate.buildDefinition.resolvedDependencies.digest.gitCommit) {
                Write-Log Error "Images in the deployment template are from different commits."
                exit 1
            }
        }

        if ($attestation.verificationResult.statement.predicate.buildDefinition.externalParameters.workflow.path -ne $workflowPath) {
            Write-Log Error "Images in the deployment template are from different workflows."
            exit 1
        }

        if ($attestation.verificationResult.statement.predicate.buildDefinition.externalParameters.workflow.ref -ne $workflowRef) {
            Write-Log Error "Images in the deployment template are from different branches."
            exit 1
        }
    }

    Write-Log Information "All images are built off the commit '$commitId'."
}
