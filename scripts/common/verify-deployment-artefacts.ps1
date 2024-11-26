function Assert-CleanroomAttestation {
    param (
        [Parameter(Mandatory = $true)]
        [string[]]$containerImages,

        [string]$containerTag = "2.0.0"
    )
    #https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
    $ErrorActionPreference = 'Stop'
    $PSNativeCommandUseErrorActionPreference = $true

    $repo = "Azure/azure-cleanroom"
    $workflowPath = ".github/workflows/release.yml"
    $workflowRef = "refs/heads/main"

    Write-Log Verbose `
        "Logging in to GitHub to verify container image attestations..."
    gh auth login --web

    # Fetch the commit ID corresponding to the container versions
    $tagDetails = gh api "repos/$repo/tags" | ConvertFrom-Json | Where-Object {$_.name -eq $containerTag}
    $commitId = $tagDetails.commit.sha
  
    foreach ($image in $containerImages) {
        Write-Log Verbose "Verifying image '$image'..."
        $attestationResult = gh attestation verify `
            "oci://$image" `
            --repo $repo --format json | ConvertFrom-Json
        
        if ($LASTEXITCODE -ne 0) {
            Write-Log Error "Failed to verify image '$image'."
            exit 1
        }

        $imagePath = $image.Split("@")[0]
        $attestation = $attestationResult | Where-Object {$_.verificationResult.statement.subject.name -eq $imagePath}

        if ($attestation.verificationResult.statement.predicate.buildDefinition.resolvedDependencies.digest.gitCommit -ne $commitId) {
            Write-Log Error "Images in the deployment template are not from the latest commit."
            exit 1
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
