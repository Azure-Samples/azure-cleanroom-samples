function Assert-CleanroomAttestation {
    param (
        [Parameter(Mandatory = $true)]
        [string[]]$containerImages,

        [string]$containerTag = "2.0.0",

        [string]$samplesRoot = "/home/samples",
        [string]$privateDir = "$samplesRoot/demo-resources/private"
    )
    #https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
    $ErrorActionPreference = 'Stop'
    $PSNativeCommandUseErrorActionPreference = $true

    $repo = "Azure/azure-cleanroom"
    $workflowPath = ".github/workflows/release.yml"
    $workflowRef = "refs/heads/main"

    # Fetch the commit ID corresponding to the container versions
    $tagDetails = curl -L "https://api.github.com/repos/$repo/tags" | ConvertFrom-Json | Where-Object {$_.name -eq $containerTag}
    $commitId = $tagDetails.commit.sha

    # Verify attestations for container images offline:
    # https://docs.github.com/en/actions/security-for-github-actions/using-artifact-attestations/verifying-attestations-offline
    gh attestation trusted-root > "$privateDir/trusted_root.jsonl"
    if ($LASTEXITCODE -ne 0) {
        Write-Log Error "Failed to fetch trusted root."
        exit 1
    }
  
    foreach ($image in $containerImages) {
        Write-Log Verbose "Verifying image '$image'..."
        $digest = $image.Split("@")[1]
        rm -f "$privateDir/$digest.jsonl"

        # Fetch attestation from GitHub API.
        $attestationResult = curl -L "https://api.github.com/repos/$repo/attestations/$digest" | ConvertFrom-Json
        if ($attestationResult.Length -ne 1) {
            Write-Log Error "Invalid attestation result for image '$image'"
            exit 1
        }

        # Convert the attestation to JSONL format, required for offline verification:
        # https://docs.github.com/en/actions/security-for-github-actions/using-artifact-attestations/verifying-attestations-offline
        $attestationResult = $attestationResult[0]
        foreach ($attestation in $attestationResult.attestations) {
            $attestation.bundle | ConvertTo-Json -Depth 100 -Compress | Out-File "$privateDir/$digest.jsonl" -Append
        }

        $attestationVerificationResult = gh attestation verify `
            "oci://$image" `
            --repo $repo `
            --bundle "$privateDir/$digest.jsonl" `
            --custom-trusted-root "$privateDir/trusted_root.jsonl" `
            --format json | ConvertFrom-Json
        
        if ($LASTEXITCODE -ne 0) {
            Write-Log Error "Failed to verify image '$image'."
            exit 1
        }

        $imagePath = $image.Split("@")[0]
        $attestation = $attestationVerificationResult | Where-Object {$_.verificationResult.statement.subject.name -eq $imagePath}

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
