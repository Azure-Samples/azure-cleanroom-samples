param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("northwind", "woodgrove", "operator", IgnoreCase = $false)]
    [string]$persona,

    [Parameter(Mandatory = $true)]
    [ValidateSet("analytics-sse", "analytics-s3-sse", "analytics-cpk")]
    [string]$demo,

    [string]$resourceGroup = "",
    [string]$resourceGroupLocation = "westeurope",

    [string]$imageName = "azure-cleanroom-samples",
    [string]$dockerFileDir = "./big-data/docker",

    [string]$accessTokenProviderName = "$imageName-credential-proxy",
    [string]$ccfProviderName = "$imageName-ccf-provider",
    [string]$cleanroomClusterProviderName = "$imageName-cluster-provider",
    [string]$telemetryDashboardName = "$imageName-telemetry",
    [string]$shellContainerName = "$imageName-shell-$persona",
    
    [switch]$overwrite,
    [switch]$shareCredentials,

    [string]$repo = "cleanroomemuprregistry.azurecr.io",
    [string]$tag = "19071407933"
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/scripts/common/common.psm1

function Get-Confirmation {
    param (
        [string]$Message = "Are you sure?",
        [string]$YesLabel = "Yes",
        [string]$NoLabel = "No"
    )

    do {
        $choice = Read-Host "$($PSStyle.Bold)$Message ('$YesLabel'/'$NoLabel')$($PSStyle.Reset)"
        $choice = $choice.ToLower()
        switch ($choice) {
            $YesLabel.ToLower() {
                $response = $choice
                break
            }
            $NoLabel.ToLower() {
                $response = $choice
                break
            }
            default {
                Write-Log Error `
                    "Invalid input. Please enter '$YesLabel' or '$NoLabel'."
            }
        }
    } while ($response -ne $YesLabel.ToLower() -and $response -ne $NoLabel.ToLower())

    return ($response -eq $YesLabel.ToLower())
}

function Get-SemanticVersionFromTag {
    param (
        [string]$tag
    )

    # Check if the tag is in the format of x.y.z (with optional parts)
    if ($tag -match '^\d+\.\d+\.\d+') {
        $parts = $tag.Split('.')
        $major = $parts[0]
        $minor = $parts[1]
        $patch = $parts[2]
        $extra = ""
        if ($parts.Count -eq 4) {
            $extra = "-" + $parts[3]
        }

        return "$major.$minor.$patch$extra"
    }
    
    # If not a version format, truncate to 8 characters and use as suffix
    $truncatedTag = $tag
    if ($tag.Length -gt 8) {
        $truncatedTag = $tag.Substring(0, 8)
    }
    
    # Return the formatted version. Add a "v" prefix below as $truncatedTag like 0207 gives
    # error "Version segment starts with 0".
    return "1.0.42-v$truncatedTag"
}

$hostBase = "$pwd/demo-resources"
$sharedBase = "$hostBase/shared"
$personaBase = "$hostBase/$persona"
$virtualBase = "/home/samples/demo-resources"

#
# Create host directories shared by sample environment containers for all persona.
#
$publicDir = "$sharedBase/public"
New-Item -ItemType Directory -Force -Path $publicDir
$telemetryDir = "$sharedBase/telemetry"
New-Item -ItemType Directory -Force -Path $telemetryDir

#
# Create host directories private to sample environment containers per persona.
#
$privateDir = "private"
New-Item -ItemType Directory -Force -Path "$personaBase/$privateDir"
$secretDir = "secret"
New-Item -ItemType Directory -Force -Path "$personaBase/$secretDir"

#
# Launch credential proxy for operator or if sharing credentials.
#
if ($persona -eq "woodgrove" -and $demo -eq "analytics-s3-sse") {
    Write-Log Verbose `
        "No credential sharing infrastructure required for '$persona' for '$demo' demo..."
}
else {
    if ($shareCredentials -or ($persona -eq "operator")) {
        & {
            Write-Log OperationStarted `
                "Setting up credential sharing infrastructure..."

            # Create a bridge network to host the credential proxy.
            $networkName = "$imageName-network"
            $network = (docker network ls --filter "name=^$networkName$" --format 'json') | ConvertFrom-Json
            if ($null -eq $network) {
                Write-Log Verbose `
                    "Creating docker network '$networkName'..."
                docker network create $networkName
            }

            # Bring up a credential proxy container.
            $containerName = $accessTokenProviderName
            $container = (docker container ls -a --filter "name=^$containerName$" --format 'json') | ConvertFrom-Json
            if ($null -eq $container) {
                # The latest version of the proxy image is giving a JsonDeserialization error.
                # TODO: Use the latest version of the proxy image once the issue is fixed.
                $proxyImage = "workleap/azure-cli-credentials-proxy:1.2.5"
                Write-Log Verbose `
                    "Creating credential proxy '$containerName' using image '$proxyImage'..."
                docker container create `
                    -p "0:8080" `
                    --network $networkName `
                    --name $containerName `
                    $proxyImage
            }
            else {
                Write-Log Warning `
                    "Reusing existing credential proxy container '$($container.Names)'" `
                    "(ID: $($container.ID))."
            }

            docker container start $containerName

            # Interactively login to proxy if required.
            Write-Log Verbose `
                "Checking validity of Azure access token..."
            & {
                # Disable $PSNativeCommandUseErrorActionPreference for this scriptblock
                $PSNativeCommandUseErrorActionPreference = $false
                $(docker exec $containerName sh -c "az account get-access-token" 1>$null)
            }

            if (0 -ne $LASTEXITCODE) {
                Write-Log OperationStarted `
                    "Logging into Azure..."
                docker exec -it $containerName sh -c "az login"
            }

            docker exec $containerName sh -c "az account show"
        }

        # Fetch credential proxy host port details.
        $credentialProxyPort = (docker port $accessTokenProviderName 8080).Split(':')[1]
        $credentialProxyEndpoint = "http://host.docker.internal:$credentialProxyPort/token"
        Write-Log OperationCompleted `
            "Credential sharing infrastructure deployed at '$credentialProxyEndpoint'."
    }
}

#
# Launch CCF provider for 'operator' using shared Azure credentials.
#
if ($persona -eq "operator") {
    & {
        Write-Log OperationStarted `
            "Setting up CCF provider..."
        #
        # Use AZCLI_ overrides till latest images are available in mcr.microsoft.com.
        #
        $envVars = @{
            "CREDENTIAL_PROXY_ENDPOINT"                                        = $credentialProxyEndpoint
            "AZCLI_CCF_PROVIDER_CLIENT_IMAGE"                                  = "$repo/ccf/ccf-provider-client:$tag"
            "AZCLI_CCF_PROVIDER_PROXY_IMAGE"                                   = "$repo/ccr-proxy:$tag"
            "AZCLI_CCF_PROVIDER_ATTESTATION_IMAGE"                             = "$repo/ccr-attestation:$tag"
            "AZCLI_CCF_PROVIDER_SKR_IMAGE"                                     = "$repo/skr:$tag"
            "AZCLI_CCF_PROVIDER_RUN_JS_APP_VIRTUAL_IMAGE"                      = "$repo/ccf/app/run-js/virtual:$tag"
            "AZCLI_CCF_PROVIDER_RUN_JS_APP_SNP_IMAGE"                          = "$repo/ccf/app/run-js/snp:$tag"
            "AZCLI_CCF_PROVIDER_RECOVERY_AGENT_IMAGE"                          = "$repo/ccf/ccf-recovery-agent:$tag"
            "AZCLI_CCF_PROVIDER_RECOVERY_SERVICE_IMAGE"                        = "$repo/ccf/ccf-recovery-service:$tag"
            "AZCLI_CCF_PROVIDER_CONTAINER_REGISTRY_URL"                        = "$repo"
            "AZCLI_CCF_PROVIDER_NETWORK_SECURITY_POLICY_DOCUMENT_URL"          = "$repo/policies/ccf/ccf-network-security-policy:$tag"
            "AZCLI_CCF_PROVIDER_RECOVERY_SERVICE_SECURITY_POLICY_DOCUMENT_URL" = "$repo/policies/ccf/ccf-recovery-service-security-policy:$tag"
        }
        $proc = Start-Process docker -ArgumentList "compose -p $ccfProviderName -f $dockerFileDir/ccf/docker-compose.yaml up -d --remove-orphans" -Environment $envVars -Wait -PassThru
        if (0 -ne $proc.ExitCode) {
            throw "Command failed."
        }

        $providerPort = (docker compose -p $ccfProviderName port "client" 8080).Split(':')[1]
        Write-Log OperationCompleted `
            "CCF provider deployed at 'http://localhost:$providerPort'."
    }

    & {
        Write-Log OperationStarted `
            "Setting up cleanroom cluster provider..."
        #
        # Use AZCLI_ overrides till latest images are available in mcr.microsoft.com.
        #
        $semanticVersion = Get-SemanticVersionFromTag $tag
        $envVars = @{
            "CREDENTIAL_PROXY_ENDPOINT"                                                           = $credentialProxyEndpoint
            "AZCLI_CLEANROOM_CLUSTER_PROVIDER_CLIENT_IMAGE"                                       = "$repo/cleanroom-cluster/cleanroom-cluster-provider-client:$tag"
            "AZCLI_CLEANROOM_CLUSTER_PROVIDER_PROXY_IMAGE"                                        = "$repo/ccr-proxy:$tag"
            "AZCLI_CLEANROOM_CLUSTER_PROVIDER_ATTESTATION_IMAGE"                                  = "$repo/ccr-attestation:$tag"
            "AZCLI_CLEANROOM_CLUSTER_PROVIDER_GOVERNANCE_IMAGE"                                   = "$repo/ccr-governance:$tag"
            "AZCLI_CLEANROOM_CLUSTER_PROVIDER_SKR_IMAGE"                                          = "$repo/skr:$tag"
            "AZCLI_CLEANROOM_CLUSTER_PROVIDER_CONTAINER_REGISTRY_URL"                             = "$repo"
            "AZCLI_CLEANROOM_SIDECARS_POLICY_DOCUMENT_REGISTRY_URL"                               = "$repo"
            "AZCLI_CLEANROOM_CLUSTER_PROVIDER_SPARK_ANALYTICS_AGENT_IMAGE"                        = "$repo/workloads/cleanroom-spark-analytics-agent:$tag"
            "AZCLI_CLEANROOM_CLUSTER_PROVIDER_SPARK_ANALYTICS_AGENT_SECURITY_POLICY_DOCUMENT_URL" = "$repo/policies/workloads/cleanroom-spark-analytics-agent-security-policy:$tag"
            "AZCLI_CLEANROOM_CLUSTER_PROVIDER_SPARK_ANALYTICS_AGENT_CHART_URL"                    = "$repo/workloads/helm/cleanroom-spark-analytics-agent:$semanticVersion"
            "AZCLI_CLEANROOM_CLUSTER_PROVIDER_SPARK_FRONTEND_IMAGE"                               = "$repo/workloads/cleanroom-spark-frontend:$tag"
            "AZCLI_CLEANROOM_CLUSTER_PROVIDER_SPARK_FRONTEND_SECURITY_POLICY_DOCUMENT_URL"        = "$repo/policies/workloads/cleanroom-spark-frontend-security-policy:$tag"
            "AZCLI_CLEANROOM_CLUSTER_PROVIDER_SPARK_FRONTEND_CHART_URL"                           = "$repo/workloads/helm/cleanroom-spark-frontend:$semanticVersion"
            "AZCLI_CLEANROOM_SIDECARS_VERSIONS_DOCUMENT_URL"                                      = "$repo/sidecar-digests:$tag"
            "AZCLI_CLEANROOM_ANALYTICS_APP_IMAGE_URL"                                             = "$repo/workloads/cleanroom-spark-analytics-app:$tag"
            "AZCLI_CLEANROOM_ANALYTICS_APP_IMAGE_POLICY_DOCUMENT_URL"                             = "$repo/policies/workloads/cleanroom-spark-analytics-app-security-policy:$tag"
        }
        $proc = Start-Process docker -ArgumentList "compose -p $cleanroomClusterProviderName -f $dockerFileDir/cleanroom-cluster/docker-compose.yaml up -d --remove-orphans" -Environment $envVars -Wait -PassThru
        if (0 -ne $proc.ExitCode) {
            throw "Command failed."
        }

        $providerPort = (docker compose -p $cleanroomClusterProviderName port "client" 8080).Split(':')[1]
        Write-Log OperationCompleted `
            "Cleanroom cluster provider deployed at 'http://localhost:$providerPort'."
    }
}

#
# Launch sample environment.
#
& {
    $containerName = $shellContainerName
    $container = (docker container ls -a --filter "name=^$containerName$" --format 'json') | ConvertFrom-Json
    if ($null -eq $container) {
        $createContainer = $true
    }
    else {
        Write-Log Warning `
            "Samples environment for '$persona' already exists - '$($container.Names)' (ID: $($container.ID))."
        $overwrite = $overwrite -or
        (Get-Confirmation -Message "Overwrite container '$containerName'?" -YesLabel "Y" -NoLabel "N")
        if ($overwrite) {
            Write-Log Warning `
                "Deleting container '$containerName'..."
            docker container rm -f $containerName
            $createContainer = $true
        }
        else {
            $createContainer = $false
        }
    }

    if ($createContainer) {
        Write-Log OperationStarted `
            "Creating samples environment '$containerName' using image '$imageName'..."

        # TODO: Cut across to a pre-built docker image?
        $dockerArgs = "image build -t $imageName -f $dockerFileDir/Dockerfile.azure-cleanroom-samples `"./big-data`""
        $customCliExtensions = @(Get-Item -Path "./docker/*.whl")
        if (0 -ne $customCliExtensions.Count) {
            Write-Log Warning `
                "Using custom az cli extensions: $customCliExtensions..."
            $dockerArgs += " --build-arg EXTENSION_SOURCE=local"
        }
        else {
            $fileName = "cleanroom-1.0.42-py2.py3-none-any"
            Write-Log Warning `
                "Using custom az cli extension from: $repo/cli/cleanroom-whl:$tag and filename:$fileName ..."
            $dockerArgs += " --build-arg EXTENSION_REGISTRY=$repo/cli"
            $dockerArgs += " --build-arg EXTENSION_TAG=$tag"
            $dockerArgs += " --build-arg EXTENSION_FILENAME=$fileName"
        }
        $proc = Start-Process docker $dockerArgs -Wait -PassThru
        if (0 -ne $proc.ExitCode) {
            throw "Command failed."
        }

        if ($resourceGroup -eq "") {
            $resourceGroup = Get-Content $personaBase/$($privateDir)/resourceGroup.name -ErrorAction SilentlyContinue
            if ($null -eq $resourceGroup) {
                $resourceGroup = "$persona-$((New-Guid).ToString().Substring(0, 8))"
            }
            else {
                Write-Log Verbose `
                    "Re-using resource group '$resourceGroup' for '$demo'..."
            }
        }

        docker container create `
            --env PERSONA=$persona `
            --env DEMO=$demo `
            --env RESOURCE_GROUP=$resourceGroup `
            --env RESOURCE_GROUP_LOCATION=$resourceGroupLocation `
            --env IDENTITY_ENDPOINT=$credentialProxyEndpoint `
            --env IDENTITY_HEADER="dummy_required_value" `
            --env HOST_PERSONA_PRIVATE_DIR=$personaBase/$($privateDir) `
            --env CLEANROOM_REPO=$repo `
            --env CLEANROOM_TAG=$tag `
            --env USER=$env:USER `
            -v "//var/run/docker.sock:/var/run/docker.sock" `
            -v "$($sharedBase):$virtualBase" `
            -v "$personaBase/$($privateDir):$virtualBase/$privateDir" `
            -v "$personaBase/$($secretDir):$virtualBase/$secretDir" `
            -v "./big-data/scripts:/home/samples/scripts" `
            -v "./big-data/demos:/home/samples/demos" `
            --network host `
            --add-host host.docker.internal:host-gateway `
            --name $containerName `
            -it $imageName
        Write-Log OperationCompleted `
            "Created container '$containerName' to start samples environment for" `
            "'$persona'. Environment will be using resource group '$resourceGroup'."
    }

    # Stop any "orphan" instances of the container that are already running.
    $container = (docker container ps --filter "name=^$containerName$" --format 'json') | ConvertFrom-Json
    if ($null -ne $container) {
        Write-Log Warning `
            "Stopping container '$containerName'..."
        docker container stop --signal SIGKILL $containerName
    }

    Write-Log OperationStarted `
        "Starting samples environment using container '$containerName'..."
    docker container start -a -i $containerName

    Write-Log Warning `
        "Samples environment exited!"
}
