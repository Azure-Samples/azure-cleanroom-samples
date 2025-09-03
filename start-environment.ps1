param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("litware", "fabrikam", "contosso", "client", "operator", IgnoreCase = $false)]
    [string]$persona,

    [string]$resourceGroup = "",
    [string]$resourceGroupLocation = "westus",

    [string]$imageName = "azure-cleanroom-samples",
    [string]$dockerFileDir = "./docker",

    [string]$accessTokenProviderName = "$imageName-credential-proxy",
    [string]$ccfProviderName = "$imageName-ccf-provider",
    [string]$telemetryDashboardName = "$imageName-telemetry",
    [string]$shellContainerName = "$imageName-shell-$persona",

    [switch]$overwrite,
    [switch]$shareCredentials
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

#
# Launch telemetry dashboard for 'litware'.
#
if ($persona -eq "litware") {
    & {
        Write-Log OperationStarted `
            "Setting up telemetry dashboard..."
        docker build -f $dockerFileDir/Dockerfile.azure-cleanroom-samples-otelcollector -t $imageName-otelcollector $dockerFileDir

        $env:TELEMETRY_FOLDER = $telemetryDir
        $dashboardName = "$imageName-telemetry"
        docker compose -p $dashboardName -f $dockerFileDir/telemetry/docker-compose.yaml up -d --remove-orphans

        $dashboardPort = (docker compose -p $dashboardName port "aspire" 18888).Split(':')[1]
        Write-Log OperationCompleted `
            "Aspire dashboard deployed at 'http://localhost:$dashboardPort'."
    }
}

#
# Launch CCF provider for 'operator' using shared Azure credentials.
#
if ($persona -eq "operator") {
    & {
        Write-Log OperationStarted `
            "Setting up CCF provider..."

        $env:CREDENTIAL_PROXY_ENDPOINT = $credentialProxyEndpoint
        docker compose -p $ccfProviderName -f $dockerFileDir/ccf/docker-compose.yaml up -d --remove-orphans

        $providerPort = (docker compose -p $ccfProviderName port "client" 8080).Split(':')[1]
        Write-Log OperationCompleted `
            "CCF provider deployed at 'http://localhost:$providerPort'."
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
        $dockerArgs = "image build -t $imageName -f $dockerFileDir/Dockerfile.azure-cleanroom-samples `".`""
        $customCliExtensions = @(Get-Item -Path "./docker/*.whl")
        if (0 -ne $customCliExtensions.Count) {
            Write-Log Warning `
                "Using custom az cli extensions: $customCliExtensions..."
            $dockerArgs += " --build-arg EXTENSION_SOURCE=local"
        }
        Start-Process docker $dockerArgs -Wait -PassThru
        if (0 -ne $proc.ExitCode) {
            throw "Command failed."
        }

        if ($resourceGroup -eq "") {
            $resourceGroup = "$persona-$((New-Guid).ToString().Substring(0, 8))"
        }

        docker container create `
            --env PERSONA=$persona `
            --env RESOURCE_GROUP=$resourceGroup `
            --env RESOURCE_GROUP_LOCATION=$resourceGroupLocation `
            --env IDENTITY_ENDPOINT=$credentialProxyEndpoint `
            --env IDENTITY_HEADER="dummy_required_value" `
            -v "//var/run/docker.sock:/var/run/docker.sock" `
            -v "$($sharedBase):$virtualBase" `
            -v "$personaBase/$($privateDir):$virtualBase/$privateDir" `
            -v "$personaBase/$($secretDir):$virtualBase/$secretDir" `
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
