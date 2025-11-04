param(
    [string]$persona = "$env:PERSONA",
    [string]$resourceGroup = "$env:RESOURCE_GROUP",
    [string]$resourceGroupLocation = "$env:RESOURCE_GROUP_LOCATION",

    [string]$samplesRoot = "/home/samples",
    [string]$privateDir = "$samplesRoot/demo-resources/private",
    [string]$publicDir = "$samplesRoot/demo-resources/public",

    [string]$cgsClient = "azure-cleanroom-samples-governance-client-$persona",
    [string]$clusterProviderClient = "azure-cleanroom-samples-cluster-provider",
    [string]$ccfEndpoint = "$publicDir/ccfEndpoint.json",
    [string]$contractId = "",

    [string]$environmentConfig = "$privateDir/$resourceGroup.generated.json",
    [string]$oidcContainerName = "az-cleanroom-samples-$resourceGroup-oidc",

    [string]$repo = "$env:CLEANROOM_REPO",
    [string]$tag = "$env:CLEANROOM_TAG"
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../common/common.psm1

Test-AzureAccessToken

$clusterName = $persona + "-cluster"

#
# Create a cleanroom cluster instance.
#
$subscriptionId = az account show --query id --output tsv
$tenantId = az account show --query tenantId --output tsv
@"
{
    "location": "$resourceGroupLocation",
    "subscriptionId": "$subscriptionId",
    "resourceGroupName": "$resourceGroup",
    "tenantId": "$tenantId"
}
"@  | Out-File $privateDir/providerConfig.json

# Register the ACI and AKS RP so that confidential container/AKS usage is enabled in the subscription.
$aciRpName = "Microsoft.ContainerInstance"
$aciRpRegistration = (az provider show -n $aciRpName --query registrationState --output tsv)
if ($aciRpRegistration -ne "Registered") {
    Write-Log Verbose `
        "$aciRpName provider is not registered on the subscription. Registering provider (this can take a while)..."
    az provider register --namespace $aciRpName --wait
    Write-Log OperationCompleted `
        "$aciRpName provider is registered."
}

$aksRpName = "Microsoft.ContainerService"
$aksRpRegistration = (az provider show -n $aksRpName --query registrationState --output tsv)
if ($aksRpRegistration -ne "Registered") {
    Write-Log Verbose `
        "$aksRpName provider is not registered on the subscription. Registering provider (this can take a while)..."
    az provider register --namespace $aksRpName --wait
    Write-Log OperationCompleted `
        "$aksRpName provider is registered."
}

$cluster = & {
    # Disable $PSNativeCommandUseErrorActionPreference for this scriptblock
    $PSNativeCommandUseErrorActionPreference = $false
    return (az cleanroom cluster show `
            --name $clusterName `
            --provider-config $privateDir/providerConfig.json `
            --provider-client $clusterProviderClient | ConvertFrom-Json)
}

if ($null -eq $cluster) {
    Write-Log OperationStarted `
        "Creating cleanroom cluster '$clusterName' in resource group '$resourceGroup'..."

    az cleanroom cluster create `
        --name $clusterName `
        --provider-config $privateDir/providerConfig.json `
        --provider-client $clusterProviderClient
    Write-Log OperationCompleted `
        "Created cleanroom cluster '$clusterName'."
}
else {
    Write-Log Warning `
        "Connected to existing cleanroom cluster '$clusterName'."
}

$response = az cleanroom cluster show `
    --name $clusterName `
    --provider-config $privateDir/providerConfig.json `
    --provider-client $clusterProviderClient
$response | Out-File $privateDir/cl-cluster.json

#
# Note: For samples/demo purposes the kubeconfig is written out to the public directory. In production this will not be public information.
#
$kubeConfig = "${publicDir}/k8s-credentials.yaml"
az cleanroom cluster get-kubeconfig `
    --name $clusterName `
    --provider-config $privateDir/providerConfig.json `
    -f $kubeConfig `
    --provider-client $clusterProviderClient

Write-Log OperationCompleted `
    "Cleanroom cluster configured."

# Propose the analytics contract for the cleanroom cluster.
$ccfEndpointUrl = (Get-Content $ccfEndpoint | ConvertFrom-Json).url
$agent = Get-Content $publicDir/ccf.recovery-agent.json | ConvertFrom-Json
$agentEndpoint = $agent.endpoint
$agentNetworkReport = curl -k -s -S $agentEndpoint/network/report | ConvertFrom-Json
$reportDataContent = $agentNetworkReport.reportDataPayload | base64 -d | ConvertFrom-Json

# Propose a contract for the cleanroom cluster analytics deployment.
if ($contractId -eq "") {
    $contractId = Get-Content $publicDir/analytics.contract-id -ErrorAction SilentlyContinue
    if ($null -eq $contractId) {
        $contractId = "analytics-$((New-Guid).ToString().Substring(0, 8))"
    }
}

$contract = & {
    # Disable $PSNativeCommandUseErrorActionPreference for this scriptblock
    $PSNativeCommandUseErrorActionPreference = $false
    return (az cleanroom governance contract show `
            --id $contractId `
            --governance-client $cgsClient | ConvertFrom-Json)
}

$recoveryMembers = az cleanroom governance member show --governance-client $cgsClient | jq '[.value[] | select(.publicEncryptionKey != null) | .memberId]' -c 
@"
{
  "ccrgovEndpoint": "$ccfEndpointUrl",
  "ccrgovApiPathPrefix": "/app/contracts/$contractId",
  "ccrgovServiceCertDiscovery" : {
    "endpoint": "$agentEndpoint/network/report",
    "snpHostData": "$($agent.snpHostData)",
    "constitutionDigest": "$($reportDataContent.constitutionDigest)",
    "jsappBundleDigest": "$($reportDataContent.jsappBundleDigest)"
  },
  "ccfNetworkRecoveryMembers": $recoveryMembers
}
"@ > $privateDir/contract.json

if ($null -eq $contract) {
    $data = Get-Content -Raw $privateDir/contract.json
    Write-Output "Creating contract '$contractId'..."
    az cleanroom governance contract create `
        --data "$data" `
        --id $contractId `
        --governance-client $cgsClient
    $contract = (az cleanroom governance contract show `
            --id $contractId `
            --governance-client $cgsClient | ConvertFrom-Json)
}
else {
    Write-Output "Contract '$contractId' already exists."
    $contract | ConvertTo-Json -Depth 100 | jq
    $expected = Get-Content $privateDir/contract.json | sha256sum | cut -d ' ' -f 1
    $actual = $contract.data.TrimEnd("`n") | sha256sum | cut -d ' ' -f 1
    if ($expected -ne $actual) {
        throw "Contract data does not match expected data. Expected hash value: $expected, Actual hash value: $actual"
    }
}

$contractId | Out-File $publicDir/analytics.contract-id

if ($contract.state -eq "Draft") {
    # Submitting a contract proposal.
    $version = (az cleanroom governance contract show `
            --id $contractId `
            --query "version" `
            --output tsv `
            --governance-client $cgsClient)

    az cleanroom governance contract propose `
        --version $version `
        --id $contractId `
        --governance-client $cgsClient
    $contract = (az cleanroom governance contract show `
            --id $contractId `
            --governance-client $cgsClient | ConvertFrom-Json)
}

if ($contract.state -eq "Proposed") {
    # Accept it.
    az cleanroom governance contract vote `
        --id $contractId `
        --proposal-id $contract.proposalId `
        --action accept `
        --governance-client $cgsClient
    $contract = (az cleanroom governance contract show `
            --id $contractId `
            --governance-client $cgsClient | ConvertFrom-Json)
}

if ($contract.state -ne "Accepted") {
    $contract | ConvertTo-Json -Depth 100 | jq
    throw "Contract should have been in accepted state."
}

Write-Output "Enabling CA..."
az cleanroom governance ca propose-enable `
    --contract-id $contractId `
    --governance-client $cgsClient

# Vote on the proposed CA enable.
$proposalId = az cleanroom governance ca show `
    --contract-id $contractId `
    --governance-client $cgsClient `
    --query "proposalIds[0]" `
    --output tsv

az cleanroom governance proposal vote `
    --proposal-id $proposalId `
    --action accept `
    --governance-client $cgsClient

az cleanroom governance ca generate-key `
    --contract-id $contractId `
    --governance-client $cgsClient

az cleanroom governance ca show `
    --contract-id $contractId `
    --governance-client $cgsClient `
    --query "caCert" `
    --output tsv > $publicDir/cleanroomca.crt

Write-Output "Setting up OIDC issuer url endpoint..."

#
# Setup OIDC issuer endpoint.
#
$environmentConfigResult = Get-Content $environmentConfig | ConvertFrom-Json
$issuerInfo = (az cleanroom governance oidc-issuer show `
        --governance-client $cgsClient | ConvertFrom-Json)
if ($null -ne $issuerInfo.issuerUrl) {
    $issuerUrl = $issuerInfo.issuerUrl
    Write-Log Warning `
        "OIDC issuer already set for tenant to '$issuerUrl'. Skipping!"
}
else {
    $oidcsa = $environmentConfigResult.oidcsa.name
    if ($tenantId -eq "72f988bf-86f1-41af-91ab-2d7cd011db47") {
        Write-Log Verbose `
            "Use pre-provisioned storage account $oidcsa for OIDC setup"
    }
    else {

        Write-Log Verbose `
            "Setting up OIDC issuer using storage account '$oidcsa'..."

        Write-Log Verbose `
            "Setting up static website on storage account to setup oidc documents endpoint"
        az storage blob service-properties update `
            --account-name $oidcsa `
            --static-website `
            --404-document error.html `
            --index-document index.html `
            --auth-mode login
    }

    $objectId = GetLoggedInEntityObjectId
    $role = "Storage Blob Data Contributor"
    $roleAssignment = (az role assignment list `
            --assignee-object-id $objectId `
            --scope $environmentConfigResult.oidcsa.id `
            --role $role `
            --fill-principal-name false `
            --fill-role-definition-name false) | ConvertFrom-Json

    if ($roleAssignment.Length -eq 1) {
        Write-Host "$role permission on the storage account already exists, skipping assignment"
    }
    else {
        Write-Host "Assigning $role on the storage account"
        az role assignment create `
            --role $role `
            --scope $environmentConfigResult.oidcsa.id `
            --assignee-object-id $objectId `
            --assignee-principal-type $(Get-Assignee-Principal-Type)
    }

    $webUrl = (az storage account show `
            --name $oidcsa `
            --query "primaryEndpoints.web" `
            --output tsv)
    Write-Host "Storage account static website URL: $webUrl"

    Write-Log Verbose `
        "Uploading openid-configuration to container '$oidcContainerName' in '$oidcsa'..." `
        "$($PSStyle.Reset)"
    @"
{
    "issuer": "$webUrl${oidcContainerName}",
    "jwks_uri": "$webUrl${oidcContainerName}/openid/v1/jwks",
    "response_types_supported": [
        "id_token"
    ],
    "subject_types_supported": [
        "public"
    ],
    "id_token_signing_alg_values_supported": [
        "RS256"
    ]
}
"@ | Out-File $privateDir/openid-configuration.json
    az storage blob upload `
        --container-name '$web' `
        --file $privateDir/openid-configuration.json `
        --name ${oidcContainerName}/.well-known/openid-configuration `
        --account-name $oidcsa `
        --overwrite `
        --auth-mode login

    Write-Log Verbose `
        "Uploading jwks to container '$oidcContainerName' in '$oidcsa'..."
    $url = "$ccfEndpointUrl/app/oidc/keys"
    curl -sL -k $url | jq | Out-File $privateDir/jwks.json
    az storage blob upload `
        --container-name '$web' `
        --file $privateDir/jwks.json `
        --name ${oidcContainerName}/openid/v1/jwks `
        --account-name $oidcsa `
        --overwrite `
        --auth-mode login

    $issuerUrl = "$webUrl${oidcContainerName}"
    Write-Log OperationCompleted `
        "Set OIDC issuer to '$issuerUrl'."
}

$issuerUrl | Out-File $publicDir/issuer.url

$option = "cached-debug"
Write-Output "Generating deployment template/policy with $option creation option for analytics workload..."
mkdir -p $privateDir/deployments
az cleanroom cluster analytics-workload deployment generate `
    --contract-id $contractId `
    --governance-client $cgsClient `
    --output-dir $privateDir/deployments `
    --security-policy-creation-option $option `
    --provider-client $clusterProviderClient `
    --provider-config $privateDir/providerConfig.json

Write-Output "Setting deployment template..."
az cleanroom governance deployment template propose `
    --contract-id $contractId `
    --template-file $privateDir/deployments/analytics-workload.deployment-template.json `
    --governance-client $cgsClient

# Vote on the proposed deployment template.
$proposalId = az cleanroom governance deployment template show `
    --contract-id $contractId `
    --governance-client $cgsClient `
    --query "proposalIds[0]" `
    --output tsv

az cleanroom governance proposal vote `
    --proposal-id $proposalId `
    --action accept `
    --governance-client $cgsClient

Write-Output "Setting clean room policy..."
az cleanroom governance deployment policy propose `
    --policy-file $privateDir/deployments/analytics-workload.governance-policy.json `
    --contract-id $contractId `
    --governance-client $cgsClient

# Vote on the proposed cce policy.
$proposalId = az cleanroom governance deployment policy show `
    --contract-id $contractId `
    --governance-client $cgsClient `
    --query "proposalIds[0]" `
    --output tsv

az cleanroom governance proposal vote `
    --proposal-id $proposalId `
    --action accept `
    --governance-client $cgsClient

# Deploy the analytics agent using the CGS /deploymentspec endpoint as the analytics config endpoint.
$ccfName = $persona + "-ccf"
$serviceCertFileName = "${ccfName}_service_cert.pem"
$serviceCert = "$publicDir/$serviceCertFileName"
@"
{
    "url": "${ccfEndpointUrl}/app/contracts/$contractId/deploymentspec",
    "caCert": "$((Get-Content $serviceCert -Raw).ReplaceLineEndings("\n"))"
}
"@ > $privateDir/analytics-workload-config-endpoint.json

pwsh $PSScriptRoot/enable-analytics-workload.ps1 `
    -privateDir $privateDir `
    -publicDir $publicDir `
    -securityPolicyCreationOption $option `
    -configEndpointFile $privateDir/analytics-workload-config-endpoint.json

Write-Output "Fetching deployment information..."
# Get the analytics endpoint from the deployed cluster.
$clCluster = Get-Content $privateDir/cl-cluster.json | ConvertFrom-Json
$analyticsEndpoint = $clCluster.analyticsWorkloadProfile.endpoint

#
# Instead of accessing the service via ${analyticsEndpoint}, we will use kubectl proxy to access it via localhost.
# This is needed as the public IP address for AKS load balancer is not accessible from machines that are not on corpnet.
# https://kubernetes.io/docs/tasks/access-application-cluster/access-cluster-services/#manually-constructing-apiserver-proxy-urls
# For Kind cluster infra also this technique works fine to access the service as it would be having a clusterIP
# and thus not reachable from outside the cluster.
#
$analyticsEndpoint = "http://localhost:8181/api/v1/namespaces/cleanroom-spark-analytics-agent/services/https:cleanroom-spark-analytics-agent:443/proxy"

Write-Output "Using analytics endpoint: $analyticsEndpoint"
$deploymentInformation = @{
    url = $analyticsEndpoint
} | ConvertTo-Json
az cleanroom governance deployment information propose `
    --deployment-information $deploymentInformation `
    --contract-id $contractId `
    --governance-client $cgsClient

# Vote on the proposed deployment information.
$proposalId = az cleanroom governance deployment information show `
    --contract-id $contractId `
    --governance-client $cgsClient `
    --query "proposalIds[0]" `
    --output tsv

az cleanroom governance proposal vote `
    --proposal-id $proposalId `
    --action accept `
    --governance-client $cgsClient