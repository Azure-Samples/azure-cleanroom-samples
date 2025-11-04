param(
    [string]$demo = "$env:DEMO",

    [string]$persona = "$env:PERSONA",
    [string]$resourceGroup = "$env:RESOURCE_GROUP",

    [string]$samplesRoot = "/home/samples",
    [string]$publicDir = "$samplesRoot/demo-resources/public",
    [string]$privateDir = "$samplesRoot/demo-resources/private",
    [string]$secretDir = "$samplesRoot/demo-resources/secret",
    [string]$demosRoot = "$samplesRoot/demos",
    [string]$sa = "",
    [switch]$skipUpload,
    [string]$awsProfileName = "default",
    [string]$cgsClient = "azure-cleanroom-samples-governance-client-$persona",

    [string]$environmentConfig = "$privateDir/$resourceGroup.generated.json",
    [string]$secretstoreConfig = "$privateDir/secretstores.config",
    [string]$datastoreConfig = "$privateDir/datastores.config",
    [string]$datastoreDir = "$privateDir/datastores",
    [string]$demoPath = "$demosRoot/$demo",
    [string]$datasourcePath = "$demoPath/datasource/$persona",
    [string]$datasinkPath = "$demoPath/datasink/$persona"
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../common/common.psm1

if ($persona -eq "woodgrove" -and $demo -eq "analytics-s3-sse") {
    $awsCreds = Get-AWSCredential -ProfileName $awsProfileName
    if ($null -eq $awsCreds) {
        throw "AWS credentials not found for profile '$awsProfileName'. Ensure the profile is configured correctly."
    }
    $awsAccessKeyId = $awsCreds.GetCredentials().AccessKey
    $awsSecretAccessKey = $awsCreds.GetCredentials().SecretKey

    Write-Log OperationStarted `
        "Saving AWS credentials as a secret in CGS..."
    $secretConfig = @{
        awsAccessKeyId     = $awsAccessKeyId
        awsSecretAccessKey = $awsSecretAccessKey
    } | ConvertTo-Json | base64 -w 0

    $awsConfigCgsSecretName = "consumer-aws-config"
    $contractId = Get-Content $publicDir/analytics.contract-id
    $awsConfigCgsSecretId = (az cleanroom governance contract secret set `
            --secret-name $awsConfigCgsSecretName `
            --value $secretConfig `
            --contract-id $contractId `
            --governance-client $cgsClient `
            --query "secretId" `
            --output tsv)
}
else {
    Test-AzureAccessToken
}

if (Test-Path -Path $datasourcePath) {
    $dirs = Get-ChildItem -Path $datasourcePath -Directory -Name
    foreach ($dir in $dirs) {
        $datastoreName = "$demo-$persona-$dir".ToLower()
        Write-Log Verbose `
            "Enumerated datasource '$datastoreName' in '$datasourcePath'..."

        if ($persona -eq "woodgrove" -and $demo -eq "analytics-s3-sse") {
            $user = $env:USER
            $bucketName = "$datastoreName-${user}" -replace "_", "-"
            $region = "us-west-1"
            if (-not (Get-S3Bucket | Where-Object { $_.BucketName -eq $bucketName })) {
                Write-Output "Creating $bucketName..."
                New-S3Bucket -BucketName $bucketName -Region $region
                Write-Output "Bucket created."
            }
            else {
                Write-Output "Bucket $bucketName already exists."
            }
            
            # Create a datastore entry for the AWS S3 storage to with CGS secret Id as 
            # its configuration.
            $awsUrl = "https://s3.amazonaws.com"
            az cleanroom datastore add `
                --name $datastoreName `
                --config $datastoreConfig `
                --backingstore-type Aws_S3 `
                --backingstore-id $awsUrl `
                --aws-config-cgs-secret-id $awsConfigCgsSecretId `
                --container-name $bucketName `
                --schema-format "csv" `
                --schema-fields "date:date,time:string,author:string,mentions:string"
            $datastorePath = "$datastoreDir/$datastoreName"
            mkdir -p $datastorePath
            Write-Log OperationCompleted `
                "Created data store '$datastoreName' backed by S3 bucket '$bucketName'."

            if (!$skipUpload) {
                cp -r $datasourcePath/$dir/* $datastorePath
                Write-S3Object -BucketName $bucketName -Folder $datastorePath -Recurse  -KeyPrefix "/" -Region $region
            }
        }
        else {
            if ($sa -eq "") {
                $initResult = Get-Content $environmentConfig | ConvertFrom-Json
                $sa = $initResult.datasa.id
            }

            if ($demo -eq "analytics-cpk") {
                az cleanroom datastore add `
                    --name $datastoreName `
                    --config $datastoreConfig `
                    --secretstore-config $secretStoreConfig `
                    --secretstore $persona-local-store `
                    --encryption-mode CPK `
                    --backingstore-type Azure_BlobStorage `
                    --backingstore-id $sa `
                    --schema-format "csv" `
                    --schema-fields "date:date,time:string,author:string,mentions:string"
            }
            elseif ($demo -eq "analytics-sse" -or $demo -eq "analytics-s3-sse") {
                az cleanroom datastore add `
                    --name $datastoreName `
                    --config $datastoreConfig `
                    --encryption-mode SSE `
                    --backingstore-type Azure_BlobStorage `
                    --backingstore-id $sa `
                    --schema-format "csv" `
                    --schema-fields "date:date,time:string,author:string,mentions:string"
            }
            else {
                throw "Demo $demo not handled in publish-data. Fix this."
            }

            $datastorePath = "$datastoreDir/$datastoreName"
            mkdir -p $datastorePath
            Write-Log OperationCompleted `
                "Created data store '$datastoreName' backed by '$sa'."

            if (!$skipUpload) {
                cp -r $datasourcePath/$dir/* $datastorePath
                az cleanroom datastore upload `
                    --name $datastoreName `
                    --config $datastoreConfig `
                    --src $datastorePath
            }
        }
        Write-Log OperationCompleted `
            "Published data from '$datasourcePath/$dir' as data store '$datastoreName'."
    }
}
else {
    Write-Log Warning `
        "No datasource available for persona '$persona' in demo '$demo'."
}

if (Test-Path -Path $datasinkPath) {
    $dirs = Get-ChildItem -Path $datasinkPath -Directory -Name
    foreach ($dir in $dirs) {
        $datastoreName = "$demo-$persona-$dir".ToLower()
        Write-Log Verbose `
            "Enumerated datasink '$datastoreName' in '$datasinkPath'..."

        if ($persona -eq "woodgrove" -and $demo -eq "analytics-s3-sse") {
            $user = $env:CODESPACES -eq "true" ? $env:GITHUB_USER : $env:USER
            $bucketName = "$datastoreName-${user}" -replace "_", "-"
            $region = "us-west-1"
            if (-not (Get-S3Bucket | Where-Object { $_.BucketName -eq $bucketName })) {
                New-S3Bucket -BucketName $bucketName -Region $region
                Write-Output "Bucket $bucketName created."
            }
            else {
                Write-Output "Bucket $bucketName already exists."
            }

            # Create a datastore entry for the AWS S3 storage to with CGS secret Id as 
            # its configuration.
            $awsUrl = "https://s3.amazonaws.com"
            az cleanroom datastore add `
                --name $datastoreName `
                --config $datastoreConfig `
                --backingstore-type Aws_S3 `
                --backingstore-id $awsUrl `
                --aws-config-cgs-secret-id $awsConfigCgsSecretId `
                --container-name $bucketName `
                --schema-format "csv" `
                --schema-fields "author:string,Number_Of_Mentions:long,Restricted_Sum:number"
            Write-Log OperationCompleted `
                "Created data store '$datastoreName' backed by S3 bucket '$bucketName'."
        }
        else {
            if ($demo -eq "analytics-cpk") {
                az cleanroom datastore add `
                    --name $datastoreName `
                    --config $datastoreConfig `
                    --secretstore-config $secretStoreConfig `
                    --secretstore $persona-local-store `
                    --encryption-mode CPK `
                    --backingstore-type Azure_BlobStorage `
                    --backingstore-id $sa `
                    --schema-format "csv" `
                    --schema-fields "author:string,Number_Of_Mentions:long,Restricted_Sum:number"
            }
            elseif ($demo -eq "analytics-sse" -or $demo -eq "analytics-s3-sse") {
                az cleanroom datastore add `
                    --name $datastoreName `
                    --config $datastoreConfig `
                    --encryption-mode SSE `
                    --backingstore-type Azure_BlobStorage `
                    --backingstore-id $sa `
                    --schema-format "csv" `
                    --schema-fields "author:string,Number_Of_Mentions:long,Restricted_Sum:number"
            }
            else {
                throw "Demo $demo not handled in publish-data. Fix this."
            }

            $datastorePath = "$datastoreDir/$datastoreName"
            mkdir -p $datastorePath
            Write-Log OperationCompleted `
                "Created data store '$datastoreName' backed by '$sa'."
        }
    }
}
else {
    Write-Log Warning `
        "No datasink available for persona '$persona' in demo '$demo'."
}

#
# Create datasource/datasink entries in the configuration file.
#
pwsh $PSScriptRoot/../specification/initialize-specification.ps1 -demo $demo

# And add the datasources/datasinks to the specification and publish
pwsh $PSScriptRoot/../specification/add-specification-data.ps1 -demo $demo