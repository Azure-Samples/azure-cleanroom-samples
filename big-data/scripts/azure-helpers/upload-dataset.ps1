. $PSScriptRoot/key-helpers.ps1
function Upload-Dataset {
    param(
        [string]$folderPath,

        [string]$secretStoreUrl,

        [string]$kekName,

        [string]$keyStoreUrl,

        [string]$storageAccountName,

        [string]$resourceGroup,

        [string]$maaUrl,

        [string]$keysFolderPath,

        [bool]$isReadOnly = $true,

        [string]$datasetName,

        [string]$wrappedDekSecretName,

        [string]$containerName
    )

    if (-not (Test-Path -Path $folderPath)) {
        $result = New-Item -Path $folderPath -ItemType Directory
    }

    if (-not (Test-Path -Path $keysFolderPath)) {
        $result = New-Item -Path $keysFolderPath -ItemType Directory
    }

    Write-Host "Generating CPK for folder $folderPath"
    $folderName = Split-Path $folderPath -Leaf
    $result = Create-SymmetricKeys -keyName $folderName -keyFilePath "$keysFolderPath/$folderName.bin"

    $keyDetails =@{
        keyName = "$folderName"
        keyFilePath = "$keysFolderPath/$folderName.bin"
        wrappedDekSecretName = $wrappedDekSecretName
    }

    Write-Host "Creating container $containerName in storage account"
    $result = az storage container create --resource-group $resourceGroup --account-name $storageAccountName --name $containerName --auth-mode login

    Write-Host "Listing files in $folderPath"
    $files = Get-ChildItem -Path $folderPath/* -Name -Exclude *.bin
    foreach ($file in $files)
    {
        $details = EncryptAndUploadData -keyFilePath "$keysFolderPath/$folderName.bin" -containerName "$containerName" -filePath "$folderPath/$file" -storageAccountName $storageAccountName
    }

    $dataset = Get-Dataset `
        -datasetName $datasetName `
        -secretStoreUrl $secretStoreUrl `
        -kekName $kekName `
        -dekName $keyDetails.keyName `
        -keyStoreUrl $keyStoreUrl `
        -storageAccountName $storageAccountName `
        -containerName $containerName `
        -maaUrl $maaUrl `
        -isReadOnly $isReadOnly `
        -wrappedDekSecretName $wrappedDekSecretName
    return $dataset, $keyDetails
}

function Prepare-Telemetry-DataSink
{
    param(
        [string]$type,

        [string]$containerName,

        [string]$keysFolderPath,

        [string]$secretStoreUrl,

        [string]$kekName,

        [string]$keyStoreUrl,

        [string]$storageAccountName,

        [string]$resourceGroup,

        [string]$maaUrl,

        [string]$wrappedDekSecretName
    )

    if (-not (Test-Path -Path $keysFolderPath)) {
        $result = New-Item -Path $keysFolderPath -ItemType Directory
    }

    Write-Host "Creating container $containerName in storage account $storageAccountName"
    $result = az storage container create --resource-group $resourceGroup --account-name $storageAccountName --name $containerName --auth-mode login

    $key =@{
        keyName = $type
        keyFilePath = "$keysFolderPath/$type.bin"
        wrappedDekSecretName = $wrappedDekSecretName
    }


    $result = Create-SymmetricKeys -keyName $($key.keyName) -keyFilePath $($key.keyFilePath)

    $datasink = Get-Dataset `
        -datasetName $type-telemetry `
        -secretStoreUrl $secretStoreUrl `
        -kekName $kekName `
        -dekName $key.keyName `
        -keyStoreUrl $keyStoreUrl `
        -storageAccountName $storageAccountName `
        -containerName $containerName `
        -maaUrl $maaUrl `
        -isReadOnly $false `
        -wrappedDekSecretName $wrappedDekSecretName

    return $datasink, $key
}

function Get-Dataset
{
    param(
        [string]$datasetName,

        [string]$secretStoreUrl,

        [string]$kekName,

        [string]$dekName,

        [string]$keyStoreUrl,

        [string]$storageAccountName,

        [string]$containerName,

        [string]$maaUrl,

        [bool]$isReadOnly = $true,

        [string]$wrappedDekSecretName
    )
    $datasetType = $isReadOnly ? "Volume__ReadOnly" : "Volume__ReadWrite"
    $proxyType = $isReadOnly ? "SecureVolume__ReadOnly__AzureStorage__BlobContainer" : "SecureVolume__ReadWrite__AzureStorage__BlobContainer"
    $storageUrl = "https://$storageAccountName.blob.core.windows.net/$containerName"
    $blobDetails = @{
        "__NAME__" = $datasetName
        "__READONLY__" = $datasetType
        "__SA_URL__" = $storageUrl
        "__PROXY_TYPE__" = $proxyType
        "__KEY_NAME__" = $dekName
        "__HSM_URL__" = $($keyStoreUrl.Split("https://").TrimEnd("/")[1])
        "__MAA_URL__" = $($maaUrl.Split("https://").TrimEnd("/")[1])
        "__WRAPPED_DEK_NAME__" = "$wrappedDekSecretName"
        "__AZURE_KEYVAULT_URL__" = $secretStoreUrl
        "__KEK_NAME__" = $kekName
    }

    $scriptDir = { Split-Path -Parent $MyInvocation.ScriptName }
    $content = Replace-Strings -filePath "$(&$scriptDir)/blob.yaml" -replacements $blobDetails
    return ConvertFrom-Yaml $content -Ordered
}


function EncryptAndUploadData
{
    param(
        [Parameter(Mandatory=$true)]
        [string]$keyFilePath,

        [Parameter(Mandatory=$true)]
        [string]$storageAccountName,

        [Parameter(Mandatory=$true)]
        [string]$containerName,

        [Parameter(Mandatory=$true)]
        [string]$filePath
    )

    $keyFilePath = $(Resolve-Path $keyFilePath).Path
    $filePath = $(Resolve-Path $filePath).Path

    $keyBytes = [System.IO.File]::ReadAllBytes($keyFilePath)
    $keyBase64Str = [Convert]::ToBase64String($keyBytes)

    $shaGenerator = [System.Security.Cryptography.SHA256]::Create()
    $shaBytes = $shaGenerator.ComputeHash($keyBytes)
    $shaBase64 = [Convert]::ToBase64String($shaBytes)
    Write-Host "SHA base64 of the key: $shaBase64"

    # azcopy with CPK needs the values below for encryption
    # https://learn.microsoft.com/en-us/azure/storage/common/storage-ref-azcopy-copy
    $env:CPK_ENCRYPTION_KEY = "$keyBase64Str"
    $env:CPK_ENCRYPTION_KEY_SHA256 = "$shaBase64"

    $endDate = (Get-Date -AsUTC).AddHours(1)
    $end = Get-Date -Date $endDate -Format yyyy-MM-ddTHH:mmZ
    $sasUri = az storage container generate-sas --name $containerName --permissions dlrw --expiry $end --account-name $storageAccountName --auth-mode login --as-user

    $fileName = Split-Path $filePath -Leaf
    Write-Host "Uploading file $filePath to storage account $storageAccountName in container $containerName"
    $containerUrl = "https://$storageAccountName.blob.core.windows.net/$containerName"
    $url = "$containerUrl/$($fileName)?$($sasUri.Trim('"'))"
    $upload = azcopy copy $filePath $url --cpk-by-value

    $shaGenerator.Dispose()
    $blobUrl = "$containerUrl/$fileName"

    return @{
        blobUrl = $blobUrl
        key = $containerName
        file = $fileName
        containerUrl = $containerUrl
    }
}



