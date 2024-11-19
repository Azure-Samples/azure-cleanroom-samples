function Create-SymmetricKeys{
    param(
        [Parameter(Mandatory=$true)]
        [string]$keyName,

        [Parameter(Mandatory=$true)]
        [string]$keyFilePath
    )

    if (Test-Path $keyFilePath)
    {
        Write-Host "Skipping creation of key $keyName. Reusing existing key"
        $keyBytes = [System.IO.File]::ReadAllBytes($keyFilePath)
        $keyHexString = ($keyBytes | ForEach-Object ToString x2) -join ''
        return $keyHexString
    }

    Write-Host "Creating key $keyName"
    $aesKey = [System.Security.Cryptography.Aes]::Create()

    Write-Host "Writing key to $keyFilePath"
    [System.IO.File]::WriteAllBytes($keyFilePath, $aesKey.Key)
    $keyHexString = ($aesKey.Key | ForEach-Object ToString x2) -join ''
    $aesKey.Dispose()
    return $keyHexString
}

function Create-Rsa-Key {
    param (
        [Parameter(Mandatory=$true)]
        [string]$keyFilePathPrefix
    )

    $rsa = [System.Security.Cryptography.RSAOpenSsl]::Create(2048)
    $publicKeyFile = $keyFilePathPrefix + "-public.pem"
    $privateKeyFile = $keyFilePathPrefix + "-private.pem"
    $rsa.ExportRSAPublicKeyPem() | Out-File $publicKeyFile
    $rsa.ExportPkcs8PrivateKeyPem() | Out-File $privateKeyFile

    Write-Host "Exported public key to $publicKeyFile"
    Write-Host "Exported private key to $privateKeyFile"
    return $rsa
}

function Save-KeyBytes-To-KeyVault {
    param (
        [string]$keyVaultName,
        [string]$keyName,
        [string]$keyFilePath
    )

    $keyBytes = [System.IO.File]::ReadAllBytes($keyFilePath)
    $keySecret = ($keyBytes | ForEach-Object ToString x2) -join ''

    Write-Host "Writing out key $keyName to Key Vault $keyVaultName as a secret"
    az keyvault secret set --name $keyName --vault-name $keyVaultName --value $keySecret
}