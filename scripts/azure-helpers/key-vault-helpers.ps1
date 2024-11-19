function Create-Hsm {
    param(
        [string]$resourceGroup,

        [string]$hsmName,

        [string]$adminObjectId,

        [string]$outDir
    )

    Write-Host "Creating MHSM $hsmName in resource group $resourceGroup with $adminObjectId as administrator"
    $mhsmResult = (az keyvault create --resource-group $resourceGroup --hsm-name $hsmName --retention-days 90 --administrators $adminObjectId) | ConvertFrom-Json
    CheckLastExitCode

    if ($mhsmResult.properties.securityDomainProperties.activationStatus -ne "Active") {
        openssl req -newkey rsa:2048 -nodes -keyout $outDir/cert_0.key -x509 -days 365 -out $outDir/cert_0.cer -subj "/C=US/CN=Microsoft"
        openssl req -newkey rsa:2048 -nodes -keyout $outDir/cert_1.key -x509 -days 365 -out $outDir/cert_1.cer -subj "/C=US/CN=Microsoft"
        openssl req -newkey rsa:2048 -nodes -keyout $outDir/cert_2.key -x509 -days 365 -out $outDir/cert_2.cer -subj "/C=US/CN=Microsoft"

        Write-Host "Activating HSM"
        $activationResult = az keyvault security-domain download --hsm-name $hsmName --sd-wrapping-keys $outDir/cert_0.cer $outDir/cert_1.cer $outDir/cert_2.cer --sd-quorum 2 --security-domain-file "securitydomain$keyVaultName.json"
        CheckLastExitCode

        Write-Host "Assigning permissions to object ID $adminObjectId"
        $roleAssignment = az keyvault role assignment create --role "Managed HSM Crypto Officer" --scope "/" --assignee-object-id $adminObjectId --hsm-name $hsmName
        CheckLastExitCode
        $roleAssignment = az keyvault role assignment create --role "Managed HSM Crypto User" --scope "/" --assignee-object-id $adminObjectId --hsm-name $hsmName
        CheckLastExitCode
    }
    else {
        Write-Host "HSM is already active"
    }

    return $mhsmResult
}

function Create-KeyVault {
    param(
        [string]$resourceGroup,
        [string]$keyVaultName,
        [string]$adminObjectId,
        [string]$sku = "standard"
    )

    Write-Host "Creating $sku Key Vault $keyVaultName in resource group $resourceGroup"
    $keyVaultResult = (az keyvault create --resource-group $resourceGroup --name $keyVaultName --sku $sku --enable-rbac-authorization true --enable-purge-protection true) | ConvertFrom-Json

    # When the Key Vault already exists, $keyVaultResult will be null. In such cases, we try to pick the pre-existing Key Vault
    if ($null -eq $keyVaultResult) {
        $keyVaultResult = (az keyvault show --name $keyVaultName --resource-group $resourceGroup) | ConvertFrom-Json
    }

    Write-Host "Assigning 'Key Vault Administrator' permissions to $adminObjectId on Key Vault $($keyVaultResult.id)"
    $role = az role assignment create --role "Key Vault Administrator" --scope $keyVaultResult.id --assignee-object-id $adminObjectId --assignee-principal-type $(Get-Assignee-Principal-Type)

    return $keyVaultResult
}

function Import-Aes-Key {
    param(
        [string]$keyFilePath,

        [string]$keyName,

        [string]$mhsmName,

        [string]$ccePolicyHash,

        [string]$authority,

        [bool]$isImmutable = $false,

        [array]$keyOps = @()
    )


    $keyBytes = [System.IO.File]::ReadAllBytes($keyFilePath)

    $keyBase64Str = [Convert]::ToBase64String($keyBytes)
    $keyUrlEncoded = $keyBase64Str.TrimEnd("=").Replace("+", "-").Replace('/', '_') # needs to be base64URL encode

    $key = @{
        "kty"      = "oct-HSM"
        "key_ops"  = $keyOps
        "k"        = "$keyUrlEncoded"
        "key_size" = 256
    }

    $policyJson = @"
{
    "anyOf": [
        {
            "allOf": [
                {
                    "claim": "x-ms-sevsnpvm-hostdata",
                    "equals": "$ccePolicyHash"
                },
                {
                    "claim": "x-ms-compliance-status",
                    "equals": "azure-compliant-uvm"
                },
                {
                    "claim": "x-ms-attestation-type",
                    "equals": "sevsnpvm"
                }
            ],
            "authority": "$authority"
        }
    ],
    "version": "1.0.0"
}
"@

    $releasePolicyBytes = [System.Text.Encoding]::UTF8.GetBytes($policyJson)
    $releasePolicyBase64Str = [Convert]::ToBase64String($releasePolicyBytes)
    Write-Host "Release policy base64: $releasePolicyBase64Str"

    $importKeyRequest = @{
        "key"            = $key
        "hsm"            = $true
        "attributes"     = @{
            "exportable" = $true
        }
        "release_policy" = @{
            "contentType" = "application/json; charset=utf-8"
            "data"        = "$releasePolicyBase64Str"
            "immutable"   = $isImmutable
        }
    }

    $uri = "https://$mhsmName.managedhsm.azure.net/keys/$($keyName)?api-version=7.4"
    $accessTokenResult = (az account get-access-token --resource "https://managedhsm.azure.net") | ConvertFrom-Json

    Write-Host $uri
    Write-Host $($importKeyRequest | ConvertTo-Json)
    $secure = ConvertTo-SecureString $accessTokenResult.accessToken -AsPlainText 
    Invoke-WebRequest $uri -Body $($importKeyRequest | ConvertTo-Json) -Authentication Bearer -Token $secure -Method Put -ContentType "application/json; charset=utf-8"
}

function Import-Rsa-Key {
    param(
        [string]$privateKeyPath,

        [string]$keyName,

        [string]$mhsmName,

        [bool]$isMhsm,

        [string]$ccePolicyHash,

        [string]$authority,

        [bool]$isImmutable,

        [array]$keyOps = @()
    )

    $policyJson = @"
{
    "anyOf": [
        {
            "allOf": [
                {
                    "claim": "x-ms-sevsnpvm-hostdata",
                    "equals": "$ccePolicyHash"
                },
                {
                    "claim": "x-ms-compliance-status",
                    "equals": "azure-compliant-uvm"
                },
                {
                    "claim": "x-ms-attestation-type",
                    "equals": "sevsnpvm"
                }
            ],
            "authority": "$authority"
        }
    ],
    "version": "1.0.0"
}
"@
    if ($isMhsm) {
        az keyvault key import `
            --name $keyName `
            --pem-file $privateKeyPath `
            --policy $policyJson `
            --hsm-name $mhsmName `
            --exportable $true `
            --protection hsm `
            --ops $keyOps `
            --immutable $isImmutable
        CheckLastExitCode
    }
    else {
        az keyvault key import `
            --name $keyName `
            --pem-file $privateKeyPath `
            --policy $policyJson `
            --vault-name $mhsmName `
            --exportable $true `
            --protection hsm `
            --ops $keyOps `
            --immutable $isImmutable
        CheckLastExitCode
    }
}

function Save-Pem-To-KeyVault {
    param(
        [string]$keyVaultName,
        [string]$secretName,
        [string]$pemFilePath
    )

    Write-Host "Saving $pemFilePath to Key Vault $keyVaultName"
    $pembase64 = cat $pemFilePath | base64 --wrap=0
    az keyvault secret set --name $secretName --vault-name $keyVaultName --value $pembase64
}

function Get-Assignee-Principal-Type {
    if ($env:GITHUB_ACTIONS -eq "true") {
        return "ServicePrincipal"
    }
    else {
        return "User"
    }
}