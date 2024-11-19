param(
    [string]$persona = "$env:PERSONA",

    [string]$samplesRoot = "/home/samples",
    [string]$secretDir = "$samplesRoot/demo-resources/secret",
    [string]$publicDir = "$samplesRoot/demo-resources/public"
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../common/common.psm1

$memberCert = $persona +"_cert.pem"
$encryptionCert = $persona +"_enc_pubk.pem"

if ((Test-Path -Path "$publicDir/$memberCert") -or 
    (Test-Path -Path "$publicDir/$encryptionCert"))
{
    Write-Log Warning `
        "Identity and/or encryption key pairs for '$persona' already exist in '$publicDir'."
    return
}

# Generate member identity and encryption public-private key pair.
Write-Log Verbose `
    "Generating identity and encryption key pairs for '$persona' in '$secretDir'..." 
az cleanroom governance member keygenerator-sh | bash -s -- --gen-enc-key --name $persona --out $secretDir

# Share the public keys for the member.
cp "$secretDir/$memberCert" $publicDir
cp "$secretDir/$encryptionCert" $publicDir
Write-Log OperationCompleted `
    "Shared public keys for '$persona' to '$publicDir'." 

# Share the tenant ID.
$memberTenantId = az account show --query "tenantId" --output tsv
$memberTenantId | Out-File "$publicDir/$persona.tenantid"
Write-Log OperationCompleted `
    "Shared tenant ID '$memberTenantId' for '$persona' to '$publicDir/$persona.tenantid'."
