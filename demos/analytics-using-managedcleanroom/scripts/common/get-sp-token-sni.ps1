<#
.SYNOPSIS
    Acquires an access token for a service principal using MSAL with SNI (Subject Name/Issuer)
    certificate-based authentication via trustedCertificateSubjects.

.DESCRIPTION
    In the MSFT tenant, corporate policy blocks direct cert/password credentials on app
    registrations. Instead, apps use trustedCertificateSubjects in the manifest with certs
    issued by an integrated CA (OneCert). This script uses Python MSAL to acquire tokens
    with the x5c claim (SNI auth).

    The acquired token is saved to a file and can be used for frontend REST calls.

.PARAMETER appId
    Application (client) ID of the app registration.

.PARAMETER tenantId
    Azure AD tenant ID.

.PARAMETER certPemPath
    Path to PEM file containing private key + cert chain (from KV secret download).

.PARAMETER scope
    Token scope (default: https://management.azure.com/.default).

.PARAMETER tokenOutputFile
    Path to save the acquired token (default: /tmp/sp-access-token.txt).

.OUTPUTS
    Returns the access token string. Also saves to $tokenOutputFile.

.EXAMPLE
    $token = ./scripts/common/get-sp-token-sni.ps1 -appId "<your-app-id>" -tenantId "<your-tenant-id>" -certPemPath "./generated/certs/<your-cert-name>.pem"
#>
param(
    [Parameter(Mandatory)]
    [string]$appId,

    [Parameter(Mandatory)]
    [string]$tenantId,

    [Parameter(Mandatory)]
    [string]$certPemPath,

    [string]$scope = "https://management.azure.com/.default",

    [string]$tokenOutputFile = "/tmp/sp-access-token.txt"
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $certPemPath)) {
    throw "Certificate PEM file not found: $certPemPath"
}

# Use Python MSAL for SNI auth (az login doesn't support trustedCertificateSubjects)
$pythonScript = @"
import msal, json, re, sys
from cryptography import x509
from cryptography.hazmat.primitives import hashes

APP_ID = "$appId"
TENANT_ID = "$tenantId"
CERT_PATH = "$certPemPath"
SCOPE = ["$scope"]

with open(CERT_PATH, 'r') as f:
    pem_content = f.read()

key_match = re.search(r'(-----BEGIN (?:RSA )?PRIVATE KEY-----.*?-----END (?:RSA )?PRIVATE KEY-----)', pem_content, re.DOTALL)
cert_matches = re.findall(r'(-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----)', pem_content, re.DOTALL)

if not key_match or not cert_matches:
    print("ERROR: Could not extract key/cert from PEM file", file=sys.stderr)
    sys.exit(1)

private_key_pem = key_match.group(1)

# Find the leaf cert (non-CA cert with our domain)
leaf_cert = None
for cert_pem in cert_matches:
    cert_obj = x509.load_pem_x509_certificate(cert_pem.encode())
    try:
        bc = cert_obj.extensions.get_extension_for_class(x509.BasicConstraints)
        if bc.value.ca:
            continue
    except x509.ExtensionNotFound:
        pass
    leaf_cert = cert_pem
    leaf_obj = cert_obj
    break

if not leaf_cert:
    # Fallback: pick the cert whose subject contains our domain
    for cert_pem in cert_matches:
        cert_obj = x509.load_pem_x509_certificate(cert_pem.encode())
        cn = cert_obj.subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)
        if cn and "cleanroom" in cn[0].value.lower():
            leaf_cert = cert_pem
            leaf_obj = cert_obj
            break

if not leaf_cert:
    print("ERROR: Could not identify leaf certificate", file=sys.stderr)
    sys.exit(1)

thumbprint = leaf_obj.fingerprint(hashes.SHA1()).hex()
chain_certs = [c for c in cert_matches if c != leaf_cert]
ordered_chain = [leaf_cert] + chain_certs

app = msal.ConfidentialClientApplication(
    APP_ID,
    authority=f"https://login.microsoftonline.com/{TENANT_ID}",
    client_credential={
        "private_key": private_key_pem,
        "thumbprint": thumbprint,
        "public_certificate": "\n".join(ordered_chain),
    },
)

result = app.acquire_token_for_client(scopes=SCOPE)

if "access_token" in result:
    print(result["access_token"])
else:
    print(f"ERROR: {result.get('error')}: {result.get('error_description','')}", file=sys.stderr)
    sys.exit(1)
"@

Write-Host "Acquiring SP token via MSAL SNI..." -ForegroundColor Cyan
$token = python3 -c $pythonScript
if ($LASTEXITCODE -ne 0 -or -not $token) {
    throw "Failed to acquire SP token via MSAL SNI"
}

# Save token to file
$token | Out-File -FilePath $tokenOutputFile -NoNewline -Encoding utf8
Write-Host "Token saved to: $tokenOutputFile" -ForegroundColor Green

# Decode and display claims
$parts = $token.Split('.')
$payload = $parts[1]
$padLen = (4 - $payload.Length % 4) % 4
$padded = $payload + ('=' * $padLen)
$claims = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($padded)) | ConvertFrom-Json
Write-Host "  oid: $($claims.oid)" -ForegroundColor Yellow
Write-Host "  appid: $($claims.appid)" -ForegroundColor Yellow
Write-Host "  tid: $($claims.tid)" -ForegroundColor Yellow

return $token
