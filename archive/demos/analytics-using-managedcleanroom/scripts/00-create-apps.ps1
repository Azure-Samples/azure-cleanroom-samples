<#
.SYNOPSIS
    Creates Azure AD applications for E2E testing.

.DESCRIPTION
    Creates two Azure AD app registrations with service principals (northwind, woodgrove)
    and stores credentials in Azure Key Vault. Designed for automated E2E testing.

.PARAMETER appNamePrefix
    Prefix for app display names (default: cleanroom-e2e).

.PARAMETER keyVaultName
    Existing Key Vault to store credentials. If not provided and --createKeyVault
    is specified, a new KV will be created.

.PARAMETER createKeyVault
    Create a new Key Vault for storing credentials.

.PARAMETER resourceGroup
    Resource group for creating Key Vault (required if --createKeyVault).

.PARAMETER location
    Azure region for Key Vault (default: westus).

.PARAMETER subscription
    Optional Azure subscription name or ID.

.PARAMETER outDir
    Output directory for app metadata files (default: ./generated).
#>
param(
    [string]$appNamePrefix = "cleanroom-e2e",
    [string]$keyVaultName,
    [switch]$createKeyVault,
    [string]$resourceGroup,
    [string]$location = "westus",
    [string]$subscription,
    [string]$outDir = "./generated"
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

function Invoke-AzCommand {
    param([string[]]$Arguments)
    $cmdLine = "az $($Arguments -join ' ')"
    Write-Host "[CMD] $cmdLine" -ForegroundColor DarkGray
    & az @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
}

function Invoke-AzSafe {
    param([string[]]$Arguments)
    $PSNativeCommandUseErrorActionPreference = $false
    $result = & az @Arguments 2>$null
    if ($LASTEXITCODE -eq 0) { return $result }
    return $null
}

Write-Host "=== Creating Azure AD Applications for E2E Testing ===" -ForegroundColor Cyan

if ($subscription) {
    Write-Host "Setting subscription to '$subscription'..." -ForegroundColor Yellow
    Invoke-AzCommand @("account", "set", "--subscription", $subscription)
}

$currentAccount = Invoke-AzCommand @("account", "show") | ConvertFrom-Json
$tenantId = $currentAccount.tenantId
$subscriptionId = $currentAccount.id

Write-Host "Tenant ID: $tenantId" -ForegroundColor Yellow
Write-Host "Subscription ID: $subscriptionId" -ForegroundColor Yellow

if ($createKeyVault) {
    if (-not $resourceGroup) {
        Write-Host "ERROR: --resourceGroup required when --createKeyVault is specified." -ForegroundColor Red
        exit 1
    }
    
    Write-Host "`n=== Creating Resource Group ===" -ForegroundColor Cyan
    $rgExists = Invoke-AzSafe @("group", "exists", "--name", $resourceGroup)
    if ($rgExists -eq "false") {
        Invoke-AzCommand @("group", "create", "--name", $resourceGroup, "--location", $location, "--output", "none")
        Write-Host "Resource group '$resourceGroup' created." -ForegroundColor Green
    } else {
        Write-Host "Resource group '$resourceGroup' already exists (skipped)." -ForegroundColor Yellow
    }
    
    Write-Host "`n=== Creating Key Vault ===" -ForegroundColor Cyan
    $bytes = [System.Text.Encoding]::UTF8.GetBytes("$subscriptionId-$resourceGroup")
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $hashBytes = $sha.ComputeHash($bytes)
    $hash = (-join ($hashBytes | ForEach-Object { $_.ToString("x2") })).Substring(0, 8)
    
    if (-not $keyVaultName) {
        $keyVaultName = "cr-e2e-secrets-$hash"
    }
    
    $existingKv = Invoke-AzSafe @("keyvault", "show", "--name", $keyVaultName, "--output", "json")
    if (-not $existingKv) {
        Invoke-AzCommand @("keyvault", "create", 
            "--name", $keyVaultName, 
            "--resource-group", $resourceGroup, 
            "--location", $location, 
            "--enable-rbac-authorization", "true",
            "--output", "none")
        Write-Host "Key Vault '$keyVaultName' created." -ForegroundColor Green
        
        $callerObjectId = Invoke-AzCommand @("ad", "signed-in-user", "show", "--query", "id", "-o", "tsv")
        $kvId = Invoke-AzCommand @("keyvault", "show", "--name", $keyVaultName, "--query", "id", "-o", "tsv")
        
        Write-Host "Assigning 'Key Vault Secrets Officer' to current user..." -ForegroundColor Cyan
        Invoke-AzSafe @("role", "assignment", "create", 
            "--role", "Key Vault Secrets Officer", 
            "--assignee-object-id", $callerObjectId, 
            "--assignee-principal-type", "User", 
            "--scope", $kvId, 
            "--output", "none")
        
        Write-Host "Waiting for RBAC propagation (15 seconds)..." -ForegroundColor Yellow
        Start-Sleep -Seconds 15
    } else {
        Write-Host "Key Vault '$keyVaultName' already exists (using existing)." -ForegroundColor Yellow
    }
}

if (-not $keyVaultName) {
    Write-Host "ERROR: --keyVaultName required (or use --createKeyVault)." -ForegroundColor Red
    exit 1
}

$appsDir = Join-Path $outDir "apps"
if (-not (Test-Path $appsDir)) {
    New-Item -ItemType Directory -Path $appsDir -Force | Out-Null
}

function New-AppRegistration {
    param(
        [string]$Persona,
        [string]$DisplayName
    )
    
    Write-Host "`n=== Creating App Registration: $DisplayName ===" -ForegroundColor Cyan
    
    $existingApp = Invoke-AzSafe @("ad", "app", "list", "--display-name", $DisplayName) | ConvertFrom-Json
    if ($existingApp -and $existingApp.Count -gt 0) {
        Write-Host "App '$DisplayName' already exists. Using existing app." -ForegroundColor Yellow
        $app = $existingApp[0]
        $appId = $app.appId
        $objectId = $app.id
        
        $sp = Invoke-AzSafe @("ad", "sp", "list", "--filter", "appId eq '$appId'") | ConvertFrom-Json
        if (-not $sp -or $sp.Count -eq 0) {
            Write-Host "Creating service principal..." -ForegroundColor Cyan
            $sp = Invoke-AzCommand @("ad", "sp", "create", "--id", $appId) | ConvertFrom-Json
        } else {
            $sp = $sp[0]
        }
    } else {
        Write-Host "Creating new app registration..." -ForegroundColor Cyan
        # Microsoft tenant requires ServiceManagementReference
        $appPayload = @{
            displayName = $DisplayName
            signInAudience = "AzureADMyOrg"
            serviceManagementReference = "a67ab109-c3b4-45f5-865a-ae3dcde5ee1b"
        } | ConvertTo-Json -Compress
        
        $app = az rest --method POST --url "https://graph.microsoft.com/v1.0/applications" --headers "Content-Type=application/json" --body $appPayload | ConvertFrom-Json
        $appId = $app.appId
        $objectId = $app.id
        
        Write-Host "Creating service principal..." -ForegroundColor Cyan
        $sp = Invoke-AzCommand @("ad", "sp", "create", "--id", $appId) | ConvertFrom-Json
    }
    
    # Certificate handling (for both new and existing apps)
    Write-Host "Setting up certificate credential..." -ForegroundColor Cyan
    $certName = "$Persona-auth-cert"
    Write-Host "  Checking certificate '$certName' in Key Vault..." -ForegroundColor Cyan
    
    # Check if certificate already exists
    $existingCert = Invoke-AzSafe @("keyvault", "certificate", "show", "--vault-name", $keyVaultName, "--name", $certName, "--output", "json")
    
    if (-not $existingCert) {
        # Create self-signed certificate (valid for 2 years)
        Write-Host "  Creating new certificate..." -ForegroundColor Cyan
        $certPolicy = @{
            issuerParameters = @{ name = "Self" }
            keyProperties = @{
                exportable = $true
                keySize = 2048
                keyType = "RSA"
                reuseKey = $false
            }
            secretProperties = @{ contentType = "application/x-pkcs12" }
            x509CertificateProperties = @{
                subject = "CN=$DisplayName"
                validityInMonths = 24
            }
        } | ConvertTo-Json -Compress -Depth 10
        
        Invoke-AzCommand @("keyvault", "certificate", "create", 
            "--vault-name", $keyVaultName, 
            "--name", $certName,
            "--policy", $certPolicy,
            "--output", "none")
        
        Write-Host "  Waiting for certificate creation..." -ForegroundColor Yellow
        Start-Sleep -Seconds 10
    } else {
        Write-Host "  Certificate '$certName' already exists (reusing)." -ForegroundColor Yellow
    }
    
    # Check if app already has this certificate credential
    $existingCreds = Invoke-AzSafe @("ad", "app", "credential", "list", "--id", $appId, "--cert") | ConvertFrom-Json
    $certInfo = Invoke-AzCommand @("keyvault", "certificate", "show", "--vault-name", $keyVaultName, "--name", $certName) | ConvertFrom-Json
    $certThumbprint = $certInfo.x509ThumbprintHex
    
    $hasCert = $false
    if ($existingCreds) {
        foreach ($cred in $existingCreds) {
            if ($cred.customKeyIdentifier -eq $certThumbprint) {
                $hasCert = $true
                Write-Host "  Certificate already assigned to app (skipping upload)." -ForegroundColor Yellow
                break
            }
        }
    }
    
    if (-not $hasCert) {
        # Download certificate to temporary file
        $tempCertFile = [System.IO.Path]::GetTempFileName()
        Invoke-AzCommand @("keyvault", "certificate", "download",
            "--vault-name", $keyVaultName,
            "--name", $certName,
            "--file", $tempCertFile,
            "--encoding", "PEM")
        
        # Upload certificate credential to app registration
        Write-Host "  Uploading certificate to app registration..." -ForegroundColor Cyan
        $credResult = Invoke-AzCommand @("ad", "app", "credential", "reset",
            "--id", $appId,
            "--cert", $tempCertFile,
            "--append") | ConvertFrom-Json
        
        Remove-Item $tempCertFile -ErrorAction SilentlyContinue
        
        Write-Host "  Certificate credential added successfully." -ForegroundColor Green
    } else {
    }
    
    $servicePrincipalId = $sp.id
    
    Write-Host "  App ID: $appId" -ForegroundColor Green
    Write-Host "  Object ID: $objectId" -ForegroundColor Green
    Write-Host "  Service Principal ID: $servicePrincipalId" -ForegroundColor Green
    
    Write-Host "Storing credentials in Key Vault..." -ForegroundColor Cyan
    Invoke-AzCommand @("keyvault", "secret", "set", 
        "--vault-name", $keyVaultName, 
        "--name", "$Persona-app-id", 
        "--value", $appId, 
        "--output", "none")
    
    # Store certificate name instead of app secret
    Invoke-AzCommand @("keyvault", "secret", "set", 
        "--vault-name", $keyVaultName, 
        "--name", "$Persona-cert-name", 
        "--value", "$Persona-auth-cert", 
        "--output", "none")
    
    Invoke-AzCommand @("keyvault", "secret", "set", 
        "--vault-name", $keyVaultName, 
        "--name", "$Persona-tenant-id", 
        "--value", $tenantId, 
        "--output", "none")
    
    Invoke-AzCommand @("keyvault", "secret", "set", 
        "--vault-name", $keyVaultName, 
        "--name", "$Persona-sp-object-id", 
        "--value", $servicePrincipalId, 
        "--output", "none")
    
    $appMetadata = @{
        appId               = $appId
        objectId            = $objectId
        servicePrincipalId  = $servicePrincipalId
        tenantId            = $tenantId
        displayName         = $DisplayName
        keyVaultName        = $keyVaultName
        authMethod          = "certificate"
        secretNames         = @{
            appId       = "$Persona-app-id"
            certName    = "$Persona-cert-name"
            tenantId    = "$Persona-tenant-id"
            spObjectId  = "$Persona-sp-object-id"
        }
    }
    
    $metadataPath = Join-Path $appsDir "$Persona-app.json"
    $appMetadata | ConvertTo-Json -Depth 5 | Out-File -FilePath $metadataPath -Encoding utf8
    Write-Host "App metadata saved to: $metadataPath" -ForegroundColor Green
    
    return $appMetadata
}

$northwindApp = New-AppRegistration -Persona "northwind" -DisplayName "$appNamePrefix-northwind"
$woodgroveApp = New-AppRegistration -Persona "woodgrove" -DisplayName "$appNamePrefix-woodgrove"

Write-Host "`n=== Apps Created Successfully ===" -ForegroundColor Green
Write-Host "`nNorthwind App:" -ForegroundColor Cyan
Write-Host "  App ID: $($northwindApp.appId)" -ForegroundColor Yellow
Write-Host "  Service Principal ID: $($northwindApp.servicePrincipalId)" -ForegroundColor Yellow
Write-Host "  Metadata: $(Join-Path $appsDir 'northwind-app.json')" -ForegroundColor Yellow

Write-Host "`nWoodgrove App:" -ForegroundColor Cyan
Write-Host "  App ID: $($woodgroveApp.appId)" -ForegroundColor Yellow
Write-Host "  Service Principal ID: $($woodgroveApp.servicePrincipalId)" -ForegroundColor Yellow
Write-Host "  Metadata: $(Join-Path $appsDir 'woodgrove-app.json')" -ForegroundColor Yellow

Write-Host "`nKey Vault:" -ForegroundColor Cyan
Write-Host "  Name: $keyVaultName" -ForegroundColor Yellow
Write-Host "  Authentication: Certificate-based" -ForegroundColor Yellow
Write-Host "  Stored secrets:" -ForegroundColor Yellow
Write-Host "    - northwind-app-id, northwind-cert-name, northwind-tenant-id, northwind-sp-object-id" -ForegroundColor Yellow
Write-Host "    - woodgrove-app-id, woodgrove-cert-name, woodgrove-tenant-id, woodgrove-sp-object-id" -ForegroundColor Yellow
Write-Host "  Certificates: northwind-auth-cert, woodgrove-auth-cert" -ForegroundColor Yellow

Write-Host "`nNext Steps:" -ForegroundColor Green
Write-Host "  1. Run 00a-assign-app-permissions.ps1 to assign RBAC roles (after creating resources)" -ForegroundColor Yellow
Write-Host "  2. Use --appKeyVaultName $keyVaultName in other scripts for app-based auth" -ForegroundColor Yellow
