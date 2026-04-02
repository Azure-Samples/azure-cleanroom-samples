<#
.SYNOPSIS
    Fetches and caches frontend self-signed certificate for TLS validation.

.DESCRIPTION
    Retrieves the self-signed certificate from the frontend /report endpoint
    and caches it for use in subsequent API calls. This avoids using the
    insecure -k flag for every request.

.PARAMETER frontendEndpoint
    Frontend service URL (e.g., https://dogfood.workload-frontendwestus.cleanroom.cloudapp.azure-test.net).

.PARAMETER outDir
    Output directory for certificate file (default: ./generated).

.PARAMETER force
    Force re-fetch even if certificate already cached.

.OUTPUTS
    PSCustomObject with certPath and settings properties.
#>
param(
    [Parameter(Mandatory)]
    [string]$frontendEndpoint,

    [string]$outDir = "./generated",
    [switch]$force
)

$ErrorActionPreference = 'Stop'

$certFile = Join-Path $outDir "frontend-ca-cert.pem"

if ((Test-Path $certFile) -and -not $force) {
    Write-Host "Using cached certificate: $certFile" -ForegroundColor Yellow
    
    return [PSCustomObject]@{
        certPath = $certFile
        cached = $true
    }
}

Write-Host "Fetching frontend certificate from $frontendEndpoint/report..." -ForegroundColor Cyan

try {
    $handler = [System.Net.Http.HttpClientHandler]::new()
    $handler.ServerCertificateCustomValidationCallback = `
        [System.Net.Http.HttpClientHandler]::DangerousAcceptAnyServerCertificateValidator
    
    $client = [System.Net.Http.HttpClient]::new($handler)
    $client.Timeout = [TimeSpan]::FromSeconds(30)
    
    $reportUrl = "$($frontendEndpoint.TrimEnd('/'))/report"
    Write-Host "Calling: $reportUrl" -ForegroundColor DarkGray
    
    $response = $client.GetAsync($reportUrl).Result
    
    if (-not $response.IsSuccessStatusCode) {
        throw "HTTP $($response.StatusCode): $($response.ReasonPhrase)"
    }
    
    $body = $response.Content.ReadAsStringAsync().Result | ConvertFrom-Json
    
    if (-not $body.certificate) {
        throw "No certificate found in /report response"
    }
    
    if (-not (Test-Path $outDir)) {
        New-Item -ItemType Directory -Path $outDir -Force | Out-Null
    }
    
    $body.certificate | Out-File -FilePath $certFile -Encoding ascii
    Write-Host "Certificate saved to: $certFile" -ForegroundColor Green
    
    if ($body.settings) {
        Write-Host "`nFrontend Settings:" -ForegroundColor Cyan
        Write-Host "  Consortium Manager: $($body.settings.CONSORTIUM_MANAGER_ENDPOINT)" -ForegroundColor Yellow
        Write-Host "  Membership Manager: $($body.settings.MEMBERSHIP_MANAGER_ENDPOINT)" -ForegroundColor Yellow
        Write-Host "  Analytics Workload ID: $($body.settings.ANALYTICS_WORKLOAD_ID)" -ForegroundColor Yellow
    }
    
    return [PSCustomObject]@{
        certPath = $certFile
        settings = $body.settings
        cached = $false
    }
}
catch {
    Write-Host "ERROR: Failed to fetch frontend certificate." -ForegroundColor Red
    Write-Host "  Endpoint: $frontendEndpoint" -ForegroundColor Red
    Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
    throw
}
finally {
    if ($client) { $client.Dispose() }
    if ($handler) { $handler.Dispose() }
}
