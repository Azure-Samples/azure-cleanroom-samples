<#
.SYNOPSIS
    Downloads sample Twitter CSV data for a single persona.

.DESCRIPTION
    Each collaborator runs this to download their own demo data.
    Northwind (publisher) and Woodgrove (consumer) get different Twitter handle datasets.

.PARAMETER persona
    The collaborator persona (northwind or woodgrove).

.EXAMPLE
    .\generate-data.ps1 -persona northwind
#>
param(
    [Parameter(Mandatory)]
    [ValidateSet("northwind", "woodgrove", IgnoreCase = $false)]
    [string]$persona
)

$ErrorActionPreference = 'Stop'

$scriptDir = $PSScriptRoot
$commonScripts = "$scriptDir/../scripts/common"

. $commonScripts/get-input-data.ps1

$dataDir = "$scriptDir/datasource/$persona/input"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

if ($persona -eq "northwind") {
    Write-Host "Downloading publisher (Northwind) data..." -ForegroundColor Yellow
    $today = [DateTimeOffset]"2025-09-01"
    Get-PublisherData -dataDir $dataDir -startDate $today -format "csv" -schemaFields "date:date,time:string,author:string,mentions:string"
}
else {
    Write-Host "Downloading consumer (Woodgrove) data..." -ForegroundColor Yellow
    Get-ConsumerData -dataDir $dataDir -format "csv" -schemaFields "date:date,time:string,author:string,mentions:string"
}

Write-Host "[OK] Demo data generated for '$persona'." -ForegroundColor Green
Write-Host "  Data directory: $dataDir"
