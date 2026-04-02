<#
.SYNOPSIS
    Generates synthetic demo CSV data for a single persona.

.DESCRIPTION
    Northwind data: audience_id, hashed_email, annual_income, region
    Woodgrove data: user_id, hashed_email, purchase_history

.PARAMETER persona
    The collaborator persona (northwind or woodgrove).

.EXAMPLE
    .\generate-data.ps1 -persona northwind
    .\generate-data.ps1 -persona woodgrove
#>
param(
    [Parameter(Mandatory)]
    [ValidateSet("northwind", "woodgrove", IgnoreCase = $false)]
    [string]$persona
)

$ErrorActionPreference = 'Stop'

# ── Internal settings (not exposed as parameters) ────────────────────
$rowCount   = 100000
$overlapPct = 0.20
$seed       = 42

# ── Helpers ──────────────────────────────────────────────────────────
function New-Sha256 ([string]$text) {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($text)
    $hash  = [System.Security.Cryptography.SHA256]::HashData($bytes)
    return [BitConverter]::ToString($hash).Replace('-', '').ToLowerInvariant()
}

# ── Configuration ────────────────────────────────────────────────────
$regions = @('US', 'UK', 'IN', 'CA')
$purchaseItems = @(
    'Floor Rug', 'Bar Stools', 'TV Stand', 'Kids Bunk Bed', 'Lamp',
    'TV Unit', 'Dining Chairs', 'Side Table', 'Coffee Table', 'Bookshelf',
    'Console Table', 'Bed', 'Wardrobe', 'Gaming Chair', 'Recliner',
    'Office Desk and Chair', 'Futon', 'Sectional Sofa', 'Sofa', 'Patio Set'
)
$dates = @('2025-09-01', '2025-09-02', '2025-09-03', '2025-09-04')

$overlapCount  = [int]($rowCount * $overlapPct)
$uniqueCount   = $rowCount - $overlapCount

# ── Build the shared (overlap) email pool ────────────────────────────
# These emails appear in BOTH datasets. Spread equally across the 4 regions.
$rng = [System.Random]::new($seed)
$sharedEmails = [System.Collections.Generic.List[string]]::new($overlapCount)
for ($i = 0; $i -lt $overlapCount; $i++) {
    $sharedEmails.Add((New-Sha256 "shared-user-$i"))
}

# ── Output directory ─────────────────────────────────────────────────
$scriptDir = $PSScriptRoot
$dataDir   = Join-Path $scriptDir "datasource" $persona "input" "csv"
if (Test-Path $dataDir) { Remove-Item $dataDir -Recurse -Force }

# Pre-create date folders
$dateFolders = @()
foreach ($d in $dates) {
    $folder = Join-Path $dataDir $d
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
    $dateFolders += $folder
}

$folderCount  = $dateFolders.Count
$rowsPerFolder = [int][Math]::Ceiling($rowCount / $folderCount)

Write-Host "Generating $rowCount rows for '$persona'..." -ForegroundColor Yellow

# ── Generate rows ────────────────────────────────────────────────────
# Split overlap emails equally across folders so each folder has its share.
$overlapPerFolder = [int][Math]::Ceiling($overlapCount / $folderCount)
$uniquePerFolder  = $rowsPerFolder - $overlapPerFolder

$globalUniqueIdx = 0

for ($fi = 0; $fi -lt $folderCount; $fi++) {
    $sb = [System.Text.StringBuilder]::new(1MB)
    $overlapStart = $fi * $overlapPerFolder
    $overlapEnd   = [Math]::Min($overlapStart + $overlapPerFolder, $overlapCount)

    # Write overlap rows for this folder
    for ($i = $overlapStart; $i -lt $overlapEnd; $i++) {
        $email  = $sharedEmails[$i]
        $region = $regions[$i % $regions.Count]

        if ($persona -eq 'northwind') {
            $id     = "A{0:D6}" -f ($i + 1)
            $income = 30000 + $rng.Next(0, 171) * 1000   # 30k–200k
            [void]$sb.AppendLine("$id,$email,$income,$region")
        }
        else {
            $id   = "U{0:D6}" -f ($i + 1)
            $item = $purchaseItems[$rng.Next($purchaseItems.Count)]
            [void]$sb.AppendLine("$id,$email,$item")
        }
    }

    # Write unique rows for this folder
    for ($j = 0; $j -lt $uniquePerFolder; $j++) {
        $idx    = $globalUniqueIdx++
        $email  = New-Sha256 "$persona-unique-$idx"
        $region = $regions[$idx % $regions.Count]

        if ($persona -eq 'northwind') {
            $id     = "A{0:D6}" -f ($overlapCount + $idx + 1)
            $income = 30000 + $rng.Next(0, 171) * 1000
            [void]$sb.AppendLine("$id,$email,$income,$region")
        }
        else {
            $id   = "U{0:D6}" -f ($overlapCount + $idx + 1)
            $item = $purchaseItems[$rng.Next($purchaseItems.Count)]
            [void]$sb.AppendLine("$id,$email,$item")
        }
    }

    $csvPath = Join-Path $dateFolders[$fi] "$persona-data.csv"
    [System.IO.File]::WriteAllText($csvPath, $sb.ToString())
    $size = [Math]::Round((Get-Item $csvPath).Length / 1MB, 2)
    Write-Host "  $($dates[$fi]): $csvPath ($size MB)" -ForegroundColor Cyan
}

$totalSize = (Get-ChildItem $dataDir -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "`n[OK] Demo data generated for '$persona'." -ForegroundColor Green
Write-Host "  Data directory: $dataDir"
Write-Host "  Total size:     $([Math]::Round($totalSize, 2)) MB"
Write-Host "  Overlap emails: $overlapCount ($(($overlapPct * 100))% of $rowCount)"
