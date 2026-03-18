###############################################################################################
# utils.ps1
#
# Shared utility functions used by multiple scripts. Dot-source this file at the top of
# any script that needs these helpers:
#   . "$PSScriptRoot/common/utils.ps1"     (from scripts/)
#   . "$PSScriptRoot/utils.ps1"            (from scripts/common/)
###############################################################################################

<#
.SYNOPSIS
    Runs an az CLI command, returning $null instead of throwing on failure.
#>
function Invoke-AzSafe {
    param([string[]]$Arguments)
    $PSNativeCommandUseErrorActionPreference = $false
    $result = & az @Arguments 2>$null
    if ($LASTEXITCODE -eq 0) { return $result }
    return $null
}

<#
.SYNOPSIS
    Returns a deterministic hex hash string from a seed value.
    Used to derive globally-unique Azure resource names.
#>
function Get-ResourceNameHash {
    param([string]$seed)
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($seed)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $hash = $sha.ComputeHash($bytes)
    $hex = -join ($hash | ForEach-Object { $_.ToString("x2") })
    return $hex
}
