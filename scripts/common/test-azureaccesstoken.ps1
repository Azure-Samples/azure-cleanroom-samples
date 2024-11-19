. $PSScriptRoot/write-log.ps1

function Test-AzureAccessToken {
    #https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
    $ErrorActionPreference = 'Stop'
    $PSNativeCommandUseErrorActionPreference = $true

    Write-Log OperationStarted `
        "Verifying Azure access token..."
    $token = (az account get-access-token | ConvertFrom-Json)
    if ([datetime]$token.expiresOn -le [datetime]::Now)
    {
        Write-Log Error `
            "Access token has expired - login again."
        throw
    }

    Write-Log Verbose `
        "Azure login details:"
    az account show

    Write-Log OperationCompleted `
        "Verified Azure access token."
}