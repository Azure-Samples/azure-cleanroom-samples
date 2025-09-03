function Prompt {
    "PS [" + $env:PERSONA + "][" + $env:RESOURCE_GROUP + "] " + (Get-Location) + "> "
}

Set-PSReadLineOption -PredictionSource History
set-PSReadLineOption -PredictionViewStyle ListView