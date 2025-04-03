param(
    [ValidateSet("fabrikam", "contosso")]
    [string]$persona = "$env:PERSONA",
    [string]$demo = "$(Split-Path $PSScriptRoot -Leaf)"
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../../scripts/common/common.psm1

if (-not (("fabrikam", "contosso") -contains $persona))
{
    Write-Log Warning `
        "No action required for persona '$persona' in demo '$demo'."
    return
}

if ("fabrikam" -eq $persona)
{
    $dataDir = "$PSScriptRoot/datasource/fabrikam/model"
    Write-Log OperationStarted `
        "Generating inference model for demo '$demo'..."

    pip install -r $PSScriptRoot/model/requirements.txt
    python3 $PSScriptRoot/model/get_model.py --output-path $dataDir

    Write-Log OperationCompleted `
        "Generated inference model for demo '$demo' in '$dataDir'."
}

#
# Use sample dataset at https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes/tree/main
#
if ("contosso" -eq $persona)
{
    $dataDir = "$PSScriptRoot/datasource/contosso/data"
    Write-Log OperationStarted `
        "Downloading inference data for demo '$demo' from" `
        "'https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes/tree/main'" `
        "..."

    $src = "https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes/resolve/main"
    foreach ($dataset in ("test.parquet", "train.parquet", "validation.parquet"))
    {
        Write-Log Verbose `
            "Downloading '$src/$dataset'..."
        curl -L "$src/$dataset" -o "$dataDir/$dataset"
    }

    Write-Log OperationCompleted `
        "Downloaded inference data for demo '$demo' to '$dataDir'."
}