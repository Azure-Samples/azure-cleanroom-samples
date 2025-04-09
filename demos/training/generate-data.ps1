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
        "Generating untrained model for demo '$demo'..."

    pip install -r $PSScriptRoot/model/requirements.txt
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    python3 $PSScriptRoot/model/get_model.py --output-path $dataDir

    Write-Log OperationCompleted `
        "Generated untrained model for demo '$demo' in '$dataDir'."
}

#
# We'll use sample the CIFAR10 dataset to build and train the image classification model.
# CIFAR10 is a widely used dataset for machine learning research. It consists of 50,000 training images and 10,000 test images.
# All of them are of size 3x32x32, which means 3-channel color images of 32x32 pixels in size.
# The images are divided to 10 classes: ‘airplane’ (0), ‘automobile’ (1), ‘bird’ (2), ‘cat’ (3),
# ‘deer’ (4), ‘dog’ (5), ‘frog’ (6), ‘horse’ (7), ‘ship’ (8), ‘truck’ (9).
# The CIFAR10 dataset is available at https://www.cs.toronto.edu/~kriz/cifar.html.
if ("contosso" -eq $persona)
{
    pip install -r $PSScriptRoot/trainingData/requirements.txt
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    $dataDir = "$PSScriptRoot/datasource/contosso/data"
    Write-Log OperationStarted `
        "Downloading data for demo '$demo'"

    python3 $PSScriptRoot/trainingData/get_data.py --data-path $dataDir

    Write-Log OperationCompleted `
        "Downloaded data for demo '$demo' to '$dataDir'."
}