function Get-InputData {
    param(
        [string]
        $dataDir,

        $handles,

        [DateTimeOffset]
        $startDate,

        [Parameter(Mandatory = $true)]
        [ValidateSet("csv", "json", "parquet")][string]
        $format = "csv",

        [Parameter(Mandatory = $true)]
        [string]
        $schemaFields = "date:date,time:string,author:string,mentions:string",

        [string]
        $baseUrl = "https://github.com/Azure-Samples/Synapse/raw/refs/heads/main/Data/Tweets"
    )

    $ErrorActionPreference = 'Stop'
    $PSNativeCommandUseErrorActionPreference = $true

    $scriptDir = $PSScriptRoot
    $currentDate = $startDate
    foreach ($handle in $handles) {
        New-Item -ItemType Directory -Force -Path "$dataDir/csv" | Out-Null
        $outputDir = "$dataDir/csv"
        if ($startDate -ne [DateTimeOffset]::MinValue) {
            $outputDir = Join-Path "$dataDir/csv" ($currentDate.ToString("yyyy-MM-dd"))
            $currentDate = $currentDate.AddDays(1)
        }

        New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

        $csvUrl = "$baseUrl/$handle.csv"
        $csvPath = Join-Path $outputDir "$handle.csv"

        Write-Host "Downloading data for $handle from $csvUrl..."
        
        # Retry logic for downloads (handles transient network failures)
        $maxRetries = 3
        $retryDelay = 5
        $downloaded = $false
        
        for ($attempt = 1; $attempt -le $maxRetries; $attempt++) {
            try {
                Invoke-WebRequest -Uri $csvUrl -OutFile $csvPath -ErrorAction Stop
                $downloaded = $true
                Write-Host "  Downloaded successfully." -ForegroundColor Green
                break
            }
            catch {
                if ($attempt -lt $maxRetries) {
                    Write-Host "  Download attempt $attempt failed: $_" -ForegroundColor Yellow
                    Write-Host "  Retrying in $retryDelay seconds..." -ForegroundColor Yellow
                    Start-Sleep -Seconds $retryDelay
                }
                else {
                    Write-Host "  Download failed after $maxRetries attempts: $_" -ForegroundColor Red
                    throw "Failed to download data for $handle after $maxRetries attempts"
                }
            }
        }
    }
    if ($format -ne "csv") {
        New-Item -ItemType Directory -Force -Path "$dataDir/$format" | Out-Null
        $pythonScript = "$scriptDir/convert_data.py"
        $argsList = @(
            $pythonScript,
            "--data-dir", "$dataDir/csv",
            "--output-dir", "$dataDir/$format",
            "--format", $format,
            "--schema-fields", $schemaFields
        )
        python3 @argsList
        if ($LASTEXITCODE -ne 0) {
            throw "Error executing data conversion script"
        }
    }
}
function Get-PublisherData {
    param(
        [string]
        $dataDir,

        [DateTimeOffset]
        $startDate = [DateTimeOffset]::MinValue,

        [Parameter(Mandatory = $true)]
        [ValidateSet("csv", "json", "parquet")][string]
        $format = "csv",

        [Parameter(Mandatory = $true)]
        [string]
        $schemaFields = "date:date,time:string,author:string,mentions:string"
    )
    $handles = ("RahulPotharajuTweets", "raghurwiTweets", "MikeDoesBigDataTweets", "SQLCindyTweets")
    Get-InputData -dataDir $dataDir -handles $handles -startDate $startDate -format $format -schemaFields $schemaFields
}

function Get-ConsumerData {
    param(
        [string]
        $dataDir,

        [DateTimeOffset]
        $startDate = [DateTimeOffset]::MinValue,

        [Parameter(Mandatory = $true)]
        [ValidateSet("csv", "json", "parquet")][string]
        $format = "csv",

        [Parameter(Mandatory = $true)]
        [string]
        $schemaFields = "date:date,time:string,author:string,mentions:string"
    )
    $handles = ("BrigitMurtaughTweets", "FranmerMSTweets", "JeremyLiknessTweets", "mwinkleTweets")
    Get-InputData -dataDir $dataDir -handles $handles -startDate $startDate -format $format -schemaFields $schemaFields
}