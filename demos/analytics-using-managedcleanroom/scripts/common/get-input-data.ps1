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
        Invoke-WebRequest -Uri $csvUrl -OutFile $csvPath
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