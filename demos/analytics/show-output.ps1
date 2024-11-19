param(
    [Parameter(Mandatory = $true)]
    [string]$contractId,

    [string]$persona = "$env:PERSONA",

    [string]$samplesRoot = "/home/samples",
    [string]$privateDir = "$samplesRoot/demo-resources/private",
    [string]$publicDir = "$samplesRoot/demo-resources/public",
    [string]$demosRoot = "$samplesRoot/demos",

    [string]$cleanRoomName = "cleanroom-$contractId",
    [string]$cleanroomEndpoint = (Get-Content "$publicDir/$cleanRoomName.endpoint"),

    [string]$datastoreDir = "$privateDir/datastores",

    [string]$demo = "$(Split-Path $PSScriptRoot -Leaf)",
    [string]$datasinkPath = "$demosRoot/$demo/datasink/$persona",

    [string]$cgsClient = "azure-cleanroom-samples-governance-client-$persona",
    [switch]$interactive
)

#https://learn.microsoft.com/en-us/powershell/scripting/learn/experimental-features?view=powershell-7.4#psnativecommanderroractionpreference
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Import-Module $PSScriptRoot/../../scripts/common/common.psm1

if ($cleanroomEndpoint -eq '')
{
    Write-Log Warning `
        "No endpoint details available for cleanroom '$cleanRoomName' at" `
        "'$publicDir/$cleanRoomName.endpoint'."
    return
}

Write-Log OperationStarted `
    "Showing output from cleanroom '$cleanRoomName' (${cleanroomEndpoint}) for '$persona' in" `
    "the '$demo' demo and contract '$contractId'..."

# TODO: Get rid of this filtering logic once the CGS endpoint sends contract ID as
# part of the list output.
$allDocs = (az cleanroom governance document show `
    --governance-client $cgsClient `
    --output json) | ConvertFrom-Json
$candidateDocs = [Ordered]@{}
$index = 1
foreach ($docId in $allDocs)
{
    $doc = (az cleanroom governance document show `
        --id $docId.id `
        --governance-client $cgsClient `
        --output json) | ConvertFrom-Json
    if ($doc.contractId -eq $contractId)
    {
        $candidateDocs.Add("$index", $doc)
        $index++
    }
    else
    {
        Write-Log Verbose `
            "Skipping document '$($docId.id)' associated with contract '$($doc.contractId)'"
    }
}

if ($candidateDocs.Count -eq 0)
{
    Write-Log Warning `
        "No output available for persona '$persona' in demo '$demo' and contract '$contractId'."
    return
}

if ($false -eq $interactive)
{
    foreach ($doc in $candidateDocs.GetEnumerator())
    {
        $query = $doc.Value.id
        Write-Log OperationStarted `
            "Executing query '$query'..."
        (curl -k https://${cleanroomEndpoint}:8310/app/run_query/$query) | jq
    }
}
else {
    $message = "Select query to execute ('1'-'$($index-1)'/'q')"
    do 
    {
        foreach ($doc in $candidateDocs.GetEnumerator())
        {
            Write-Log OperationCompleted `
                "[$($doc.Key)]: $($doc.Value.id)"
        }
    
        $choice = Read-Host "$($PSStyle.Bold)$message$($PSStyle.Reset)"
        $choice = $choice.ToLower()
        switch ($choice) {
            'q' {
                $response = 'q'
                break
            }
            default {
                try
                {
                    $queryId = [convert]::ToInt32($choice)
                    if ((0 -lt $queryId) -and ($queryId -lt $index))
                    {
                        $query = $candidateDocs["$queryId"].id
                        Write-Log OperationStarted `
                            "Executing query {$query}..."
                        curl -k https://${cleanroomEndpoint}:8310/app/run_query/$query | jq
                        break
                    }
                }
                catch
                {
                }
    
                Write-Log Error `
                    "Invalid input. Please enter ('1'-'$($index-1)'/'q')."
            }
        }
    } while ($response -ne 'q')
}

Write-Log OperationCompleted `
    "Completed showing output from cleanroom '$cleanRoomName' (${cleanroomEndpoint})" `
    "for '$persona' in the '$demo' demo and contract '$contractId'."
