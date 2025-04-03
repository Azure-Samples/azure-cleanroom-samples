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
    [string]$datastoreConfig = "$privateDir/datastores.config",

    [string]$demo = "$(Split-Path $PSScriptRoot -Leaf)",
    [string]$datasinkPath = "$demosRoot/$demo/datasink/$persona",

    [string]$cgsClient = "azure-cleanroom-samples-governance-client-$persona",
    [switch]$extended
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

function Get-Sentiment {
    param (
        [string]$review,
        [string]$expectedRating
    )

    $payload = @{
        data = "$review"
    } | ConvertTo-Json

    Write-Log Verbose `
        "Checking sentiment..."
    $response = (curl -s -k --fail-with-body `
        -X POST -H "Content-Type: application/json" -d $payload `
        https://${cleanroomEndpoint}:8000/infer)
    $output = @{
        SubmittedReview = "$review"
        InferredRating = ("1" -eq $response) ? "Good" : "Bad"
    }

    if ($expectedRating -eq $output.InferredRating)
    {
        Write-Log Information `
            "$($output | ConvertTo-Json)"
    }
    else
    {
        Write-Log Warning `
            "$($output | ConvertTo-Json)"
    }
}

#
# Review snippets of good movies from IMDB - expected sentiment is positive.
#
$shawshank =
    "is a poignant story about hope. Hope gets me. That's what makes a film like this more " +
    "than a movie. It tells a lesson about life. Those are the films people talk about 50 " +
    "or even 100 years from you. It's also a story for freedom. Freedom from isolation, from " +
    "rule, from bigotry and hate."
Get-Sentiment -review $shawshank -expectedRating "Good"

$godfather =
    "is quite simply a masterful piece of film-making, an epic in the truest sense of the word " +
    "and by far the finest gangster film ever shot. Made with finesse, style to spare and..."
Get-Sentiment -review $godfather -expectedRating "Good"

#
# Review snippets of bad movies from IMDB - expected sentiment is negative.
#
$battlefield =
    "It was with a certain morbid curiosity and a near certainty that I would be seeing an " +
    "awful movie that I rented it, and I must say the movie exceeded all of my expectations, " +
    "it is indeed exceptionally terrible."
Get-Sentiment -review $battlefield -expectedRating "Bad"

#
# Review snippet of bad movie already seen by the model as part of training - expected sentiment
# is negative.
#
$unknown =
    "plays like a glossy melodrama that occasionally verges on camp ."
Get-Sentiment -review $unknown -expectedRating "Bad"

#
# TODO: Uncomment this once the application API has been fixed.
#
# if ("contosso" -eq $persona)
# {
#     if ($extended)
#     {
#         $splits = ("test", "train", "validation")
#     }
#     else
#     {
#         $splits = ("test")
#     }

#     foreach ($split in $splits)
#     {
#         Write-Log Verbose `
#             "Invoking cleanroom to check inference success rate for split '$split'..."
#         curl -k --fail-with-body https://${cleanroomEndpoint}:8000/check/${split} | jq
#     }
# }

Write-Log OperationCompleted `
    "Completed showing output from cleanroom '$cleanRoomName' (${cleanroomEndpoint})" `
    "for '$persona' in the '$demo' demo and contract '$contractId'."
