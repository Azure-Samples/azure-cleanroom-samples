function GetLoggedInEntityObjectId {
    if ($env:GITHUB_ACTIONS -eq "true") {
        Write-Host "Running inside GitHub Actions. Fetching Azure credentials"

        if ($env:TEST_ENVIRONMENT -eq "PR") {
            $clientId = $($env:AZURE_CREDENTIALS | ConvertFrom-Json).clientId
        }
        elseif ($env:TEST_ENVIRONMENT -eq "BVT") {
            $clientId = $env:RELEASE_CLIENT_ID
        }
        else {
            throw "Required environment variables are unavailable"
            $clientId = ""
        }

        $spDetails = (az ad sp show --id $clientId) | ConvertFrom-Json
        $objectId = $spDetails.id

        Write-Host "Fetched object ID $objectId for client ID $clientID"
        return $objectId
    }
    else {
        Write-Host "Fetching object ID of currently logged in user"
        # Since some tenant (including Microsoft tenant) has Conditional Access policies that block
        # accessing Microsoft Graph with device code (#22629), querying Microsoft Graph API is no
        # longer possible with device code.
        # Using manual workaround per https://github.com/Azure/azure-cli/issues/22776
        Write-Host "Extracting object ID from access token"
        Write-Host "$(pip3 install --upgrade pyjwt)"
        $objectId = (az account get-access-token --query accessToken --output tsv | `
                tr -d '\n' | `
                python3 -c "import jwt, sys; print(jwt.decode(sys.stdin.read(), algorithms=['RS256'], options={'verify_signature': False})['oid'])")
        return $objectId
    }
}