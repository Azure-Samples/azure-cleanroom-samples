param(
    [Parameter(Mandatory = $true)]
    [string]$KubeConfigPath,
    [Parameter(Mandatory = $true)]
    [string]$GrafanaUrl,
    [Parameter(Mandatory = $true)]
    [string]$UserName,
    [Parameter(Mandatory = $true)]
    [SecureString]$UserPassword,
    [ValidateSet("Viewer", "Editor")]
    [string]$UserRole = "Viewer"
)

$GrafanaSecretName = "cleanroom-spark-grafana"
$GrafanaNamespace = "telemetry"
$AdminUserNameKey = "admin-user"
$AdminPasswordKey = "admin-password"

$TimeoutSeconds = 60
$PollIntervalSeconds = 5

Write-Host "Waiting for Grafana at $GrafanaUrl to become healthy..."

$healthUrl = "$GrafanaUrl/api/health"
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)

while ((Get-Date) -lt $deadline) {
    try {
        $resp = Invoke-WebRequest -Uri $healthUrl -Method GET -UseBasicParsing -TimeoutSec 10
        if ($resp.StatusCode -eq 200) {
            Write-Host "Grafana is healthy."
            break
        }
    }
    catch {
        # Ignore and retry
    }
    Write-Host "Grafana not ready yet, retrying in $PollIntervalSeconds seconds..."
    Start-Sleep -Seconds $PollIntervalSeconds
}

if ((Get-Date) -ge $deadline) {
    throw "Timed out waiting for Grafana to be healthy at $GrafanaUrl"
}

# Convert SecureString password to plain text for use in API calls
$BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($UserPassword)
$PlainPassword = [System.Runtime.InteropServices.Marshal]::PtrToStringBSTR($BSTR)
[System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($BSTR)

# Read admin credentials from Kubernetes Secret
Write-Host "Reading admin credentials from secret '$GrafanaSecretName' in namespace '$GrafanaNamespace'..."

$secretJson = kubectl --kubeconfig=$KubeConfigPath get secret $GrafanaSecretName -n $GrafanaNamespace -o json 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Failed to get secret '$GrafanaSecretName' in namespace '$GrafanaNamespace'."
}

$secretData = $secretJson | ConvertFrom-Json | Select-Object -ExpandProperty data

$adminPasswordBase64 = $secretData.$AdminPasswordKey
$adminPassword = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($adminPasswordBase64))
$adminUserNameBase64 = $secretData.$AdminUserNameKey
$adminUserName = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($adminUserNameBase64))
$basicAuth = [System.Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes("${adminUserName}:${adminPassword}"))
$headers = @{ Authorization = "Basic $basicAuth" }
$headers["Content-Type"] = "application/json"

$searchUserUrl = "$GrafanaUrl/api/users/lookup?loginOrEmail=$UserName"
Write-Host "Checking if user '$UserName' already exists..."

try {
    $existingUser = Invoke-RestMethod -Uri $searchUserUrl `
        -Method GET `
        -Headers $headers `
        -ErrorAction Stop
    
    if ($existingUser -and $existingUser.id) {
        throw "User '$UserName' already exists with id $($existingUser.id)."
    }
}
catch {
    # If 404, user doesn't exist - continue
    if ($_.Exception.Response.StatusCode.value__ -ne 404) {
        # Re-throw if it's not a 404 error
        throw
    }
    Write-Host "User '$UserName' does not exist. Proceeding with creation..."
}

$createUserUrl = "$GrafanaUrl/api/admin/users"
Write-Host "Creating Grafana user '$UserName'..."
$createBody = @{
    name     = $UserName
    login    = $UserName
    password = $PlainPassword
} | ConvertTo-Json

Write-Host "Creating user '$UserName'..."

$createResp = Invoke-WebRequest -Uri $createUserUrl `
    -Method POST `
    -Headers $headers `
    -Body $createBody
$response = $createResp.Content | ConvertFrom-Json
$userId = $response.id
if (-not $userId) {
    throw "Failed to get user id from create response: $($response | ConvertTo-Json -Depth 5)"
}

Write-Host "User created with id = $userId"

# 3. Assign org role (Viewer/Editor/Admin) in org $OrgId
$assignRoleUrl = "$GrafanaUrl/api/orgs/1/users/$userId"
$roleBody = @{ role = $UserRole } | ConvertTo-Json

Write-Host "Assigning role '$UserRole' to user id $userId"

Invoke-RestMethod -Uri $assignRoleUrl `
    -Method PATCH `
    -Headers $headers `
    -Body $roleBody

Write-Host "Successfully assigned role '$UserRole' to user '$UserName'."

$UserSecretName = "grafana-user-$UserName"

Write-Host "Creating Kubernetes secret '$UserSecretName' in namespace '$GrafanaNamespace'..."

kubectl --kubeconfig=$KubeConfigPath get secret $UserSecretName -n $GrafanaNamespace -o json 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Secret '$UserSecretName' already exists. Deleting it first..."
    kubectl --kubeconfig=$KubeConfigPath delete secret $UserSecretName -n $GrafanaNamespace
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Failed to delete existing secret, but continuing..."
    }
}

# Create the secret
kubectl --kubeconfig=$KubeConfigPath create secret generic $UserSecretName `
    -n $GrafanaNamespace `
    --from-literal=username=$UserName `
    --from-literal=password=$PlainPassword

if ($LASTEXITCODE -eq 0) {
    Write-Host "Successfully created secret '$UserSecretName' with user credentials."
}
else {
    Write-Warning "Failed to create secret '$UserSecretName'. User was created in Grafana but secret creation failed."
}
