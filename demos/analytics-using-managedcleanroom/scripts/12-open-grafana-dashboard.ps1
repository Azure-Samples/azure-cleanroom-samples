param(
    [Parameter(Mandatory = $true)]
    [string]$KubeConfigPath,
    [int]$LocalPort = 3000
)

$GrafanaSecretName = "cleanroom-spark-grafana"
$GrafanaNamespace = "telemetry"
$GrafanaService = "cleanroom-spark-grafana"

# Get admin credentials
Write-Host "Reading Grafana admin credentials..."
$encoded = kubectl --kubeconfig $KubeConfigPath `
    get secret $GrafanaSecretName -n $GrafanaNamespace `
    -o jsonpath="{.data.admin-password}"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to read Grafana admin secret '$GrafanaSecretName' in namespace '$GrafanaNamespace'."
}
$password = [System.Text.Encoding]::UTF8.GetString(
    [Convert]::FromBase64String($encoded))
Write-Host "Username: admin"
Write-Host "Password: $password"

# Print URL for user to open
$url = "http://localhost:$LocalPort"
Write-Host "`nGrafana will be available at: $url"

# Port-forward (blocks until Ctrl+C)
Write-Host "`nPort-forwarding $GrafanaService to localhost:$LocalPort (Ctrl+C to stop)..."
kubectl --kubeconfig $KubeConfigPath `
    port-forward svc/$GrafanaService ${LocalPort}:80 -n $GrafanaNamespace
