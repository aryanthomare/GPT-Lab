<#
.SYNOPSIS
Start the GPT-Lab UI in WSL (unless it's already running) and open it in the browser.

.DESCRIPTION
The server runs under a hidden wsl.exe process. That process also keeps WSL itself
running: once no wsl.exe is attached, WSL shuts the distribution down within seconds,
which would stop any training run. Stop the server from the UI (Settings > Shut down
server) when you no longer need it.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File Start-GPTLabUI.ps1
powershell -ExecutionPolicy Bypass -File Start-GPTLabUI.ps1 -Distro Ubuntu -RepoDir GPT-Lab -Port 8000
#>
param(
    [string]$Distro = "Ubuntu",
    [string]$RepoDir = "GPT-Lab",  # relative to the Linux home directory
    [int]$Port = 8000,
    [switch]$NoBrowser             # start the server only
)

$url = "http://localhost:$Port"

function Test-GPTLab {
    # 127.0.0.1, not localhost: Windows PowerShell tries IPv6 (::1) first and only falls
    # back to IPv4 after about two seconds, while the server listens on IPv4.
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 5
        return [bool]$health.ok
    } catch {
        return $false
    }
}

if (-not (Test-GPTLab)) {
    $wslArgs = @("-d", $Distro, "--cd", "~", "--exec", "bash", "$RepoDir/scripts/ui.sh", "--port", "$Port")
    Start-Process -FilePath "wsl.exe" -ArgumentList $wslArgs -WindowStyle Hidden
    $deadline = (Get-Date).AddSeconds(90)
    while (-not (Test-GPTLab)) {
        if ((Get-Date) -gt $deadline) {
            Add-Type -AssemblyName PresentationFramework
            [System.Windows.MessageBox]::Show(
                "The GPT-Lab UI didn't start within 90 seconds.`n`nIts log is ~/$RepoDir/runs/.ui-server.log in $Distro.",
                "GPT-Lab") | Out-Null
            exit 1
        }
        Start-Sleep -Milliseconds 500
    }
}
if (-not $NoBrowser) {
    Start-Process $url
}
