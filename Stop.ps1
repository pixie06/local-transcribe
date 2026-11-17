$taskPidFile = Join-Path $PSScriptRoot 'server.pid'
if (Test-Path -LiteralPath $taskPidFile) {
    $taskServerId = [int](Get-Content -LiteralPath $taskPidFile)
    $taskServer = Get-CimInstance Win32_Process -Filter "ProcessId = $taskServerId"
    $taskAppPath = Join-Path $PSScriptRoot 'app.py'
    if ($taskServer -and $taskServer.CommandLine.Contains($taskAppPath)) {
        $taskChildren = Get-CimInstance Win32_Process -Filter "ParentProcessId = $taskServerId"
        foreach ($taskChild in $taskChildren) {
            if ($taskChild.CommandLine -and $taskChild.CommandLine.Contains($taskAppPath)) { Stop-Process -Id $taskChild.ProcessId -ErrorAction SilentlyContinue }
        }
        Stop-Process -Id $taskServerId -ErrorAction SilentlyContinue
    }
}
