$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskUrl = 'http://127.0.0.1:7864'
try {
    $taskExisting = (Invoke-WebRequest "$taskUrl/config" -UseBasicParsing -TimeoutSec 2).Content
    if ($taskExisting -match '"title"\s*:\s*"LocalTranscribe - VibeVoice-ASR"') { Start-Process $taskUrl; exit 0 }
    throw 'Port 7864 is used by another application.'
} catch {
    if ($_.Exception.Message -eq 'Port 7864 is used by another application.') { throw }
}
New-Item -ItemType Directory -Force -Path (Join-Path $taskRoot 'logs') | Out-Null
$env:PYTHONUTF8 = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$taskProcess = Start-Process -FilePath (Join-Path $taskRoot 'runtime\Scripts\python.exe') -ArgumentList @('-u', ('"' + (Join-Path $taskRoot 'app.py') + '"')) -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRoot 'logs\server.log') -RedirectStandardError (Join-Path $taskRoot 'logs\server-error.log')
$taskProcess.Id | Set-Content -LiteralPath (Join-Path $taskRoot 'server.pid')
for ($taskAttempt = 0; $taskAttempt -lt 90; $taskAttempt++) {
    Start-Sleep -Seconds 1
    if ($taskProcess.HasExited) { throw 'Server stopped. See logs/server-error.log.' }
    try { $taskReady = (Invoke-WebRequest "$taskUrl/config" -UseBasicParsing -TimeoutSec 1).Content; if ($taskReady -match '"title"\s*:\s*"LocalTranscribe - VibeVoice-ASR"') { Start-Process $taskUrl; exit 0 } } catch {}
}
throw 'Startup took longer than expected. Check logs/server-error.log.'
