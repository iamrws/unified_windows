param(
    [string]$Endpoint = "tcp://127.0.0.1:8765",
    [int]$Iterations = 200,
    [string]$ReportsDir = "reports/runlogs",
    [string]$RunId = "",
    [string]$WorkloadId = "custom",
    [string]$ModelName = "demo-model",
    [string]$TensorName = "kv",
    [int64]$TensorBytes = 1024,
    [string]$StepName = "decode",
    [switch]$SkipUnitTests,
    [switch]$UseExistingService,
    [int]$GpuSampleSeconds = 1,
    [bool]$RotateCallerPid = $true,
    [int]$BaseCallerPid = 1000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$script:PythonExe = (Get-Command python -ErrorAction Stop).Source

function Write-Log {
    param([string]$Message)
    $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    Write-Host "[$stamp] $Message"
}

function Parse-TcpEndpoint {
    param([string]$Value)
    if ($Value -notmatch '^tcp://([^:]+):(\d+)$') {
        throw "Endpoint must match tcp://<host>:<port>. Received: $Value"
    }
    return @{
        Host = $Matches[1]
        Port = [int]$Matches[2]
    }
}

function Wait-ForListener {
    param(
        [string]$ListenHost,
        [int]$Port,
        [int]$TimeoutSeconds = 20
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        if ($listeners) {
            if ($ListenHost -eq "localhost") {
                return $true
            }
            $match = $listeners | Where-Object { $_.LocalAddress -eq $ListenHost }
            if ($match) {
                return $true
            }
        }
        Start-Sleep -Milliseconds 250
    }
    return $false
}

function Wait-ForServiceReady {
    param(
        [string]$EndpointValue,
        [int]$TimeoutSeconds = 20
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $probe = Invoke-CliJson -EndpointValue $EndpointValue -CliArgs @("create-session") -CallerPid 1
            if ($probe.ok -and $probe.result.session_id) {
                try {
                    Invoke-Step -EndpointValue $EndpointValue -CliArgs @("close-session", $probe.result.session_id) -StepName "ProbeCloseSession" -CallerPid 1 | Out-Null
                } catch {
                }
                return $true
            }
        } catch {
        }
        Start-Sleep -Milliseconds 300
    }
    return $false
}

function Invoke-CliJson {
    param(
        [string]$EndpointValue,
        [string[]]$CliArgs,
        [int]$CallerPid = 1
    )
    $allArgs = @("-m", "astrawave.cli", "--endpoint", $EndpointValue, "--caller-pid", "$CallerPid") + $CliArgs
    $rawItems = & $script:PythonExe @allArgs 2>&1
    $raw = (($rawItems | ForEach-Object { "$_" }) -join "`n").Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "CLI command failed: $script:PythonExe $($allArgs -join ' '); output: $raw"
    }
    try {
        return ($raw | ConvertFrom-Json)
    } catch {
        throw "CLI returned non-JSON output: $raw"
    }
}

function Invoke-Step {
    param(
        [string]$EndpointValue,
        [string[]]$CliArgs,
        [string]$StepName,
        [int]$CallerPid = 1
    )
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $result = Invoke-CliJson -EndpointValue $EndpointValue -CliArgs $CliArgs -CallerPid $CallerPid
    $sw.Stop()
    if (-not $result.ok) {
        $errCode = $result.error.code
        $errMsg = $result.error.message
        throw "$StepName failed: [$errCode] $errMsg"
    }
    return @{
        Response = $result
        DurationMs = [math]::Round($sw.Elapsed.TotalMilliseconds, 3)
    }
}

function Get-Percentile {
    param(
        [double[]]$Values,
        [double]$Percentile
    )
    if (-not $Values -or $Values.Count -eq 0) {
        return $null
    }
    $sorted = $Values | Sort-Object
    $index = [math]::Ceiling(($Percentile / 100.0) * $sorted.Count) - 1
    if ($index -lt 0) { $index = 0 }
    if ($index -ge $sorted.Count) { $index = $sorted.Count - 1 }
    return [math]::Round([double]$sorted[$index], 3)
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
Set-Location $repoRoot

if (-not $RunId) {
    $RunId = (Get-Date).ToString("yyyy-MM-dd_HHmmss")
}

$runDir = Join-Path $repoRoot $ReportsDir
New-Item -ItemType Directory -Path $runDir -Force | Out-Null

$unitLog = Join-Path $runDir "unittest_$RunId.txt"
$soakJsonl = Join-Path $runDir "soak_$RunId.jsonl"
$summaryJson = Join-Path $runDir "summary_$RunId.json"
$serviceOut = Join-Path $runDir "service_stdout_$RunId.log"
$serviceErr = Join-Path $runDir "service_stderr_$RunId.log"
$gpuLog = Join-Path $runDir "gpu_$RunId.csv"

$endpointParts = Parse-TcpEndpoint -Value $Endpoint
$endpointHost = $endpointParts.Host
$port = $endpointParts.Port

$serviceProcess = $null
$gpuProcess = $null
$unitExitCode = $null

try {
    Write-Log "Run id: $RunId"
    Write-Log "Repo root: $repoRoot"
    Write-Log "Endpoint: $Endpoint"
    Write-Log "Iterations: $Iterations"
    Write-Log "Rotate caller pid: $RotateCallerPid"
    Write-Log "Workload: $WorkloadId"
    Write-Log "Model: $ModelName"
    Write-Log "Tensor: $TensorName ($TensorBytes bytes)"
    Write-Log "Step: $StepName"

    if ($TensorBytes -le 0) {
        throw "TensorBytes must be positive. Received: $TensorBytes"
    }
    if (-not $TensorName) {
        throw "TensorName must not be empty."
    }
    if (-not $ModelName) {
        throw "ModelName must not be empty."
    }
    if (-not $StepName) {
        throw "StepName must not be empty."
    }

    if (-not $SkipUnitTests) {
        Write-Log "Running unit test suite..."
        $unitCmd = "`"$script:PythonExe`" -m unittest discover -s tests -v > `"$unitLog`" 2>&1"
        cmd /c $unitCmd | Out-Null
        $unitExitCode = $LASTEXITCODE
        Get-Content -Path $unitLog | Out-Host
        if ($unitExitCode -ne 0) {
            throw "Unit tests failed with exit code $unitExitCode. See $unitLog"
        }
    } else {
        Write-Log "Skipping unit tests (--SkipUnitTests set)."
    }

    if ($UseExistingService) {
        Write-Log "Validating existing service on $Endpoint..."
        if (-not (Wait-ForServiceReady -EndpointValue $Endpoint -TimeoutSeconds 20)) {
            throw "No responsive service found for -UseExistingService at $Endpoint"
        }
    }

    if (-not $UseExistingService) {
        Write-Log "Starting AstraWeave service..."
        $serviceArgs = @("-m", "astrawave.cli", "serve", "--transport", "tcp", "--endpoint", $Endpoint, "--duration-seconds", "86400")
        $serviceProcess = Start-Process -FilePath $script:PythonExe -ArgumentList $serviceArgs -WorkingDirectory $repoRoot -PassThru -NoNewWindow -RedirectStandardOutput $serviceOut -RedirectStandardError $serviceErr
        Start-Sleep -Milliseconds 300
        if ($serviceProcess.HasExited) {
            $exitCode = $serviceProcess.ExitCode
            throw "Service process exited early with code $exitCode. See $serviceOut / $serviceErr"
        }
        if (-not (Wait-ForServiceReady -EndpointValue $Endpoint -TimeoutSeconds 20)) {
            throw "Service did not become API-ready at $Endpoint within timeout. See $serviceOut / $serviceErr"
        }
    } else {
        Write-Log "Using existing service on $Endpoint"
    }

    $activeListeners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($listener in $activeListeners) {
        if ($listener.LocalAddress -notin @("127.0.0.1", "::1", "localhost")) {
            throw "Non-loopback listener detected on port ${port}: $($listener.LocalAddress)"
        }
    }

    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        Write-Log "Starting GPU telemetry capture -> $gpuLog"
        $gpuArgs = @(
            "--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv",
            "-l", "$GpuSampleSeconds",
            "-f", $gpuLog
        )
        $gpuProcess = Start-Process -FilePath "nvidia-smi" -ArgumentList $gpuArgs -PassThru -WindowStyle Hidden
    } else {
        Write-Log "nvidia-smi not found; skipping GPU telemetry capture."
    }

    Write-Log "Running soak loop..."
    $startedAt = Get-Date
    $successCount = 0
    $failureCount = 0
    $runStepDurations = New-Object System.Collections.Generic.List[double]
    $firstFailure = $null

    for ($i = 1; $i -le $Iterations; $i++) {
        $record = [ordered]@{
            iteration = $i
            started_at = (Get-Date).ToString("o")
            status = "unknown"
        }

        try {
            $callerPid = if ($RotateCallerPid) { $BaseCallerPid + $i } else { 1 }
            $record.caller_pid = $callerPid

            $create = Invoke-Step -EndpointValue $Endpoint -CliArgs @("create-session") -StepName "CreateSession" -CallerPid $callerPid
            $sessionId = $create.Response.result.session_id
            if (-not $sessionId) {
                throw "CreateSession returned empty session id"
            }

            $record.session_id = $sessionId
            $record.create_ms = $create.DurationMs

            $load = Invoke-Step -EndpointValue $Endpoint -CliArgs @("load-model", $sessionId, $ModelName) -StepName "LoadModel" -CallerPid $callerPid
            $register = Invoke-Step -EndpointValue $Endpoint -CliArgs @("register-tensor", $sessionId, $TensorName, "$TensorBytes") -StepName "RegisterTensor" -CallerPid $callerPid
            $run = Invoke-Step -EndpointValue $Endpoint -CliArgs @("run-step", $sessionId, "--step-name", $StepName) -StepName "RunStep" -CallerPid $callerPid
            $close = Invoke-Step -EndpointValue $Endpoint -CliArgs @("close-session", $sessionId) -StepName "CloseSession" -CallerPid $callerPid

            $record.load_ms = $load.DurationMs
            $record.register_ms = $register.DurationMs
            $record.run_step_ms = $run.DurationMs
            $record.run_state = $run.Response.result.state
            $record.run_pressure_level = $run.Response.result.pressure_level
            $record.run_correlation_id = $run.Response.result.correlation_id
            $record.run_fallback_step = $run.Response.result.fallback_step
            $record.run_fallback_reason_code = if ($run.Response.result.fallback_result) { $run.Response.result.fallback_result.reason_code } else { $null }
            $record.close_ms = $close.DurationMs
            $record.status = "ok"

            $runStepDurations.Add([double]$run.DurationMs) | Out-Null
            $successCount++
        } catch {
            $failureCount++
            $record.status = "error"
            $record.error = $_.Exception.Message
            if (-not $firstFailure) {
                $firstFailure = $_.Exception.Message
            }
        } finally {
            $record.ended_at = (Get-Date).ToString("o")
            ($record | ConvertTo-Json -Compress -Depth 6) | Add-Content -Path $soakJsonl -Encoding utf8
        }

        if (($i % 25) -eq 0 -or $i -eq $Iterations) {
            Write-Log "Progress: $i / $Iterations (ok=$successCount, fail=$failureCount)"
        }
    }

    $endedAt = Get-Date
    $elapsed = [math]::Round(($endedAt - $startedAt).TotalSeconds, 3)

    $runStepP50 = Get-Percentile -Values ($runStepDurations.ToArray()) -Percentile 50
    $runStepP95 = Get-Percentile -Values ($runStepDurations.ToArray()) -Percentile 95

    $summary = [ordered]@{
        run_id = $RunId
        started_at = $startedAt.ToString("o")
        ended_at = $endedAt.ToString("o")
        elapsed_seconds = $elapsed
        endpoint = $Endpoint
        workload_id = $WorkloadId
        model_name = $ModelName
        tensor_name = $TensorName
        tensor_bytes = $TensorBytes
        step_name = $StepName
        iterations = $Iterations
        success_count = $successCount
        failure_count = $failureCount
        success_rate = if ($Iterations -gt 0) { [math]::Round(($successCount * 100.0) / $Iterations, 3) } else { 0.0 }
        run_step_p50_ms = $runStepP50
        run_step_p95_ms = $runStepP95
        unit_tests_skipped = [bool]$SkipUnitTests
        unit_test_exit_code = $unitExitCode
        files = @{
            unit_test_log = if ($SkipUnitTests) { $null } else { $unitLog }
            soak_log = $soakJsonl
            summary_log = $summaryJson
            gpu_log = if (Test-Path $gpuLog) { $gpuLog } else { $null }
            service_stdout = if (Test-Path $serviceOut) { $serviceOut } else { $null }
            service_stderr = if (Test-Path $serviceErr) { $serviceErr } else { $null }
        }
        first_failure = $firstFailure
        verdict = if ($failureCount -eq 0) { "pass" } else { "fail" }
    }

    $summary | ConvertTo-Json -Depth 8 | Set-Content -Path $summaryJson -Encoding utf8

    Write-Host ""
    Write-Host "========== Soak Summary =========="
    Write-Host "Run ID:          $($summary.run_id)"
    Write-Host "Iterations:      $($summary.iterations)"
    Write-Host "Success:         $($summary.success_count)"
    Write-Host "Failures:        $($summary.failure_count)"
    Write-Host "Success Rate:    $($summary.success_rate)%"
    Write-Host "RunStep p50/p95: $($summary.run_step_p50_ms) ms / $($summary.run_step_p95_ms) ms"
    Write-Host "Elapsed:         $($summary.elapsed_seconds) s"
    Write-Host "Summary:         $summaryJson"
    Write-Host "Soak Log:        $soakJsonl"
    if (Test-Path $gpuLog) {
        Write-Host "GPU Log:         $gpuLog"
    }
    Write-Host "Verdict:         $($summary.verdict)"
    Write-Host "=================================="

    if ($failureCount -gt 0) {
        exit 1
    }
    exit 0
}
finally {
    if ($gpuProcess -and -not $gpuProcess.HasExited) {
        try {
            Stop-Process -Id $gpuProcess.Id -Force -ErrorAction SilentlyContinue
        } catch {
        }
    }

    if ($serviceProcess -and -not $serviceProcess.HasExited) {
        try {
            Stop-Process -Id $serviceProcess.Id -Force -ErrorAction SilentlyContinue
        } catch {
        }
    }
}
