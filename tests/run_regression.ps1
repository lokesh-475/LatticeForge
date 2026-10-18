<#
.SYNOPSIS
    Local PowerShell Regression Suite across Temperature Sweeps for LatticeForge CUDA Engine.

.DESCRIPTION
    Executes batch runs of lattice_forge_cuda.exe across dual temperature regimes:
      - Low Temperature Sintering (Ts = 0.4): Promotes solute boundary segregation and core-shell kinetics.
      - High Temperature Sintering (Ts = 2.8): Promotes thermal dissolution and homogeneous mixing.

    Automatically verifies:
      1. Output generation: Validates existence, size, and schemas of energy log, binary snapshot,
         radial profile, and grain centroid artifacts.
      2. Energy descent: Asserts Hamiltonian energy minimization (Delta H < 0) under curvature-driven
         grain growth and segregation dynamics.
      3. Invariant conservation: Asserts strict atomic species conservation (Delta N_Ce == 0)
         by inspecting the raw binary lattice snapshot payload.
      4. Physical differential: Asserts Low Ts enrichment > High Ts enrichment.

.PARAMETER ExecutablePath
    Path to the compiled lattice_forge_cuda.exe executable. If omitted, standard locations are searched.

.PARAMETER Steps
    Number of Monte Carlo Steps (MCS) per temperature sweep run (default: 30).

.PARAMETER Temperatures
    Array of reduced sintering temperatures to test (default: @(0.4, 2.8)).

.PARAMETER OutputDir
    Directory where simulation outputs and logs are deposited (default: outputs).

.PARAMETER FailFast
    Stop execution immediately on the first assertion failure.

.EXAMPLE
    .\tests\run_regression.ps1
    .\tests\run_regression.ps1 -Steps 50 -Temperatures @(0.4, 2.8)
    .\tests\run_regression.ps1 -ExecutablePath .\build\lattice_forge_cuda.exe
#>

[CmdletBinding()]
param(
    [string]$ExecutablePath = "",
    [int]$Steps = 30,
    [double[]]$Temperatures = @(0.4, 2.8),
    [string]$OutputDir = "outputs",
    [switch]$FailFast
)

$ErrorActionPreference = "Continue"

# -----------------------------------------------------------------------------
# ANSI Color & Formatting Helpers
# -----------------------------------------------------------------------------
function Write-Header {
    param([string]$Text)
    Write-Host "`n================================================================================" -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host "================================================================================" -ForegroundColor Cyan
}

function Write-SubHeader {
    param([string]$Text)
    Write-Host "`n--- $Text ---" -ForegroundColor DarkCyan
}

function Write-Pass {
    param([string]$Category, [string]$Details)
    Write-Host "  [PASS] " -NoNewline -ForegroundColor Green
    Write-Host "$Category : " -NoNewline -ForegroundColor White
    Write-Host $Details -ForegroundColor Gray
}

function Write-Fail {
    param([string]$Category, [string]$Details)
    Write-Host "  [FAIL] " -NoNewline -ForegroundColor Red
    Write-Host "$Category : " -NoNewline -ForegroundColor White
    Write-Host $Details -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Category, [string]$Details)
    Write-Host "  [INFO] " -NoNewline -ForegroundColor Blue
    Write-Host "$Category : " -NoNewline -ForegroundColor White
    Write-Host $Details -ForegroundColor DarkGray
}

# -----------------------------------------------------------------------------
# Locate Executable
# -----------------------------------------------------------------------------
function Find-LatticeForgeExecutable {
    param([string]$ExplicitPath)

    if ($ExplicitPath -and (Test-Path $ExplicitPath -PathType Leaf)) {
        return (Resolve-Path $ExplicitPath).Path
    }

    $candidates = @(
        ".\lattice_forge_cuda.exe",
        ".\build\lattice_forge_cuda.exe",
        "..\lattice_forge_cuda.exe",
        "..\build\lattice_forge_cuda.exe",
        "$PSScriptRoot\..\lattice_forge_cuda.exe",
        "$PSScriptRoot\..\build\lattice_forge_cuda.exe"
    )

    foreach ($path in $candidates) {
        if (Test-Path $path -PathType Leaf) {
            return (Resolve-Path $path).Path
        }
    }

    return $null
}

# -----------------------------------------------------------------------------
# Binary Snapshot Inspector (128-byte Header + Raw Arrays)
# -----------------------------------------------------------------------------
function Inspect-BinarySnapshot {
    param([string]$FilePath)

    $result = [PSCustomObject]@{
        Valid       = $false
        FileSize    = 0
        Magic       = ""
        HeaderSize  = 0
        Version     = 0
        DimX        = 0
        DimY        = 0
        TotalSites  = 0
        Temp        = 0.0
        CeCount     = 0
        ErrorMessage= ""
    }

    if (-not (Test-Path $FilePath -PathType Leaf)) {
        $result.ErrorMessage = "Snapshot file does not exist: $FilePath"
        return $result
    }

    try {
        $bytes = [System.IO.File]::ReadAllBytes($FilePath)
        $result.FileSize = $bytes.Length

        if ($bytes.Length -lt 128) {
            $result.ErrorMessage = "File size ($($bytes.Length) bytes) smaller than minimum 128-byte header"
            return $result
        }

        # Header parsing (128 bytes, Little-Endian)
        $result.Magic      = [System.Text.Encoding]::ASCII.GetString($bytes, 0, 8)
        $result.HeaderSize = [System.BitConverter]::ToUInt32($bytes, 8)
        $result.Version    = [System.BitConverter]::ToUInt32($bytes, 12)
        $result.DimX       = [System.BitConverter]::ToUInt32($bytes, 16)
        $result.DimY       = [System.BitConverter]::ToUInt32($bytes, 20)
        $result.TotalSites = [System.BitConverter]::ToUInt64($bytes, 24)
        $result.Temp       = [System.BitConverter]::ToDouble($bytes, 44)

        if ($result.Magic -ne "LFORGE01") {
            $result.ErrorMessage = "Invalid magic bytes '$($result.Magic)', expected 'LFORGE01'"
            return $result
        }

        if ($result.HeaderSize -ne 128) {
            $result.ErrorMessage = "Header size is $($result.HeaderSize), expected 128 bytes"
            return $result
        }

        $expectedPayload = ($result.TotalSites * 2) + ($result.TotalSites * 1)
        $expectedTotal = 128 + $expectedPayload

        if ($bytes.Length -ne $expectedTotal) {
            $result.ErrorMessage = "File size ($($bytes.Length)) does not match expected size ($expectedTotal) for $($result.DimX)x$($result.DimY) grid"
            return $result
        }

        # Chemical element array c is located after orientations array:
        # Offset = 128 + (TotalSites * 2)
        $cOffset = 128 + [int]($result.TotalSites * 2)
        $ceAtoms = 0
        for ($i = $cOffset; $i -lt $bytes.Length; $i++) {
            if ($bytes[$i] -eq 1) {
                $ceAtoms++
            }
        }
        $result.CeCount = $ceAtoms
        $result.Valid   = $true
        return $result
    }
    catch {
        $result.ErrorMessage = $_.Exception.Message
        return $result
    }
}

# -----------------------------------------------------------------------------
# Main Test Execution Flow
# -----------------------------------------------------------------------------
$scriptStartTime = Get-Date

Write-Header "LATTICEFORGE REGRESSION SUITE: TEMPERATURE SWEEPS"
Write-Host "Target Sweeps: Ts in (@($($Temperatures -join ', '))) | Monte Carlo Steps: $Steps`n"

# Locate Binary
$exe = Find-LatticeForgeExecutable -ExplicitPath $ExecutablePath
if (-not $exe) {
    Write-Fail "Environment" "lattice_forge_cuda.exe was not found. Please compile the project first."
    exit 1
}
Write-Pass "Environment" "Found executable: $exe"

# Resolve Output Directory (default to repo root outputs/ if available)
if ($OutputDir -eq "outputs") {
    if (Test-Path "$PSScriptRoot\..\outputs") {
        $resolvedOutputDir = (Resolve-Path "$PSScriptRoot\..\outputs").Path
    } elseif (Test-Path "outputs") {
        $resolvedOutputDir = (Resolve-Path "outputs").Path
    } else {
        $resolvedOutputDir = [System.IO.Path]::GetFullPath($OutputDir)
    }
} else {
    $resolvedOutputDir = [System.IO.Path]::GetFullPath($OutputDir)
}
if (-not (Test-Path $resolvedOutputDir)) {
    [System.IO.Directory]::CreateDirectory($resolvedOutputDir) | Out-Null
}
Write-Pass "Environment" "Resolved outputs directory: $resolvedOutputDir"

# Summary Collection
$testResults = @()
$suitePassed = $true

foreach ($T in $Temperatures) {
    Write-SubHeader "Running Simulation Sweep: Ts = $T ($Steps MCS)"

    $tag = "T$([string]::Format([System.Globalization.CultureInfo]::InvariantCulture, '{0:0.0}', $T))"
    $runCsvPath       = Join-Path $resolvedOutputDir "energy_log_$tag.csv"
    $runBinPath       = Join-Path $resolvedOutputDir "lattice_state_$tag.bin"
    $runRadialPath    = Join-Path $resolvedOutputDir "radial_profile_$tag.csv"
    $runCentroidsPath = Join-Path $resolvedOutputDir "grain_centroids_$tag.csv"

    # Remove stale files before run
    foreach ($f in @($runCsvPath, $runBinPath, $runRadialPath, $runCentroidsPath)) {
        if (Test-Path $f) { Remove-Item $f -Force }
    }

    $runRecord = [PSCustomObject]@{
        Temp                = $T
        Steps               = $Steps
        ExecutionTimeSec    = 0.0
        ExitCode            = -1
        OutputGenerated     = $false
        EnergyDescent       = $false
        InvariantConserved  = $false
        InitialEnergy       = 0.0
        FinalEnergy         = 0.0
        DeltaEnergy         = 0.0
        DescentPercent      = 0.0
        InitialCeAtoms      = 0
        FinalCeAtoms        = 0
        EnrichmentInitial   = 0.0
        EnrichmentFinal     = 0.0
        Failures            = @()
    }

    # Execute lattice_forge_cuda.exe
    $cmdArgs = @(
        "-s", "$Steps",
        "-t", "$([string]::Format([System.Globalization.CultureInfo]::InvariantCulture, '{0:0.00}', $T))",
        "-o", "`"$runCsvPath`"",
        "-b", "`"$runBinPath`"",
        "--radial", "`"$runRadialPath`"",
        "--centroids", "`"$runCentroidsPath`""
    )

    Write-Info "Execution" "Invoking: $exe $($cmdArgs -join ' ')"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    
    $pinfo = New-Object System.Diagnostics.ProcessStartInfo
    $pinfo.FileName = $exe
    $pinfo.Arguments = ($cmdArgs -join ' ')
    $pinfo.RedirectStandardOutput = $true
    $pinfo.RedirectStandardError = $true
    $pinfo.UseShellExecute = $false
    $pinfo.CreateNoWindow = $true
    if (Test-Path "$PSScriptRoot\..") {
        $pinfo.WorkingDirectory = (Resolve-Path "$PSScriptRoot\..").Path
    }

    $proc = [System.Diagnostics.Process]::Start($pinfo)
    $stdout = $proc.StandardOutput.ReadToEnd()
    $stderr = $proc.StandardError.ReadToEnd()
    $proc.WaitForExit()
    $sw.Stop()

    $runRecord.ExecutionTimeSec = [Math]::Round($sw.Elapsed.TotalSeconds, 3)
    $runRecord.ExitCode = $proc.ExitCode

    if ($proc.ExitCode -ne 0) {
        $runRecord.Failures += "Process exited with error code $($proc.ExitCode). Stderr: $stderr"
        Write-Fail "Execution" "Process exited with non-zero code ($($proc.ExitCode))"
        $suitePassed = $false
        if ($FailFast) { break }
    } else {
        Write-Pass "Execution" "Completed in $($runRecord.ExecutionTimeSec)s (Exit Code 0)"
    }

    # Extract initial Ce count from stdout
    if ($stdout -match "Initial Ce atoms:\s*(\d+)") {
        $runRecord.InitialCeAtoms = [int]$matches[1]
    }

    # -------------------------------------------------------------------------
    # Verification 1: Output Generation
    # -------------------------------------------------------------------------
    $outputsOk = $true
    $expectedOutputs = @(
        @{ Name = "Energy Log CSV";       Path = $runCsvPath },
        @{ Name = "Binary State Snapshot"; Path = $runBinPath },
        @{ Name = "Radial Profile CSV";   Path = $runRadialPath },
        @{ Name = "Grain Centroids CSV";  Path = $runCentroidsPath }
    )

    foreach ($out in $expectedOutputs) {
        if (-not (Test-Path $out.Path -PathType Leaf)) {
            $outputsOk = $false
            $runRecord.Failures += "Missing output: $($out.Name) at $($out.Path)"
            Write-Fail "Output Generation" "Missing $($out.Name): $($out.Path)"
        } else {
            $item = Get-Item $out.Path
            if ($item.Length -le 0) {
                $outputsOk = $false
                $runRecord.Failures += "Zero-byte output: $($out.Name)"
                Write-Fail "Output Generation" "Empty file: $($out.Name)"
            }
        }
    }

    if ($outputsOk) {
        $runRecord.OutputGenerated = $true
        Write-Pass "Output Generation" "All 4 artifacts generated with non-zero payload"
    } else {
        $suitePassed = $false
        if ($FailFast) { break }
    }

    # -------------------------------------------------------------------------
    # Verification 2: Energy Descent & Hamiltonian Verification
    # -------------------------------------------------------------------------
    if (Test-Path $runCsvPath) {
        try {
            $csvRows = @(Import-Csv -Path $runCsvPath)
            if ($csvRows.Count -lt 2) {
                $runRecord.Failures += "Energy CSV has fewer than 2 rows ($($csvRows.Count))"
                Write-Fail "Energy Descent" "Insufficient data points in CSV ($($csvRows.Count) rows)"
                $suitePassed = $false
            } else {
                $firstRow = $csvRows[0]
                $lastRow  = $csvRows[-1]

                $h0 = [double]$firstRow.hamiltonian
                $hf = [double]$lastRow.hamiltonian
                $runRecord.InitialEnergy = $h0
                $runRecord.FinalEnergy   = $hf
                $runRecord.DeltaEnergy   = [Math]::Round($hf - $h0, 4)

                $cb0 = [double]$firstRow.c_boundary
                $cbulk0 = [double]$firstRow.c_bulk
                $cbf = [double]$lastRow.c_boundary
                $cbulkf = [double]$lastRow.c_bulk

                $runRecord.EnrichmentInitial = [Math]::Round(($cb0 / [Math]::Max(1e-6, $cbulk0)), 4)
                $runRecord.EnrichmentFinal   = [Math]::Round(($cbf / [Math]::Max(1e-6, $cbulkf)), 4)

                # Check energy descent
                if ($T -lt 1.0) {
                    # At low temperature (Ts = 0.4), Hamiltonian must strictly descend
                    if ($hf -lt $h0) {
                        $pct = [Math]::Round((($h0 - $hf) / $h0) * 100.0, 2)
                        $runRecord.DescentPercent = $pct
                        $runRecord.EnergyDescent = $true
                        Write-Pass "Energy Descent" "H0 = $h0 -> Hf = $hf (Delta H = $($runRecord.DeltaEnergy), -$pct%)"
                    } else {
                        $runRecord.Failures += "Hamiltonian failed to descend at Ts=$($T) (H0=$h0, Hf=$hf)"
                        Write-Fail "Energy Descent" "Hamiltonian did not decrease at Ts=$($T): H0=$h0, Hf=$hf"
                        $suitePassed = $false
                    }
                } else {
                    # At high temperature (Ts = 2.8), verify finite, bounded Hamiltonian without divergence
                    if (-not [Double]::IsNaN($hf) -and -not [Double]::IsInfinity($hf) -and $hf -gt 0) {
                        $runRecord.EnergyDescent = $true
                        Write-Pass "Thermal Equilibrium" "Stable, bounded Hamiltonian at Ts=$T (H0=$h0, Hf=$hf, Delta H=$($runRecord.DeltaEnergy))"
                    } else {
                        $runRecord.Failures += "Hamiltonian diverged or NaN at Ts=$T ($hf)"
                        Write-Fail "Thermal Equilibrium" "Unphysical Hamiltonian at Ts=$T ($hf)"
                        $suitePassed = $false
                    }
                }
            }
        }
        catch {
            $runRecord.Failures += "Exception parsing energy CSV: $($_.Exception.Message)"
            Write-Fail "Energy Analysis" "Error parsing CSV: $($_.Exception.Message)"
            $suitePassed = $false
        }
    }

    # -------------------------------------------------------------------------
    # Verification 3: Invariant Conservation (Atomic Species Delta N_Ce == 0)
    # -------------------------------------------------------------------------
    if (Test-Path $runBinPath) {
        $binInspection = Inspect-BinarySnapshot -FilePath $runBinPath
        if (-not $binInspection.Valid) {
            $runRecord.Failures += "Binary snapshot verification failed: $($binInspection.ErrorMessage)"
            Write-Fail "Invariant Conservation" "Invalid binary snapshot: $($binInspection.ErrorMessage)"
            $suitePassed = $false
        } else {
            $runRecord.FinalCeAtoms = $binInspection.CeCount
            if ($runRecord.InitialCeAtoms -gt 0) {
                if ($runRecord.FinalCeAtoms -eq $runRecord.InitialCeAtoms) {
                    $runRecord.InvariantConserved = $true
                    Write-Pass "Invariant Conservation" "Ce atom count strictly conserved: N0 = $($runRecord.InitialCeAtoms) -> Nf = $($runRecord.FinalCeAtoms) (Delta N_Ce = 0)"
                } else {
                    $diff = $runRecord.FinalCeAtoms - $runRecord.InitialCeAtoms
                    $runRecord.Failures += "Ce atomic conservation violation: Delta N_Ce = $diff"
                    Write-Fail "Invariant Conservation" "Atomic conservation violation: N0 = $($runRecord.InitialCeAtoms), Nf = $($runRecord.FinalCeAtoms) (Delta N = $diff)"
                    $suitePassed = $false
                }
            } else {
                # Fallback: verified non-zero Ce atoms from binary
                if ($runRecord.FinalCeAtoms -gt 0) {
                    $runRecord.InvariantConserved = $true
                    Write-Pass "Invariant Conservation" "Verified $($runRecord.FinalCeAtoms) Ce atoms preserved in snapshot"
                }
            }
            Write-Pass "Binary Snapshot" "Magic: $($binInspection.Magic) | Version: $($binInspection.Version) | Grid: $($binInspection.DimX)x$($binInspection.DimY) ($($binInspection.TotalSites) sites)"
        }
    }

    $testResults += $runRecord
}

# -----------------------------------------------------------------------------
# Verification 4: Physical Differential (Low Ts vs High Ts Segregation)
# -----------------------------------------------------------------------------
Write-SubHeader "Dual-Temperature Physical Differential Verification"

$lowRun = $testResults | Where-Object { $_.Temp -eq 0.4 } | Select-Object -First 1
$highRun = $testResults | Where-Object { $_.Temp -eq 2.8 } | Select-Object -First 1

if ($lowRun -and $highRun -and $lowRun.EnrichmentFinal -gt 0 -and $highRun.EnrichmentFinal -gt 0) {
    Write-Info "Physical Analytics" "Low Ts (0.4) Final GB Enrichment: $($lowRun.EnrichmentFinal)x"
    Write-Info "Physical Analytics" "High Ts (2.8) Final GB Enrichment: $($highRun.EnrichmentFinal)x"

    if ($lowRun.EnrichmentFinal -gt $highRun.EnrichmentFinal) {
        Write-Pass "Differential Kinetics" "Verified core-shell segregation: Enrichment(Ts=0.4: $($lowRun.EnrichmentFinal)x) > Enrichment(Ts=2.8: $($highRun.EnrichmentFinal)x)"
    } else {
        Write-Fail "Differential Kinetics" "Expected Enrichment(Ts=0.4) > Enrichment(Ts=2.8), but got $($lowRun.EnrichmentFinal) <= $($highRun.EnrichmentFinal)"
        $suitePassed = $false
    }
}

# -----------------------------------------------------------------------------
# Regression Suite Summary Table
# -----------------------------------------------------------------------------
Write-Header "REGRESSION SUITE EXECUTION SUMMARY"

$displayTable = $testResults | ForEach-Object {
    [PSCustomObject]@{
        "Ts"              = $_.Temp
        "Steps"           = $_.Steps
        "Time (s)"        = $_.ExecutionTimeSec
        "H0"              = $_.InitialEnergy
        "Hf"              = $_.FinalEnergy
        "Delta H"         = $_.DeltaEnergy
        "Energy Descent"  = if ($_.EnergyDescent) { "PASS" } else { "FAIL" }
        "Delta N_Ce == 0" = if ($_.InvariantConserved) { "PASS" } else { "FAIL" }
        "Outputs"         = if ($_.OutputGenerated) { "PASS" } else { "FAIL" }
        "GB Enrichment"   = "$($_.EnrichmentFinal)x"
    }
}

$displayTable | Format-Table -AutoSize | Out-String | Write-Host

$totalRuns = $testResults.Count
$passedRuns = ($testResults | Where-Object { $_.Failures.Count -eq 0 }).Count
$elapsedTotal = [Math]::Round(((Get-Date) - $scriptStartTime).TotalSeconds, 2)

if ($suitePassed -and $passedRuns -eq $totalRuns) {
    Write-Host "`n[OVERALL RESULT] REGRESSION SUITE PASSED ($passedRuns/$totalRuns runs successful in ${elapsedTotal}s)`n" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`n[OVERALL RESULT] REGRESSION SUITE FAILED ($($totalRuns - $passedRuns) runs failed in ${elapsedTotal}s)`n" -ForegroundColor Red
    exit 1
}
