$py = "C:\micromamba\envs\campeones\python.exe"
$scripts = @(
    "10_luminance_base_model",
    "11_luminance_spectral_model",
    "12_luminance_spectral_tde_model",
    "13_luminance_raw_tde_model"
)
foreach ($s in $scripts) {
    foreach ($e in @(0.5, 1.0)) {
        Write-Host "=== Running $s epoch $e s ==="
        & $py "scripts/modeling/$s.py" --epoch-duration $e
    }
}
