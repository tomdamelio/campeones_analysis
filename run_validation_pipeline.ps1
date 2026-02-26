$py = "C:\Users\au805392\micromamba\envs\campeones\python.exe"
$scripts = @(
    "scripts/qa/16_eeg_qa_autoreject.py",
    "scripts/modeling/17_baseline_models.py",
    "scripts/modeling/13_luminance_raw_tde_model.py",
    "scripts/modeling/18_pca_sweep.py",
    "scripts/modeling/19_delta_luminance_model.py",
    "scripts/modeling/20_change_classifier.py",
    "scripts/validation/21_erp_luminance_changes.py",
    "scripts/validation/22_cross_correlation.py",
    "scripts/reporting/15_model_comparison_report.py"
)

foreach ($s in $scripts) {
    Write-Host "=================="
    Write-Host "Running $s..."
    Write-Host "=================="
    & $py $s
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Script $s failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}
Write-Host "All scripts finished successfully."
