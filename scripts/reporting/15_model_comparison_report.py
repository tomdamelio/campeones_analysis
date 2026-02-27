"""Marimo notebook: Reporte de Análisis EEG → Luminancia.

Sujeto: sub-27 | Sesión: VR

Todas las imágenes pre-generadas se embeben como base64 para que el export
HTML estático sea completamente autocontenido.
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium", app_title="EEG → Luminancia: Reporte de Análisis")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import base64
    import json as json_mod
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    RESULTS_PATH = PROJECT_ROOT / "results" / "modeling" / "luminance"
    STIMULI_PATH = PROJECT_ROOT / "stimuli" / "luminance"
    QA_PATH = PROJECT_ROOT / "results" / "qa" / "eeg"
    VALIDATION_PATH = PROJECT_ROOT / "results" / "validation"
    SUBJECT = 27

    def embed_png(path, title="", max_width="100%"):
        """Read a PNG file and return an mo.Html with base64 embedded image."""
        p = Path(path)
        if not p.exists():
            return None
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        html = f'<div style="text-align:center; margin:10px 0;">'
        if title:
            html += f'<p style="font-weight:bold; margin-bottom:5px;">{title}</p>'
        html += f'<img src="data:image/png;base64,{b64}" style="max-width:{max_width};" />'
        html += '</div>'
        return html

    def embed_pngs(paths_titles, ncols=2, max_width="48%"):
        """Embed multiple PNGs in a grid layout."""
        items = []
        for path, title in paths_titles:
            h = embed_png(path, title, max_width=max_width)
            if h:
                items.append(h)
        if not items:
            return None
        return '<div style="display:flex; flex-wrap:wrap; justify-content:center; gap:10px;">' + ''.join(items) + '</div>'

    return (
        RESULTS_PATH, STIMULI_PATH, QA_PATH, VALIDATION_PATH,
        PROJECT_ROOT, SUBJECT, json_mod, np, pd, plt, Path,
        embed_png, embed_pngs, base64,
    )


# ============================================================
# PORTADA
# ============================================================
@app.cell
def _(mo, SUBJECT):
    mo.md(
        rf"""
        # EEG → Luminancia: Reporte Completo de Análisis

        **Sujeto:** sub-{SUBJECT} &nbsp;|&nbsp; **Sesión:** VR &nbsp;|&nbsp;
        **Fecha:** 2026-02-26

        ---

        ## Índice

        0. Datos del Estímulo
        1. Control de Calidad (EEG QA con AutoReject)
        2. Validación Neurofisiológica (ERP / TFR / Cross-Correlation)
        3. Pipeline Predictivo (TDE + Covarianza)
        4. Optimización de Dimensionalidad (PCA)
        5. Resultados del Modelo Predictivo
        6. Variantes del Target (ΔL, Clasificación)
        7. Modelos Baseline
        8. Síntesis y Próximos Pasos
        """
    )
    return


# ============================================================
# SECCIÓN 0: DATOS DEL ESTÍMULO
# ============================================================
@app.cell
def _(mo):
    mo.md(
        r"""
        ## Sección 0 — Datos del Estímulo

        Se presentaron **7 videos de luminancia** de **60 segundos** cada uno.
        Lo único que varía es la **luminancia de una pantalla verde** (rango 0–255).
        4 IDs distintos (3, 7, 9, 12) en 7 runs.

        **EEG:** 32 canales BrainVision, $f_s = 250$ Hz.
        Preprocesamiento: filtrado pasa-banda + re-referencia + ICA (ICLabel).

        > **AutoReject no forma parte del preprocesamiento** — ver Sección I.

        **ROI:** `O1, O2, P3, P4, P7, P8, Pz, CP1, CP2, CP5, CP6` (11 canales).
        """
    )
    return


@app.cell
def _(STIMULI_PATH, np, pd, plt):
    _lum_map = {
        3: "green_intensity_video_3.csv", 7: "green_intensity_video_7.csv",
        9: "green_intensity_video_9.csv", 12: "green_intensity_video_12.csv",
    }
    _vdata = {}
    for _vid, _fn in _lum_map.items():
        _p = STIMULI_PATH / _fn
        if _p.exists():
            _df = pd.read_csv(_p)
            _df.columns = [c.strip().lower() for c in _df.columns]
            if "green_mean" in _df.columns:
                _df = _df.rename(columns={"green_mean": "luminance"})
            _vdata[_vid] = _df
    _fig, _axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
    for _ax, _vid in zip(_axes, sorted(_vdata.keys())):
        _df = _vdata[_vid]
        _ts = _df["timestamp"].values if "timestamp" in _df.columns else np.arange(len(_df))
        _ax.plot(_ts, _df["luminance"].values, lw=0.4, color="green", alpha=0.8)
        _ax.set_title(f"Video {_vid} — Luminancia cruda")
        _ax.set_ylabel("Intensidad")
        _ax.set_ylim(-5, 260)
    _axes[-1].set_xlabel("Tiempo (s)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(STIMULI_PATH, np, pd, plt):
    _lum_map = {
        3: "green_intensity_video_3.csv", 7: "green_intensity_video_7.csv",
        9: "green_intensity_video_9.csv", 12: "green_intensity_video_12.csv",
    }
    _vdata = {}
    for _vid, _fn in _lum_map.items():
        _p = STIMULI_PATH / _fn
        if _p.exists():
            _df = pd.read_csv(_p)
            _df.columns = [c.strip().lower() for c in _df.columns]
            if "green_mean" in _df.columns:
                _df = _df.rename(columns={"green_mean": "luminance"})
            _vdata[_vid] = _df
    _fig, _axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
    for _ax, _vid in zip(_axes, sorted(_vdata.keys())):
        _df = _vdata[_vid]
        _lum = _df["luminance"].values
        _ts = _df["timestamp"].values if "timestamp" in _df.columns else np.arange(len(_df))
        _ax.plot((_ts[:-1]+_ts[1:])/2, np.diff(_lum), lw=0.4, color="darkorange", alpha=0.8)
        _ax.set_title(f"Video {_vid} — ΔL frame a frame")
        _ax.set_ylabel("ΔL")
        _ax.axhline(0, color="gray", ls="--", lw=0.5)
    _axes[-1].set_xlabel("Tiempo (s)")
    _fig.tight_layout()
    _fig
    return


# ============================================================
# SECCIÓN I: QA
# ============================================================
@app.cell
def _(mo):
    mo.md(
        r"""
        ## Sección I — Control de Calidad (AutoReject)

        **Script:** `16_eeg_qa_autoreject.py`

        AutoReject identifica épocas ruidosas (data-driven). Las épocas malas
        se guardan en TSV para descartarlas en el modelado posterior.
        """
    )
    return


@app.cell
def _(mo, pd):
    qa_data = pd.DataFrame([
        {"Run": "002", "Video": 12, "Rechazo (%)": 9.1, "Consenso": 0.50, "n_interp": 8},
        {"Run": "003", "Video": 9, "Rechazo (%)": 22.1, "Consenso": 0.50, "n_interp": 8},
        {"Run": "004", "Video": 3, "Rechazo (%)": 6.4, "Consenso": 0.55, "n_interp": 8},
        {"Run": "006", "Video": 7, "Rechazo (%)": 6.4, "Consenso": 0.55, "n_interp": 8},
        {"Run": "007", "Video": 12, "Rechazo (%)": 0.2, "Consenso": 0.50, "n_interp": 8},
        {"Run": "009", "Video": 9, "Rechazo (%)": 15.8, "Consenso": 0.50, "n_interp": 8},
        {"Run": "010", "Video": 7, "Rechazo (%)": 4.9, "Consenso": 0.55, "n_interp": 8},
    ])
    mo.vstack([
        mo.md(f"**Tasa media de rechazo: {qa_data['Rechazo (%)'].mean():.1f}%**"),
        mo.ui.table(qa_data),
    ])
    return (qa_data,)


@app.cell
def _(QA_PATH, SUBJECT, mo, embed_pngs):
    _d = QA_PATH / "figures"
    _files = sorted(_d.glob(f"sub-{SUBJECT}_*.png")) if _d.exists() else []
    _html = embed_pngs(
        [(str(f), f.stem.split("_", 2)[-1].replace("_", " ")) for f in _files[:3]],
        ncols=3, max_width="30%",
    )
    if _html:
        mo.vstack([
            mo.md("### Mapas de rechazo AutoReject"),
            mo.Html(_html),
        ])
    else:
        mo.md("*No se encontraron heatmaps.*")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **Hallazgos:** Artefactos frontales (VR), canales centrales limpios,
        rechazos en bloques. Épocas malas almacenadas para modelado posterior.
        """
    )
    return


# ============================================================
# SECCIÓN II: VALIDACIÓN NEUROFISIOLÓGICA
# ============================================================
@app.cell
def _(mo):
    mo.md(
        r"""
        ## Sección II — Validación Neurofisiológica

        Verificar que **el EEG refleja los estímulos de luminancia** antes
        de entrenar modelos predictivos.

        ### II.1 — ERPs y TFR ante Cambios de Luminancia

        **Scripts:** `21_erp_luminance_changes.py` + `21b_erp_tfr_comparison.py`

        Épocas centradas en los 50 momentos de mayor $|\Delta L|$ por video,
        ventana $[-200, +800]$ ms, contrastadas contra control (baja variabilidad).
        """
    )
    return


@app.cell
def _(VALIDATION_PATH, SUBJECT, mo, embed_png):
    _p = VALIDATION_PATH / "erp" / f"sub-{SUBJECT}_erp_waveforms_grand_average.png"
    _html = embed_png(str(_p), "Grand Average ERP — Cambios de luminancia")
    if _html:
        mo.Html(_html)
    else:
        mo.md("*No se encontró el plot de ERP.*")
    return


@app.cell
def _(VALIDATION_PATH, SUBJECT, mo, pd):
    _csv = VALIDATION_PATH / "erp" / f"sub-{SUBJECT}_erp_peak_amplitudes.csv"
    if _csv.exists():
        _df = pd.read_csv(_csv)
        _g = _df[_df["VideoLabel"] == "grand_average"][
            ["Channel", "PeakLatency_ms", "PeakAmplitude_uV", "MeanAmplitude_100_300_uV"]
        ].copy()
        _g.columns = ["Canal", "Latencia (ms)", "Pico (µV)", "Media 100-300ms (µV)"]
        for _c in _g.columns[1:]:
            _g[_c] = _g[_c].round(2)
        mo.vstack([
            mo.md("**Amplitudes del Grand Average ERP:**"),
            mo.ui.table(_g),
        ])
    else:
        mo.md("*Sin datos ERP.*")
    return


@app.cell
def _(VALIDATION_PATH, SUBJECT, mo, embed_pngs):
    _d = VALIDATION_PATH / "erp"
    _pairs = [(str(_d / f"sub-{SUBJECT}_erp_waveforms_video_{v}.png"), f"ERP — Video {v}")
              for v in [3, 7, 9, 12]]
    _html = embed_pngs(_pairs, ncols=2, max_width="48%")
    if _html:
        mo.vstack([mo.md("**ERPs por video individual:**"), mo.Html(_html)])
    else:
        mo.md("*Sin ERPs por video.*")
    return


@app.cell
def _(VALIDATION_PATH, SUBJECT, mo, embed_pngs):
    _d = VALIDATION_PATH / "erp_tfr_comparison"
    _pairs = [
        (str(_d / f"sub-{SUBJECT}_tfr_contrast_Occipital.png"), "TFR — Occipital (ROI)"),
        (str(_d / f"sub-{SUBJECT}_tfr_contrast_O1.png"), "TFR — O1"),
        (str(_d / f"sub-{SUBJECT}_tfr_contrast_O2.png"), "TFR — O2"),
        (str(_d / f"sub-{SUBJECT}_tfr_contrast_Pz.png"), "TFR — Pz"),
    ]
    _html = embed_pngs(_pairs, ncols=2, max_width="48%")
    if _html:
        mo.vstack([mo.md("### TFR: Contraste Alta vs. Baja Variabilidad"), mo.Html(_html)])
    else:
        mo.md("*Sin plots TFR.*")
    return


@app.cell
def _(VALIDATION_PATH, SUBJECT, mo, embed_pngs):
    _d = VALIDATION_PATH / "erp_tfr_comparison"
    _pairs = [
        (str(_d / f"sub-{SUBJECT}_erp_contrast_Occipital.png"), "ERP — Occipital (ROI)"),
        (str(_d / f"sub-{SUBJECT}_erp_contrast_O1.png"), "ERP — O1"),
        (str(_d / f"sub-{SUBJECT}_erp_contrast_O2.png"), "ERP — O2"),
    ]
    _html = embed_pngs(_pairs, ncols=3, max_width="30%")
    if _html:
        mo.vstack([mo.md("### ERP: Contraste Alta vs. Baja Variabilidad"), mo.Html(_html)])
    else:
        mo.md("*Sin plots de contraste ERP.*")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Interpretación ERP / TFR

        1. **TFR:** Diferencias **claras** en bandas alpha/beta occipitales
           → el cerebro responde diferencialmente a la luminancia.
        2. **ERP:** Respuesta más difusa, componente tardío ~632 ms.
        3. **Conclusión:** Relación genuina entre luminancia y actividad neural.

        ---

        ### II.2 — Correlación Cruzada EEG–Luminancia

        **Script:** `22_cross_correlation.py`
        """
    )
    return


@app.cell
def _(VALIDATION_PATH, SUBJECT, mo, pd):
    _csv = VALIDATION_PATH / "cross_correlation" / f"sub-{SUBJECT}_cross_correlation_results.csv"
    if _csv.exists():
        xcorr_data = pd.read_csv(_csv)
        _d = xcorr_data[["RunID", "VideoID", "SliderOrder", "OptimalLag_s", "MaxCorrelation"]].copy()
        _d.columns = ["Run", "Video", "Slider", "Lag óptimo (s)", "Max correlación"]
        _d["Lag óptimo (s)"] = _d["Lag óptimo (s)"].round(3)
        _d["Max correlación"] = _d["Max correlación"].round(4)
        mo.vstack([
            mo.md(f"**Media lag:** {xcorr_data['OptimalLag_s'].mean():.3f}s | "
                   f"**Media correlación:** {xcorr_data['MaxCorrelation'].mean():.3f}"),
            mo.ui.table(_d),
        ])
    else:
        xcorr_data = pd.DataFrame()
        mo.md("*Sin datos.*")
    return (xcorr_data,)


@app.cell
def _(VALIDATION_PATH, SUBJECT, mo, embed_pngs):
    _d = VALIDATION_PATH / "cross_correlation"
    _files = sorted(_d.glob(f"sub-{SUBJECT}_*_xcorr_*.png")) if _d.exists() else []
    _pairs = [(str(f), f.stem.replace(f"sub-{SUBJECT}_", "").replace("_", " "))
              for f in _files]
    _html = embed_pngs(_pairs, ncols=2, max_width="48%")
    if _html:
        mo.vstack([mo.md("**Correlación cruzada por run:**"), mo.Html(_html)])
    else:
        mo.md("*Sin plots de cross-correlation.*")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Interpretación

        Desfases de **0–0.55 s**. Videos 7/12: corr >0.94. Video 9: ~0.52–0.84.
        **Los datos son válidos para modelado predictivo.**
        """
    )
    return


# ============================================================
# SECCIÓN III: PIPELINE PREDICTIVO
# ============================================================
@app.cell
def _(mo):
    mo.md(
        r"""
        ## Sección III — Pipeline Predictivo: TDE + Covarianza

        ### Paso 1: EEG → $\mathbf{X}_{\text{raw}} \in \mathbb{R}^{N \times 11}$

        ### Paso 2: TDE — $\mathbf{X}_{\text{TDE}} \in \mathbb{R}^{(N-20) \times 231}$ (±10 lags, estandarizado)

        ### Paso 3: PCA Global — $\mathbf{Z} \in \mathbb{R}^{(N-20) \times 20}$ (93.4% varianza)

        ### Paso 4: Epoching — Ventanas de 500ms (125 muestras), paso 100ms

        ### Paso 5: Covarianza — $\text{vech}(\Sigma_i) \in \mathbb{R}^{210}$

        ### Paso 6: Target — $\tilde{y}_i = (\bar{L}_i - \mu_v) / \sigma_v$
        ### Paso 7: Ridge Regression
        
        Grid de alpha: {0.01, 0.1, 1, 10, 100, 1000, 10000}
        
        - **CV interna (seleccion de alpha):** Leave-One-Group-Out (cada grupo = 1 video del set de entrenamiento)
        - **CV externa (evaluacion):** Leave-One-Video-Out (7 folds, cada fold deja 1 video como test)

        ```
        EEG crudo           TDE                  PCA Global           Epoching          Covarianza         Ridge
        (N × 11)    →    (N-20 × 231)    →    (N-20 × 20)    →    (125 × 20)    →    vech(Σ) ∈ ℝ²¹⁰    →    ŷ ∈ ℝ
                       ±10 lags, estand.     Global, std PCs      500ms, 100ms paso    Upper triangle        α via CV
        ```
        """
    )
    return


# ============================================================
# SECCIÓN IV: OPTIMIZACIÓN DIMENSIONALIDAD
# ============================================================
@app.cell
def _(mo):
    mo.md("## Sección IV — Optimización de Dimensionalidad (PCA)\n\n**Script:** `18_pca_sweep.py`")
    return


@app.cell
def _(RESULTS_PATH, SUBJECT, mo, pd):
    _p = RESULTS_PATH / "pca_sweep" / f"sub-{SUBJECT}_pca_sweep_results.tsv"
    if _p.exists():
        pca_data = pd.read_csv(_p, sep="\t")
        _d = pca_data.copy()
        _d.columns = ["n_comp", "Mean r", "Mean R²", "Mean ρ", "Mean RMSE"]
        _d["Features"] = _d["n_comp"].apply(lambda k: k*(k+1)//2)
        _d = _d[["n_comp", "Features", "Mean r", "Mean R²", "Mean ρ", "Mean RMSE"]]
        for _c in ["Mean r", "Mean R²", "Mean ρ", "Mean RMSE"]:
            _d[_c] = _d[_c].round(4)
        mo.ui.table(_d)
    else:
        pca_data = pd.DataFrame()
    return (pca_data,)


@app.cell
def _(RESULTS_PATH, SUBJECT, mo, embed_png):
    _html = embed_png(
        str(RESULTS_PATH / "pca_sweep" / f"sub-{SUBJECT}_pca_cumulative_variance.png"),
        "Varianza acumulada vs. componentes PCA",
    )
    if _html:
        mo.Html(_html)
    else:
        mo.md("*Sin plot de varianza.*")
    return


@app.cell
def _(RESULTS_PATH, SUBJECT, mo, embed_png):
    _html = embed_png(
        str(RESULTS_PATH / "pca_sweep" / f"sub-{SUBJECT}_pca_sweep_performance.png"),
        "Performance del modelo vs. componentes PCA",
    )
    if _html:
        mo.Html(_html)
    else:
        mo.md("*Sin plot de performance PCA.*")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **Hallazgos:** Rango óptimo: 10–30 componentes. A 40+ overfitting. A 80+ colapso.
        **Decisión:** $K = 20$ (93.4% varianza, 210 features).
        """
    )
    return


# ============================================================
# SECCIÓN V: RESULTADOS
# ============================================================
@app.cell
def _(RESULTS_PATH, SUBJECT, pd):
    _p = RESULTS_PATH / "raw_tde" / "500ms" / f"sub-{SUBJECT}_raw_tde_model_results.csv"
    tde_results = pd.read_csv(_p) if _p.exists() else pd.DataFrame()
    return (tde_results,)


@app.cell
def _(tde_results, mo):
    if not tde_results.empty:
        _d = tde_results[["TestVideo", "TrainPearsonR", "R2", "PearsonR", "SpearmanRho", "RMSE", "BestAlpha"]].copy()
        _d.columns = ["Video", "Train r", "R²", "Test r", "ρ", "RMSE", "α"]
        for _c in ["Train r", "R²", "Test r", "ρ", "RMSE"]:
            _d[_c] = _d[_c].round(4)
        _mr = tde_results["PearsonR"].mean()
        _mr2 = tde_results["R2"].mean()
        _mrho = tde_results["SpearmanRho"].mean()
        _mtr = tde_results["TrainPearsonR"].mean()
        mo.vstack([
            mo.md(
                f"""## Sección V — Resultados del Modelo\n\n"""
                f"""Promedios across los 7 folds de LOVO-CV (cada fold deja un video como test):\n\n"""
                f"""| Métrica | Media across folds |\n"""
                f"""|---------|-------------------:|\n"""
                f"""| Train $r$ | {_mtr:.4f} |\n"""
                f"""| Test $r$ | {_mr:.4f} |\n"""
                f"""| Test $R^2$ | {_mr2:.4f} |\n"""
                f"""| Test Spearman $\\rho$ | {_mrho:.4f} |\n"""
                f"""| Gap train-test | {_mtr-_mr:.3f} |\n"""
            ),
            mo.ui.table(_d),
        ])
    return


@app.cell
def _(tde_results, np, plt):
    if not tde_results.empty:
        _rho = tde_results["SpearmanRho"].values
        _r = tde_results["PearsonR"].values
        _r2 = tde_results["R2"].values
        _fig, _axes = plt.subplots(1, 3, figsize=(15, 5))
        np.random.seed(42)
        for _ax, (_nm, _v) in zip(_axes, [("Spearman ρ", _rho), ("Pearson r", _r), ("R²", _r2)]):
            _parts = _ax.violinplot([_v], positions=[0], showmedians=True, widths=0.6)
            for _pc in _parts["bodies"]:
                _pc.set_facecolor("steelblue")
                _pc.set_alpha(0.3)
            _ax.scatter(np.random.uniform(-0.08, 0.08, len(_v)), _v,
                        s=60, c="steelblue", alpha=0.8, edgecolors="white", zorder=5)
            _ax.axhline(0, color="red", ls="--", lw=1.5, label="Baseline (r=0)")
            _ax.set_ylabel(_nm)
            _ax.set_xticks([0])
            _ax.set_xticklabels(["TDE+Cov"])
            _ax.legend(fontsize=8)
            _ax.set_title(f"{_nm} (test)")
            _ax.grid(axis="y", alpha=0.3)
        _fig.suptitle("Performance por fold — LOVO-CV", fontsize=14)
        _fig.tight_layout()
        _fig
    return


@app.cell
def _(RESULTS_PATH, SUBJECT, mo, embed_png):
    _html = embed_png(
        str(RESULTS_PATH / "raw_tde" / "500ms" / f"sub-{SUBJECT}_raw_tde_model_predictions.png"),
        "Luminancia real vs. predicha por fold",
    )
    if _html:
        mo.Html(_html)
    else:
        mo.md("*Sin plot de predicciones.*")
    return


@app.cell
def _(RESULTS_PATH, SUBJECT, mo, embed_png):
    _html = embed_png(
        str(RESULTS_PATH / "raw_tde" / "500ms" / f"sub-{SUBJECT}_raw_tde_model_cv_results.png"),
        "Resultados CV del modelo TDE",
    )
    if _html:
        mo.Html(_html)
    else:
        mo.md("")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Interpretación
        1. 6/7 folds con $r > 0$. Video 12 alcanza $R^2 > 0$.
        2. Gap train-test ~0.30 (moderado).
        3. Las series predichas capturan la tendencia general.
        """
    )
    return


# ============================================================
# SECCIÓN VI: VARIANTES
# ============================================================
@app.cell
def _(mo):
    mo.md("## Sección VI — Variantes del Target\n\n### VI.1 — Predicción de $\\Delta L$")
    return


@app.cell
def _(RESULTS_PATH, SUBJECT, mo, pd):
    _p = RESULTS_PATH / "delta_luminance" / f"sub-{SUBJECT}_delta_luminance_results.csv"
    if _p.exists():
        _df = pd.read_csv(_p)
        _s = _df.groupby("Model").agg(
            R2=("R2", "mean"), r=("PearsonR", "mean"),
            rho=("SpearmanRho", "mean"), RMSE=("RMSE", "mean"),
        ).round(4).reset_index()
        mo.vstack([
            mo.md("**No decodificable** (correlaciones ≈ 0):"),
            mo.ui.table(_s),
        ])
    return


@app.cell
def _(RESULTS_PATH, SUBJECT, mo, embed_png):
    _html = embed_png(
        str(RESULTS_PATH / "delta_luminance" / f"sub-{SUBJECT}_delta_luminance_predictions.png"),
        "ΔL real vs. predicho",
    )
    if _html:
        mo.Html(_html)
    return


@app.cell
def _(mo):
    mo.md("### VI.2 — Clasificación de Cambio vs. Estabilidad")
    return


@app.cell
def _(RESULTS_PATH, SUBJECT, mo, pd):
    _p = RESULTS_PATH / "change_classification" / f"sub-{SUBJECT}_change_classifier_cv_results.csv"
    if _p.exists():
        _df = pd.read_csv(_p)
        _cols = [c for c in ["TestVideo", "Accuracy", "Precision", "Recall", "F1", "AUC_ROC"] if c in _df.columns]
        _d = _df[_cols].copy()
        for _c in _d.columns[1:]:
            _d[_c] = _d[_c].round(4)
        _auc = _df["AUC_ROC"].dropna().mean()
        mo.vstack([
            mo.md(f"**AUC-ROC: {_auc:.3f}** — peor que azar."),
            mo.ui.table(_d),
        ])
    return


@app.cell
def _(RESULTS_PATH, SUBJECT, mo, embed_png):
    _html = embed_png(
        str(RESULTS_PATH / "change_classification" / f"sub-{SUBJECT}_change_classifier_cv_results.png"),
        "Clasificador de cambio — Resultados CV",
    )
    if _html:
        mo.Html(_html)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Interpretación

        | Target | Mean $r$ | Resultado |
        |--------|:--------:|-----------|
        | Luminancia abs. | **0.113** | ✅ Señal débil |
        | $\Delta L$ | 0.007 | ❌ No decodificable |
        | Cambio binario | AUC=0.43 | ❌ Peor que azar |
        """
    )
    return


# ============================================================
# SECCIÓN VII: BASELINES
# ============================================================
@app.cell
def _(RESULTS_PATH, SUBJECT, mo, pd):
    _bl = RESULTS_PATH / "baselines"
    _rows = []
    _mp = _bl / f"sub-{SUBJECT}_mean_baseline.csv"
    if _mp.exists():
        _mdf = pd.read_csv(_mp)
        _rows.append({"Modelo": "Mean", "R²": f"{_mdf['R2'].mean():.4f}", "r": f"{_mdf['PearsonR'].mean():.4f}", "RMSE": f"{_mdf['RMSE'].mean():.4f}"})
    _sp = _bl / f"sub-{SUBJECT}_shuffle_baseline.csv"
    if _sp.exists():
        _sdf = pd.read_csv(_sp)
        _rows.append({"Modelo": "Shuffle", "R²": f"{_sdf['R2'].mean():.4f}", "r": f"{_sdf['PearsonR'].mean():.4f}", "RMSE": f"{_sdf['RMSE'].mean():.4f}"})
    mo.vstack([
        mo.md("## Sección VII — Modelos Baseline"),
        mo.ui.table(pd.DataFrame(_rows)),
    ])
    return


# ============================================================
# SECCIÓN VIII: SÍNTESIS
# ============================================================
@app.cell
def _(mo):
    mo.md(
        r"""
        ## Sección VIII — Síntesis y Próximos Pasos

        ### Comparación consolidada

        | Modelo | Mean $r$ | Mean $R^2$ | Resultado |
        |--------|:--------:|:----------:|-----------|
        | Mean Baseline | 0.000 | 0.000 | Línea de base |
        | Shuffle Baseline | ~0.000 | ~-0.04 | Azar |
        | **TDE + Cov** | **0.113** | **-0.056** | ✅ Señal débil |
        | TDE + Cov → ΔL | 0.007 | -0.176 | ❌ No decodificable |
        | Clasif. cambio | AUC=0.43 | — | ❌ Peor que azar |

        ---

        ### Conclusiones

        1. **Datos válidos:** Los TFR muestran diferencias claras en alpha/beta
           occipital. Las cross-correlations confirman relación EEG-luminancia
           (>0.94). Hay señal que todavía no capturamos completamente.

        2. **Señal débil pero real:** $r \approx 0.11$ supera baselines.
           Los TFR sugieren información espectral que el modelo lineal no
           aprovecha óptimamente.

        3. **Cambios rápidos no decodificables:** La covarianza de 500ms
           retiene solo dinámicas tónicas/lentas.

        ---

        ### Próximos pasos

        1. **Integrar AutoReject en preprocesamiento**
        2. **Temporal Generalization** — barrer lags para encontrar la
           ventana temporal óptima de predicción
        3. **Modelos no-lineales** (gradient boosting, redes neuronales)
        4. **TDE-GLHMM completo** (estados ocultos)
        5. **Multi-sujeto**
        """
    )
    return


if __name__ == "__main__":
    app.run()
