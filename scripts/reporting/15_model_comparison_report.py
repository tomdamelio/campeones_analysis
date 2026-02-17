"""Marimo notebook: Luminance Model Comparison Report.

Interactive report comparing all EEG → luminance prediction models
implemented for sub-27. Includes exploratory analysis, model descriptions
with mathematical detail, performance metrics, and prediction visualizations.

Supports multiple epoch durations (500ms, 1000ms) with interactive selection.

Run with: marimo run scripts/reporting/15_model_comparison_report.py
Edit with: marimo edit scripts/reporting/15_model_comparison_report.py
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium", app_title="EEG → Luminance Model Comparison")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # EEG → Luminance: Comparación de Modelos Predictivos

        **Sujeto:** sub-27 &nbsp;|&nbsp; **Sesión:** VR &nbsp;|&nbsp;
        **Paradigma:** Visualización pasiva de videos con variación de luminancia

        Este reporte compara cuatro modelos de regresión que predicen la
        luminancia física del estímulo visual a partir de señales EEG
        preprocesadas. Todos los modelos usan **Leave-One-Video-Out
        Cross-Validation (LOVO-CV)** con **Ridge Regression**.

        La selección del hiperparámetro de regularización $\alpha$ se
        realiza mediante **GridSearchCV** con **LeaveOneGroupOut** como
        estrategia de inner CV. Esto garantiza que en la búsqueda de
        $\alpha$, nunca se mezclan epochs de un mismo video entre train
        y validation del inner loop, previniendo data leakage.

        $\hat{\beta} = \arg\min_{\beta} \left\| y - X\beta \right\|_2^2 + \alpha \left\| \beta \right\|_2^2$

        **Grid de $\alpha$:** $\{0.01, 0.1, 1.0, 10, 100, 1000, 10000\}$

        **Scoring:** Spearman $\rho$ (selección de $\alpha$ por correlación
        de rango, robusta a outliers).

        ---
        """
    )
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    RESULTS_PATH = PROJECT_ROOT / "results" / "modeling" / "luminance"
    STIMULI_PATH = PROJECT_ROOT / "stimuli" / "luminance"

    # Epoch durations to compare
    EPOCH_TAGS = ["500ms", "1000ms"]

    # Model definitions: label → (subdir, csv_filename)
    MODEL_DEFS = {
        "Base (Raw EEG)": ("base", "sub-27_base_model_results.csv"),
        "Spectral (Welch)": ("spectral", "sub-27_spectral_model_results.csv"),
        "Spectral TDE": ("tde", "sub-27_tde_model_results.csv"),
        "Raw TDE": ("raw_tde", "sub-27_raw_tde_model_results.csv"),
    }

    return (
        EPOCH_TAGS,
        MODEL_DEFS,
        Path,
        PROJECT_ROOT,
        RESULTS_PATH,
        STIMULI_PATH,
        np,
        pd,
        plt,
        sns,
    )



@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. Naturaleza de los Datos

        ### Estímulo: Luminancia física
        Los participantes observaron 4 videos experimentales (IDs: 3, 7, 9, 12)
        en un entorno de realidad virtual. La **luminancia física** se extrajo
        como la intensidad media del canal verde (rango 0–255) de cada frame
        del video, generando una serie temporal continua por video.

        ### Señal cerebral: EEG preprocesado
        Se registró EEG con 32 canales (sistema BrainVision, $f_s = 500$ Hz)
        durante la visualización. El preprocesamiento incluyó filtrado,
        re-referencia, rechazo de artefactos (autoreject) e ICA (ICLabel).

        ### ROI: Canales posteriores / occipitales
        Se seleccionaron **11 canales** del ROI posterior:
        O1, O2, P3, P4, P7, P8, Pz, CP1, CP2, CP5, CP6.
        Todos los modelos usan exclusivamente estos canales.

        ### Diseño experimental
        - **7 runs** (4 acq-a + 3 acq-b), cada uno con un segmento de video
        - **Epochs:** ventanas deslizantes con overlap = duración − 100 ms
        - **Variantes de duración de epoch:**
          - **500 ms:** paso = 0.1 s, overlap = 400 ms, ~250 muestras/epoch
          - **1000 ms:** paso = 0.1 s, overlap = 900 ms, ~500 muestras/epoch
        - **Target $y$:** Para cada epoch $[t_0, t_0 + \Delta t]$, se interpola
          linealmente la serie de luminancia del video y se calcula el
          **promedio** de $N = 100$ puntos equiespaciados dentro de la ventana:

          $y_i = \frac{1}{N} \sum_{k=1}^{N} L\!\left(t_0^{(i)} + \frac{k}{N} \Delta t\right)$

          Luego se aplica **z-score por video**: $\tilde{y}_i = (y_i - \mu_v) / \sigma_v$

        ### Cross-Validation
        - **Outer CV:** Leave-One-Video-Out (7 folds)
        - **Inner CV (selección de $\alpha$):** GridSearchCV con LeaveOneGroupOut
          - Scoring: **Spearman $\rho$** (robusto a outliers)
          - Grid: $\alpha \in \{0.01, 0.1, 1.0, 10, 100, 1000, 10000\}$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. Exploración de los Datos

        ### 2.1 Series temporales de luminancia (Script 08)
        """
    )
    return


@app.cell
def _(STIMULI_PATH, mo, np, pd, plt):
    _luminance_map = {
        3: "green_intensity_video_3.csv",
        7: "green_intensity_video_7.csv",
        9: "green_intensity_video_9.csv",
        12: "green_intensity_video_12.csv",
    }

    _video_data = {}
    for _vid, _fname in _luminance_map.items():
        _csv_path = STIMULI_PATH / _fname
        if _csv_path.exists():
            _df = pd.read_csv(_csv_path)
            _df.columns = [c.strip().lower() for c in _df.columns]
            if "green_mean" in _df.columns:
                _df = _df.rename(columns={"green_mean": "luminance"})
            _video_data[_vid] = _df

    _fig_lum, _axes_lum = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
    for _ax, _vid in zip(_axes_lum, sorted(_video_data.keys())):
        _df = _video_data[_vid]
        _ts = (
            _df["timestamp"].values
            if "timestamp" in _df.columns
            else np.arange(len(_df))
        )
        _lum = _df["luminance"].values
        _ax.plot(_ts, _lum, linewidth=0.4, color="green", alpha=0.8)
        _ax.set_title(f"Video {_vid} — Luminancia cruda (canal verde)")
        _ax.set_ylabel("Intensidad (0–255)")
        _ax.set_ylim(-5, 260)
    _axes_lum[-1].set_xlabel("Tiempo (s)")
    _fig_lum.tight_layout()

    mo.md("**Series temporales de luminancia** para los 4 videos experimentales:")
    _fig_lum
    return


@app.cell
def _(STIMULI_PATH, mo, np, pd, plt):
    _luminance_map2 = {
        3: "green_intensity_video_3.csv",
        7: "green_intensity_video_7.csv",
        9: "green_intensity_video_9.csv",
        12: "green_intensity_video_12.csv",
    }
    _video_data2 = {}
    for _vid, _fname in _luminance_map2.items():
        _csv_path = STIMULI_PATH / _fname
        if _csv_path.exists():
            _df = pd.read_csv(_csv_path)
            _df.columns = [c.strip().lower() for c in _df.columns]
            if "green_mean" in _df.columns:
                _df = _df.rename(columns={"green_mean": "luminance"})
            _video_data2[_vid] = _df

    _fig_diff, _axes_diff = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
    for _ax, _vid in zip(_axes_diff, sorted(_video_data2.keys())):
        _df = _video_data2[_vid]
        _lum = _df["luminance"].values
        _ts = (
            _df["timestamp"].values
            if "timestamp" in _df.columns
            else np.arange(len(_df))
        )
        _diffs = np.diff(_lum)
        _diff_ts = (_ts[:-1] + _ts[1:]) / 2.0
        _ax.plot(_diff_ts, _diffs, linewidth=0.4, color="darkorange", alpha=0.8)
        _ax.set_title(f"Video {_vid} — Diferencias temporales (Δluminancia)")
        _ax.set_ylabel("Δ Luminancia")
        _ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    _axes_diff[-1].set_xlabel("Tiempo (s)")
    _fig_diff.tight_layout()

    mo.md(
        "**Diferencias temporales** (frame a frame) — revelan la dinámica de cambio:"
    )
    _fig_diff
    return


@app.cell
def _(STIMULI_PATH, mo, np, pd):
    _luminance_map3 = {
        3: "green_intensity_video_3.csv",
        7: "green_intensity_video_7.csv",
        9: "green_intensity_video_9.csv",
        12: "green_intensity_video_12.csv",
    }
    _stats_rows = []
    for _vid, _fname in sorted(_luminance_map3.items()):
        _csv_path = STIMULI_PATH / _fname
        if _csv_path.exists():
            _df = pd.read_csv(_csv_path)
            _df.columns = [c.strip().lower() for c in _df.columns]
            if "green_mean" in _df.columns:
                _df = _df.rename(columns={"green_mean": "luminance"})
            _lum = _df["luminance"].values
            _ts = (
                _df["timestamp"].values
                if "timestamp" in _df.columns
                else np.arange(len(_df))
            )
            _stats_rows.append(
                {
                    "Video": _vid,
                    "Frames": len(_lum),
                    "Duración (s)": (
                        f"{float(_ts[-1] - _ts[0]):.1f}" if len(_ts) > 1 else "N/A"
                    ),
                    "Media": f"{np.mean(_lum):.1f}",
                    "Std": f"{np.std(_lum):.1f}",
                    "Min": f"{np.min(_lum):.0f}",
                    "Max": f"{np.max(_lum):.0f}",
                }
            )
    _stats_df = pd.DataFrame(_stats_rows)

    mo.md("### Estadísticas descriptivas de luminancia por video")
    mo.ui.table(_stats_df)
    return


@app.cell
def _(RESULTS_PATH, mo, plt):
    mo.md("### 2.2 Distribuciones de variables target (Script 14)")

    _dist_dir = RESULTS_PATH / "exploration" / "distributions"
    _dist_files = sorted(_dist_dir.glob("*.png")) if _dist_dir.exists() else []

    if _dist_files:
        _n_plots = len(_dist_files)
        _fig_dist, _axes_dist = plt.subplots(1, _n_plots, figsize=(5 * _n_plots, 4))
        if _n_plots == 1:
            _axes_dist = [_axes_dist]
        for _ax, _fpath in zip(_axes_dist, _dist_files):
            _img = plt.imread(str(_fpath))
            _ax.imshow(_img)
            _ax.set_title(
                _fpath.stem.replace("sub-27_", "").replace("_", " ").title(),
                fontsize=9,
            )
            _ax.axis("off")
        _fig_dist.tight_layout()
        _fig_dist
    else:
        mo.md(
            "*No se encontraron plots de distribución. Ejecutar Script 14 primero.*"
        )
    return



@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. Descripción de los Modelos

        Los cuatro modelos comparten la misma infraestructura de pipeline:
        mismos epochs, misma normalización (z-score por video), misma
        estrategia de CV (LOVO-CV con GridSearchCV + LeaveOneGroupOut
        para selección de $\alpha$), y mismo regresor (Ridge). La única
        diferencia es la **representación de entrada** (features).

        ---

        ### 3.1 Script 10 — Base (Raw EEG)

        **Representación:** Vectorización directa del EEG crudo.

        | Paso | Transformación | Shape |
        |------|---------------|-------|
        | 1 | Crop EEG al segmento de video (ROI: 11 canales) | `(11, n_samples_epoch)` |
        | 2 | `Vectorizer()` — aplana canales × muestras | `(11 × n_samples_epoch,)` |
        | 3 | `StandardScaler()` — z-score por feature | ídem |
        | 4 | `PCA(n_components=100)` | `(100,)` |
        | 5 | `Ridge(α)` — seleccionado por GridSearchCV | escalar |

        **Features por epoch:** 100 (PCA components)

        **Pipeline sklearn:** `Vectorizer → StandardScaler → PCA(100) → Ridge`

        ---

        ### 3.2 Script 11 — Spectral (Welch Band-Power)

        **Representación:** Potencia espectral por banda y canal.

        | Paso | Transformación | Shape |
        |------|---------------|-------|
        | 1 | Crop EEG al segmento de video (ROI: 11 canales) | `(11, n_samples_epoch)` |
        | 2 | Welch PSD por canal → potencia media en 5 bandas | `(11 × 5,)` = `(55,)` |
        | 3 | `StandardScaler()` — z-score por feature | `(55,)` |
        | 4 | `Ridge(α)` — seleccionado por GridSearchCV | escalar |

        **Bandas:** δ (1–4 Hz), θ (4–8 Hz), α (8–13 Hz), β (13–30 Hz), γ (30–45 Hz)

        **Features por epoch:** 55 (11 canales × 5 bandas, sin PCA)

        **Pipeline sklearn:** `StandardScaler → Ridge`

        ---

        ### 3.3 Script 12 — Spectral TDE (Multitaper + TDE + PCA)

        **Representación:** TDE sobre potencia espectral continua.

        | Paso | Transformación | Shape |
        |------|---------------|-------|
        | 1 | Crop EEG al segmento de video (ROI: 11 canales) | `(11, N)` |
        | 2 | Multitaper TFR → potencia continua en 5 bandas | `(N, 55)` |
        | 3 | TDE con ventana ±10 muestras (21 total) | `(N-20, 55 × 21)` = `(N-20, 1155)` |
        | 4 | `PCA(n_components=50)` sobre la matriz TDE | `(N-20, 50)` |
        | 5 | Epoching: mean + var por componente PCA | `(2 × 50,)` = `(100,)` |
        | 6 | `StandardScaler()` → `Ridge(α)` | escalar |

        **Features por epoch:** 100 (50 PCA mean + 50 PCA var)

        **Pipeline sklearn:** `StandardScaler → Ridge`

        ---

        ### 3.4 Script 13 — Raw TDE (Raw EEG + TDE + PCA)

        **Representación:** TDE directamente sobre EEG crudo.

        | Paso | Transformación | Shape |
        |------|---------------|-------|
        | 1 | Crop EEG al segmento de video (ROI: 11 canales) | `(11, N)` |
        | 2 | Transponer a time-major | `(N, 11)` |
        | 3 | TDE con ventana ±10 muestras (21 total) | `(N-20, 11 × 21)` = `(N-20, 231)` |
        | 4 | `PCA(n_components=50)` sobre la matriz TDE | `(N-20, 50)` |
        | 5 | Epoching: mean + var por componente PCA | `(2 × 50,)` = `(100,)` |
        | 6 | `StandardScaler()` → `Ridge(α)` | escalar |

        **Features por epoch:** 100 (50 PCA mean + 50 PCA var)

        **Pipeline sklearn:** `StandardScaler → Ridge`

        ---

        ### Resumen comparativo de features

        | Modelo | Input | Dim. TDE | PCA | Features/epoch |
        |--------|-------|----------|-----|----------------|
        | Base | Raw EEG vectorizado | — | 100 | 100 |
        | Spectral | Welch band-power | — | — | 55 |
        | Spectral TDE | Multitaper → TDE | 1155 | 50 | 100 |
        | Raw TDE | Raw EEG → TDE | 231 | 50 | 100 |
        """
    )
    return



@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. Resultados

        ### 4.1 Carga de resultados
        """
    )
    return


@app.cell
def _(EPOCH_TAGS, MODEL_DEFS, RESULTS_PATH, mo, pd):
    _all_frames: list = []
    _missing: list = []

    for _epoch_tag in EPOCH_TAGS:
        for _label, (_subdir, _csv_name) in MODEL_DEFS.items():
            _csv_path = RESULTS_PATH / _subdir / _epoch_tag / _csv_name
            if _csv_path.exists():
                _df = pd.read_csv(_csv_path)
                _df["ModelLabel"] = _label
                _df["Epoch"] = _epoch_tag
                _all_frames.append(_df)
            else:
                _missing.append(f"{_label} / {_epoch_tag}")

    if _all_frames:
        combined_df = pd.concat(_all_frames, ignore_index=True)
    else:
        combined_df = pd.DataFrame()

    if _missing:
        mo.md(
            f"⚠️ Resultados faltantes: {', '.join(_missing)}. "
            "Ejecutar `run_all_models.ps1` primero."
        )
    else:
        mo.md(
            f"✅ Cargados **{len(combined_df)}** filas "
            f"({len(MODEL_DEFS)} modelos × {len(EPOCH_TAGS)} epochs × 7 folds)."
        )

    combined_df
    return (combined_df,)



@app.cell
def _(combined_df, mo, pd):
    if combined_df.empty:
        _42_out = mo.md("*No hay datos cargados.*")
    else:
        _summary = (
            combined_df.groupby(["Epoch", "ModelLabel"])
            .agg(
                MeanTrainR=("TrainPearsonR", "mean"),
                MeanTestR=("PearsonR", "mean"),
                StdTestR=("PearsonR", "std"),
                MeanSpearman=("SpearmanRho", "mean"),
                MeanRMSE=("RMSE", "mean"),
                MeanBestAlpha=("BestAlpha", "mean"),
            )
            .round(4)
            .reset_index()
            .sort_values(["Epoch", "MeanTestR"], ascending=[True, False])
        )
        _42_out = mo.ui.table(_summary)
    mo.vstack([mo.md("### 4.2 Tabla resumen de métricas medias"), _42_out])
    return


@app.cell
def _(combined_df, mo, np, plt):
    if combined_df.empty:
        _43_content = mo.md("*No hay datos.*")
    else:
        _epochs = sorted(combined_df["Epoch"].unique())
        _models = sorted(combined_df["ModelLabel"].unique())
        _n_epochs = len(_epochs)

        _fig, _axes = plt.subplots(1, _n_epochs, figsize=(8 * _n_epochs, 5), squeeze=False)

        for _col_idx, _epoch_tag in enumerate(_epochs):
            _ax = _axes[0, _col_idx]
            _epoch_data = combined_df[combined_df["Epoch"] == _epoch_tag]
            _x_positions = np.arange(len(_models))
            _width = 0.12
            _videos = sorted(_epoch_data["TestVideo"].unique())

            for _vid_idx, _vid in enumerate(_videos):
                _vid_data = _epoch_data[_epoch_data["TestVideo"] == _vid]
                _vals = []
                for _model in _models:
                    _row = _vid_data[_vid_data["ModelLabel"] == _model]
                    _vals.append(float(_row["PearsonR"].values[0]) if len(_row) > 0 else 0.0)
                _offset = (_vid_idx - len(_videos) / 2 + 0.5) * _width
                _ax.bar(_x_positions + _offset, _vals, _width, label=_vid, alpha=0.8)

            for _m_idx, _model in enumerate(_models):
                _mean_r = _epoch_data[_epoch_data["ModelLabel"] == _model]["PearsonR"].mean()
                _ax.plot(
                    [_m_idx - 0.3, _m_idx + 0.3],
                    [_mean_r, _mean_r],
                    color="red",
                    linewidth=2,
                    linestyle="--",
                )

            _ax.set_xticks(_x_positions)
            _ax.set_xticklabels(_models, rotation=30, ha="right", fontsize=9)
            _ax.set_ylabel("Pearson r (test)")
            _ax.set_title(f"Epoch: {_epoch_tag}")
            _ax.legend(title="TestVideo", fontsize=7, ncol=2)
            _ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

        _fig.suptitle("Pearson r por fold y modelo", fontsize=14)
        _fig.tight_layout()
        _43_content = _fig

    mo.vstack([mo.md("### 4.3 Pearson r por fold (test)"), _43_content])
    return



@app.cell
def _(combined_df, mo, np, plt, sns):
    if combined_df.empty:
        _44_content = mo.md("*No hay datos.*")
    else:
        _epochs = sorted(combined_df["Epoch"].unique())
        _n_epochs = len(_epochs)

        _fig, _axes = plt.subplots(
            1, _n_epochs, figsize=(8 * _n_epochs, 5), squeeze=False
        )

        _model_order = [
            "Base (Raw EEG)",
            "Spectral (Welch)",
            "Spectral TDE",
            "Raw TDE",
        ]
        _palette = {
            "Base (Raw EEG)": "#4C72B0",
            "Spectral (Welch)": "#DD8452",
            "Spectral TDE": "#55A868",
            "Raw TDE": "#C44E52",
        }

        for _col_idx, _epoch_tag in enumerate(_epochs):
            _ax = _axes[0, _col_idx]
            _epoch_data = combined_df[combined_df["Epoch"] == _epoch_tag]

            sns.violinplot(
                data=_epoch_data,
                x="ModelLabel",
                y="SpearmanRho",
                hue="ModelLabel",
                order=_model_order,
                hue_order=_model_order,
                palette=_palette,
                inner=None,
                linewidth=0.8,
                alpha=0.3,
                ax=_ax,
                cut=0,
                legend=False,
            )

            sns.stripplot(
                data=_epoch_data,
                x="ModelLabel",
                y="SpearmanRho",
                hue="ModelLabel",
                order=_model_order,
                hue_order=_model_order,
                palette=_palette,
                size=7,
                jitter=0.12,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
                ax=_ax,
                legend=False,
            )

            for _m_idx, _model in enumerate(_model_order):
                _vals = _epoch_data[_epoch_data["ModelLabel"] == _model][
                    "SpearmanRho"
                ]
                if len(_vals) > 0:
                    _median = np.median(_vals)
                    _ax.plot(
                        [_m_idx - 0.25, _m_idx + 0.25],
                        [_median, _median],
                        color="black",
                        linewidth=2,
                        linestyle="--",
                        zorder=10,
                    )

            _ax.set_xticks(range(len(_model_order)))
            _ax.set_xticklabels(
                [m.replace(" ", "\n") for m in _model_order],
                fontsize=9,
            )
            _ax.set_xlabel("")
            _ax.set_ylabel("Spearman ρ (test)")
            _ax.set_title(f"Epoch: {_epoch_tag}")
            _ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

        _fig.suptitle(
            "Spearman ρ por fold — scatter + violin (mediana en negro)",
            fontsize=14,
        )
        _fig.tight_layout()
        _44_content = _fig

    mo.vstack([
        mo.md("### 4.4 Spearman ρ por modelo (scatter + violin)"),
        _44_content,
    ])
    return


@app.cell
def _(combined_df, mo, np, plt):
    if combined_df.empty:
        _45_content = mo.md("*No hay datos.*")
    else:
        _metrics = ["PearsonR", "SpearmanRho", "RMSE"]
        _titles = ["Mean Pearson r (test)", "Mean Spearman ρ (test)", "Mean RMSE"]
        _epochs = sorted(combined_df["Epoch"].unique())
        _models = sorted(combined_df["ModelLabel"].unique())

        _fig, _axes = plt.subplots(1, 3, figsize=(18, 5))
        _x = np.arange(len(_models))
        _width = 0.35
        _colors = {"500ms": "steelblue", "1000ms": "darkorange"}

        for _ax, _metric, _title in zip(_axes, _metrics, _titles):
            for _e_idx, _epoch_tag in enumerate(_epochs):
                _epoch_data = combined_df[combined_df["Epoch"] == _epoch_tag]
                _means = []
                _stds = []
                for _model in _models:
                    _model_data = _epoch_data[_epoch_data["ModelLabel"] == _model]
                    _means.append(_model_data[_metric].mean())
                    _stds.append(_model_data[_metric].std())
                _offset = (_e_idx - 0.5) * _width
                _ax.bar(
                    _x + _offset,
                    _means,
                    _width,
                    yerr=_stds,
                    label=_epoch_tag,
                    color=_colors.get(_epoch_tag, "gray"),
                    alpha=0.85,
                    capsize=3,
                )

            _ax.set_xticks(_x)
            _ax.set_xticklabels(_models, rotation=25, ha="right", fontsize=9)
            _ax.set_title(_title)
            _ax.legend()

        _fig.suptitle("Comparación de modelos: 500ms vs 1000ms", fontsize=14)
        _fig.tight_layout()
        _45_content = _fig

    mo.vstack([mo.md("### 4.5 Métricas medias por modelo y epoch"), _45_content])
    return


@app.cell
def _(combined_df, mo, plt):
    if combined_df.empty:
        _46_content = mo.md("*No hay datos.*")
    else:
        _epochs = sorted(combined_df["Epoch"].unique())
        _models = sorted(combined_df["ModelLabel"].unique())
        _n_epochs = len(_epochs)

        _fig, _axes = plt.subplots(1, _n_epochs, figsize=(7 * _n_epochs, 5), squeeze=False)

        for _col_idx, _epoch_tag in enumerate(_epochs):
            _ax = _axes[0, _col_idx]
            _epoch_data = combined_df[combined_df["Epoch"] == _epoch_tag]

            _train_means = []
            _test_means = []
            for _model in _models:
                _md = _epoch_data[_epoch_data["ModelLabel"] == _model]
                _train_means.append(_md["TrainPearsonR"].mean())
                _test_means.append(_md["PearsonR"].mean())

            _x = range(len(_models))
            _ax.bar([i - 0.2 for i in _x], _train_means, 0.35, label="Train", color="lightcoral")
            _ax.bar([i + 0.2 for i in _x], _test_means, 0.35, label="Test", color="steelblue")
            _ax.set_xticks(list(_x))
            _ax.set_xticklabels(_models, rotation=25, ha="right", fontsize=9)
            _ax.set_ylabel("Mean Pearson r")
            _ax.set_title(f"Train vs Test — {_epoch_tag}")
            _ax.legend()

        _fig.suptitle("Diagnóstico de Overfitting", fontsize=14)
        _fig.tight_layout()
        _46_content = _fig

    mo.vstack([
        mo.md("### 4.6 Train vs Test Pearson r (diagnóstico de overfitting)"),
        _46_content,
    ])
    return



@app.cell
def _(combined_df, mo, np, plt):
    if combined_df.empty or len(combined_df["Epoch"].unique()) < 2:
        _47_content = mo.md("*Se necesitan resultados de ambas duraciones de epoch.*")
    else:
        _models = sorted(combined_df["ModelLabel"].unique())
        _fig, _axes = plt.subplots(1, 3, figsize=(18, 5))
        _metrics = ["PearsonR", "SpearmanRho", "RMSE"]
        _titles = ["ΔPearson r", "ΔSpearman ρ", "ΔRMSE"]

        for _ax, _metric, _title in zip(_axes, _metrics, _titles):
            _deltas = []
            for _model in _models:
                _r500 = combined_df[
                    (combined_df["ModelLabel"] == _model) & (combined_df["Epoch"] == "500ms")
                ][_metric].mean()
                _r1000 = combined_df[
                    (combined_df["ModelLabel"] == _model) & (combined_df["Epoch"] == "1000ms")
                ][_metric].mean()
                _deltas.append(_r1000 - _r500)

            _colors = ["green" if d > 0 else "red" for d in _deltas]
            if _metric == "RMSE":
                _colors = ["green" if d < 0 else "red" for d in _deltas]

            _ax.bar(range(len(_models)), _deltas, color=_colors, alpha=0.8)
            _ax.set_xticks(range(len(_models)))
            _ax.set_xticklabels(_models, rotation=25, ha="right", fontsize=9)
            _ax.set_title(_title)
            _ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            _ax.set_ylabel(f"{_title} (1000ms − 500ms)")

        _fig.suptitle(
            "Efecto de la duración del epoch: 1000ms − 500ms (verde = mejora)",
            fontsize=13,
        )
        _fig.tight_layout()
        _47_content = _fig

    mo.vstack([
        mo.md("### 4.7 Efecto de la duración del epoch (500ms vs 1000ms)"),
        _47_content,
    ])
    return


@app.cell
def _(combined_df, mo, plt):
    if combined_df.empty:
        _48_content = mo.md("*No hay datos.*")
    else:
        _epochs = sorted(combined_df["Epoch"].unique())
        _models = sorted(combined_df["ModelLabel"].unique())
        _n_epochs = len(_epochs)

        _fig, _axes = plt.subplots(1, _n_epochs, figsize=(7 * _n_epochs, 5), squeeze=False)

        for _col_idx, _epoch_tag in enumerate(_epochs):
            _ax = _axes[0, _col_idx]
            _epoch_data = combined_df[combined_df["Epoch"] == _epoch_tag]

            for _model in _models:
                _md = _epoch_data[_epoch_data["ModelLabel"] == _model]
                _alphas = _md["BestAlpha"].values
                _ax.scatter(
                    [_model] * len(_alphas),
                    _alphas,
                    alpha=0.7,
                    s=50,
                )

            _ax.set_yscale("log")
            _ax.set_ylabel("Best α (log scale)")
            _ax.set_title(f"α seleccionado — {_epoch_tag}")
            _ax.tick_params(axis="x", rotation=25)

        _fig.suptitle("Distribución de α óptimo por fold", fontsize=14)
        _fig.tight_layout()
        _48_content = _fig

    mo.vstack([
        mo.md("### 4.8 Distribución de BestAlpha seleccionado por GridSearchCV"),
        _48_content,
    ])
    return



@app.cell
def _(combined_df, mo, plt, RESULTS_PATH):
    _49_header = mo.md(
        r"""
        ### 4.9 Predicciones del mejor modelo: luminancia real vs predicha

        Se muestran las predicciones (por fold) del modelo con mayor
        Spearman ρ medio. Cada subplot corresponde a un fold de LOVO-CV,
        con la luminancia real (z-scored) en azul y la predicha en rojo.
        Las imágenes se generan durante el entrenamiento (Scripts 10–13).
        """
    )

    if combined_df.empty:
        _49_content = mo.md("*No hay datos cargados.*")
    else:
        _model_to_subdir = {
            "Base (Raw EEG)": "base",
            "Spectral (Welch)": "spectral",
            "Spectral TDE": "tde",
            "Raw TDE": "raw_tde",
        }
        _model_to_prefix = {
            "Base (Raw EEG)": "base_model",
            "Spectral (Welch)": "spectral_model",
            "Spectral TDE": "tde_model",
            "Raw TDE": "raw_tde_model",
        }

        _epochs = sorted(combined_df["Epoch"].unique())
        _output_items = []
        for _epoch_tag in _epochs:
            _epoch_data = combined_df[combined_df["Epoch"] == _epoch_tag]
            _ranking = (
                _epoch_data.groupby("ModelLabel")["SpearmanRho"]
                .mean()
                .sort_values(ascending=False)
            )
            _best_model = _ranking.index[0]
            _best_rho = _ranking.values[0]

            _subdir = _model_to_subdir.get(_best_model, "")
            _prefix = _model_to_prefix.get(_best_model, "")
            _pred_png = (
                RESULTS_PATH
                / _subdir
                / _epoch_tag
                / f"sub-27_{_prefix}_predictions.png"
            )

            if _pred_png.exists():
                _img = plt.imread(str(_pred_png))
                _fig_pred, _ax_pred = plt.subplots(
                    figsize=(min(16, _img.shape[1] / 80), _img.shape[0] / 80)
                )
                _ax_pred.imshow(_img)
                _ax_pred.axis("off")
                _ax_pred.set_title(
                    f"{_epoch_tag} — Mejor modelo: {_best_model} "
                    f"(Spearman ρ medio = {_best_rho:.4f})",
                    fontsize=12,
                )
                _fig_pred.tight_layout()
                _output_items.append(_fig_pred)
            else:
                _output_items.append(
                    mo.md(
                        f"*{_epoch_tag}: PNG de predicciones no encontrado "
                        f"para {_best_model} ({_pred_png.name}).*"
                    )
                )

        _49_content = mo.vstack(_output_items)

    mo.vstack([_49_header, _49_content])
    return


@app.cell
def _(combined_df, mo):
    _5_header = mo.md(
        r"""
        ## 5. Interpretación

        ### Observaciones clave
        """
    )

    if not combined_df.empty:
        _epochs = sorted(combined_df["Epoch"].unique())
        _lines = []
        for _epoch_tag in _epochs:
            _epoch_data = combined_df[combined_df["Epoch"] == _epoch_tag]
            _summary = (
                _epoch_data.groupby("ModelLabel")["PearsonR"]
                .mean()
                .sort_values(ascending=False)
            )
            _best_model = _summary.index[0]
            _best_r = _summary.values[0]
            _worst_model = _summary.index[-1]
            _worst_r = _summary.values[-1]
            _lines.append(
                f"- **{_epoch_tag}:** Mejor modelo = {_best_model} "
                f"(r={_best_r:.4f}), peor = {_worst_model} (r={_worst_r:.4f})"
            )

        _lines.append("\n**Diagnóstico de overfitting (gap train−test):**")
        for _epoch_tag in _epochs:
            _epoch_data = combined_df[combined_df["Epoch"] == _epoch_tag]
            for _model in sorted(_epoch_data["ModelLabel"].unique()):
                _md = _epoch_data[_epoch_data["ModelLabel"] == _model]
                _train_r = _md["TrainPearsonR"].mean()
                _test_r = _md["PearsonR"].mean()
                _gap = _train_r - _test_r
                _lines.append(
                    f"- {_epoch_tag} / {_model}: train={_train_r:.4f}, "
                    f"test={_test_r:.4f}, gap={_gap:.4f}"
                )

        _5_content = mo.md("\n".join(_lines))
    else:
        _5_content = mo.md("*No hay datos.*")

    mo.vstack([_5_header, _5_content])
    return


@app.cell
def _(combined_df, mo):
    _6_header = mo.md(
        r"""
        ## 6. Tabla completa de resultados por fold
        """
    )

    if not combined_df.empty:
        _display_cols = [
            "Epoch",
            "ModelLabel",
            "TestVideo",
            "TrainSize",
            "TestSize",
            "TrainPearsonR",
            "PearsonR",
            "SpearmanRho",
            "RMSE",
            "BestAlpha",
        ]
        _available = [c for c in _display_cols if c in combined_df.columns]
        _full_table = (
            combined_df[_available]
            .sort_values(["Epoch", "ModelLabel", "TestVideo"])
            .reset_index(drop=True)
        )
        _6_content = mo.ui.table(_full_table)
    else:
        _6_content = mo.md("*No hay datos cargados.*")

    mo.vstack([_6_header, _6_content])
    return


if __name__ == "__main__":
    app.run()
