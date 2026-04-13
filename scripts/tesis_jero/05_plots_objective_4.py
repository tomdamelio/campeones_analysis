# ==============================
# LIBRERÍAS
# ==============================
import os
import glob
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon  # se mantiene por si se reutiliza en análisis exploratorio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ==============================
# CONFIGURACIÓN — ajustá estos paths
# ==============================
RESULTS_DIR = r"C:\Users\jeror\Desktop\Cocuco\campeones_analysis\results\analysis_models_obj2"
OUTPUT_DIR  = os.path.join(RESULTS_DIR, "comparison_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variables internas para buscar archivos (en inglés)
DIMENSIONS  = ["arousal", "valence"]

# Diccionario para traducir a la hora de hacer el plot (hardcodeo para títulos)
DIM_LABELS = {
    "arousal": "Arousal",
    "valence": "Valencia"
}

MODEL_TYPES = ["ridge", "random_forest", "gradient_boosting", "svr"]

# Métricas a comparar: (columna_modelo, columna_dummy, label, mejor_es)
METRICS = [
    ("Mean_R2",   "Mean_DummyR2",   "R²",   "higher"),
    ("Mean_RMSE", "Mean_DummyRMSE", "RMSE", "lower"),
    ("Mean_MAE",  "Mean_DummyMAE",  "MAE",  "lower"),
]

# ── Paleta ────────────────────────────────────────────────────────────────────
CLR_MODEL   = "#2C6FAC"   # azul modelo
CLR_DUMMY   = "#C0392B"   # rojo dummy
CLR_WIN     = "#27AE60"   # verde  → modelo mejor
CLR_LOSE    = "#E74C3C"   # rojo   → dummy mejor
CLR_TIE     = "#95A5A6"   # gris   → empate / sin diferencia
BG_COLOR    = "#F7F9FC"
GRID_COLOR  = "#DDE3EC"

# ==============================
# FUNCIONES AUXILIARES
# ==============================

def load_summaries(results_dir: str, dimensions: list, model_types: list) -> pd.DataFrame:
    """
    Lee todos los subject_summary_*.csv disponibles y los concatena.
    Agrega columna 'Model_Type' inferida del nombre de archivo.
    """
    frames = []
    for dim in dimensions:
        dim_path = os.path.join(results_dir, dim)
        if not os.path.isdir(dim_path):
            print(f"⚠️  Carpeta no encontrada: {dim_path}")
            continue

        for mt in model_types:
            pattern = os.path.join(dim_path, f"subject_summary_{dim}_{mt}.csv")
            files   = glob.glob(pattern)
            if not files:
                continue
            for fp in files:
                df = pd.read_csv(fp)
                df["Model_Type"] = mt
                frames.append(df)
                print(f"  ✅ Cargado: {os.path.basename(fp)}  ({len(df)} sujetos)")

    if not frames:
        raise FileNotFoundError(
            f"No se encontró ningún subject_summary_*.csv en {results_dir}. "
            "Verificá que el pipeline original ya corrió y que RESULTS_DIR es correcto."
        )

    df_all = pd.concat(frames, ignore_index=True)

    # Asegurar columnas de dummy (pueden estar ausentes en versiones viejas del CSV)
    for _, dummy_col, _, _ in METRICS:
        if dummy_col not in df_all.columns:
            df_all[dummy_col] = np.nan
            print(f"  ⚠️  Columna '{dummy_col}' no encontrada — se rellenará con NaN")

    return df_all


def group_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada combinación (Dimension, Model_Type) calcula:
      - Media y SD de modelo y dummy (por métrica)
      - Diferencia media
      - Test de Wilcoxon
      - N sujetos
    """
    rows = []
    groups = df.groupby(["Dimension", "Model_Type"])

    for (dim, mt), grp in groups:
        row = {"Dimension": dim, "Model_Type": mt, "N_Subjects": len(grp)}

        for model_col, dummy_col, label, better in METRICS:
            if model_col not in grp.columns or dummy_col not in grp.columns:
                continue

            m_vals = grp[model_col].dropna().values
            d_vals = grp[dummy_col].dropna().values

            mask = ~(np.isnan(grp[model_col].values) | np.isnan(grp[dummy_col].values))
            m_paired = grp[model_col].values[mask]
            d_paired = grp[dummy_col].values[mask]

            row[f"{label}_Model_Mean"] = np.mean(m_vals)
            row[f"{label}_Model_SD"]   = np.std(m_vals, ddof=1)
            row[f"{label}_Dummy_Mean"] = np.mean(d_vals)
            row[f"{label}_Dummy_SD"]   = np.std(d_vals, ddof=1)
            row[f"{label}_Delta_Mean"] = np.mean(m_paired - d_paired)

        rows.append(row)

    return pd.DataFrame(rows)


# ==============================
# PLOTS POR SUJETO
# ==============================

def _subject_axis(
    ax,
    df_dim: pd.DataFrame,
    model_col: str,
    dummy_col: str,
    metric_label: str,
    better: str,
    title: str,
):
    """
    Dibuja en `ax` el scatter sujeto × métrica (modelo vs dummy).
    Cada sujeto es un punto en X; modelo (azul) y dummy (rojo) como barras.
    Una línea conecta modelo-dummy por sujeto; color según quién gana.
    """
    subjects = sorted(df_dim["Subject"].unique())
    x = np.arange(len(subjects))

    model_vals = [df_dim[df_dim["Subject"] == s][model_col].mean() for s in subjects]
    dummy_vals = [df_dim[df_dim["Subject"] == s][dummy_col].mean() for s in subjects]

    ax.set_facecolor(BG_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    for i, (mv, dv) in enumerate(zip(model_vals, dummy_vals)):
        if np.isnan(mv) or np.isnan(dv):
            continue
        if better == "higher":
            winner = "model" if mv > dv else ("dummy" if dv > mv else "tie")
        else:
            winner = "model" if mv < dv else ("dummy" if dv < mv else "tie")

        lc = CLR_WIN if winner == "model" else (CLR_LOSE if winner == "dummy" else CLR_TIE)
        ax.plot([i, i], [mv, dv], color=lc, lw=1.8, zorder=1)

    ax.scatter(x, model_vals, color=CLR_MODEL, s=60, zorder=3, label="Modelo", edgecolors="white", linewidths=0.6)
    ax.scatter(x, dummy_vals, color=CLR_DUMMY,  s=60, zorder=3, label="Dummy",  edgecolors="white", linewidths=0.6, marker="D")

    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric_label, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)

    # Leyenda compacta
    patch_model = mpatches.Patch(color=CLR_MODEL, label="Modelo (●)")
    patch_dummy = mpatches.Patch(color=CLR_DUMMY, label="Dummy (◆)")
    patch_win   = mpatches.Patch(color=CLR_WIN,   label="Modelo gana")
    patch_lose  = mpatches.Patch(color=CLR_LOSE,  label="Dummy gana")
    ax.legend(handles=[patch_model, patch_dummy, patch_win, patch_lose],
              fontsize=7.5, ncol=2, loc="best", framealpha=0.85)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_subjects_by_dimension(df: pd.DataFrame, output_dir: str):
    """
    Un plot por dimensión × métrica:
      filas = Model_Types presentes, columnas = dimensiones (si combinado)
    Genera:
      - Por dimensión: 1 figura por métrica (filas = modelos)
      - Combinado:     1 figura por métrica (filas = modelos, cols = dimensiones)
    """
    model_types = sorted(df["Model_Type"].unique())
    dims        = sorted(df["Dimension"].unique())

    for model_col, dummy_col, label, better in METRICS:
        # ── Plot separado por dimensión ──────────────────────────────────────
        for dim in dims:
            df_dim = df[df["Dimension"] == dim]
            n_models = len(model_types)
            
            # Obtenemos el nombre en español
            dim_display = DIM_LABELS.get(dim, dim.capitalize())

            fig, axes = plt.subplots(
                n_models, 1,
                figsize=(max(10, len(df_dim["Subject"].unique()) * 0.55 + 2), 4.5 * n_models),
                facecolor="white",
            )
            if n_models == 1:
                axes = [axes]

            fig.suptitle(
                f"{label} — Modelo vs Dummy  |  {dim_display}",
                fontsize=14, fontweight="bold", y=1.01,
            )

            for ax, mt in zip(axes, model_types):
                df_mt = df_dim[df_dim["Model_Type"] == mt]
                _subject_axis(ax, df_mt, model_col, dummy_col, label, better,
                               title=f"{mt.replace('_', ' ').title()}")

            plt.tight_layout()
            # El nombre del archivo se mantiene en inglés (usando 'dim')
            fname = os.path.join(output_dir, f"subjects_{dim}_{label.replace('²','2')}.png")
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  💾 {os.path.basename(fname)}")

        # ── Plot combinado (todas las dimensiones lado a lado) ───────────────
        if len(dims) > 1:
            n_models = len(model_types)
            n_cols   = len(dims)
            n_subjects_max = max(df[df["Dimension"] == d]["Subject"].nunique() for d in dims)

            fig, axes = plt.subplots(
                n_models, n_cols,
                figsize=(max(10, n_subjects_max * 0.5 + 2) * n_cols, 4.5 * n_models),
                facecolor="white",
            )
            # Normalizar a 2D
            if n_models == 1 and n_cols == 1:
                axes = [[axes]]
            elif n_models == 1:
                axes = [axes]
            elif n_cols == 1:
                axes = [[ax] for ax in axes]

            fig.suptitle(
                f"{label} — Modelo vs Dummy  |  Todas las dimensiones",
                fontsize=14, fontweight="bold", y=1.01,
            )

            for row_i, mt in enumerate(model_types):
                for col_i, dim in enumerate(dims):
                    ax = axes[row_i][col_i]
                    df_sub = df[(df["Dimension"] == dim) & (df["Model_Type"] == mt)]
                    dim_display = DIM_LABELS.get(dim, dim.capitalize())
                    _subject_axis(
                        ax, df_sub, model_col, dummy_col, label, better,
                        title=f"{mt.replace('_',' ').title()} — {dim_display}",
                    )

            plt.tight_layout()
            fname = os.path.join(output_dir, f"subjects_combined_{label.replace('²','2')}.png")
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  💾 {os.path.basename(fname)}")


# ==============================
# PLOT RESUMEN GRUPAL (barras)
# ==============================

def plot_group_comparison(group_df: pd.DataFrame, output_dir: str):
    """
    Barras agrupadas: modelo (azul) vs dummy (rojo)
    por dimensión × tipo de modelo, una figura por métrica.
    Anota p-valor del Wilcoxon sobre las barras del modelo.
    """
    dims        = sorted(group_df["Dimension"].unique())
    model_types = sorted(group_df["Model_Type"].unique())
    x           = np.arange(len(model_types))
    width       = 0.35

    for _, _, label, better in METRICS:
        mean_model_col = f"{label}_Model_Mean"
        sd_model_col   = f"{label}_Model_SD"
        mean_dummy_col = f"{label}_Dummy_Mean"
        sd_dummy_col   = f"{label}_Dummy_SD"

        n_dims = len(dims)
        fig, axes = plt.subplots(1, n_dims, figsize=(6 * n_dims, 5), facecolor="white")
        if n_dims == 1:
            axes = [axes]


        for ax, dim in zip(axes, dims):
            df_d = group_df[group_df["Dimension"] == dim].set_index("Model_Type")

            m_means = [df_d.loc[mt, mean_model_col] if mt in df_d.index else np.nan for mt in model_types]
            m_sds   = [df_d.loc[mt, sd_model_col]   if mt in df_d.index else np.nan for mt in model_types]
            d_means = [df_d.loc[mt, mean_dummy_col]  if mt in df_d.index else np.nan for mt in model_types]
            d_sds   = [df_d.loc[mt, sd_dummy_col]    if mt in df_d.index else np.nan for mt in model_types]
            ns      = [int(df_d.loc[mt, "N_Subjects"]) if mt in df_d.index else 0 for mt in model_types]

            ax.set_facecolor(BG_COLOR)
            ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)

            bars_m = ax.bar(x - width/2, m_means, width, yerr=m_sds,
                            color=CLR_MODEL, alpha=0.88, capsize=4,
                            error_kw={"elinewidth": 1.2}, label="Modelo", zorder=2)
            bars_d = ax.bar(x + width/2, d_means, width, yerr=d_sds,
                            color=CLR_DUMMY, alpha=0.72, capsize=4,
                            error_kw={"elinewidth": 1.2}, label="Dummy",  zorder=2)

            # Anotar n de sujetos bajo cada grupo de barras
            all_vals = [v for v in m_means + d_means if not np.isnan(v)]
            y_range  = (max(all_vals) - min(all_vals)) if len(all_vals) > 1 else 1.0


            ax.set_xticks(x)
            ax.set_xticklabels(
                [mt.replace("_", "\n") for mt in model_types], fontsize=9
            )
            ax.set_ylabel(label, fontsize=11)
            
            # Título traducido
            dim_display = DIM_LABELS.get(dim, dim.capitalize())
            ax.set_title(dim_display, fontsize=12, fontweight="bold")
            
            ax.legend(fontsize=9, framealpha=0.85)

            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

        plt.tight_layout()
        fname = os.path.join(output_dir, f"group_{label.replace('²','2')}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  💾 {os.path.basename(fname)}")


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("=" * 60)
    print("📊 ANÁLISIS COMPARATIVO: MODELO vs DUMMY")
    print("=" * 60)

    # 1. Cargar datos
    print("\n📂 Cargando CSVs...")
    df_all = load_summaries(RESULTS_DIR, DIMENSIONS, MODEL_TYPES)
    print(f"\n   Total filas cargadas: {len(df_all)}")
    print(f"   Dimensiones:  {sorted(df_all['Dimension'].unique())}")
    print(f"   Modelos:      {sorted(df_all['Model_Type'].unique())}")
    print(f"   Sujetos únicos: {df_all['Subject'].nunique()}")

    # 2. Comparación grupal con test estadístico
    print("\n🔬 Calculando comparación grupal (Wilcoxon)...")
    group_df = group_comparison(df_all)

    # Guardar tabla grupal
    group_csv = os.path.join(OUTPUT_DIR, "group_comparison_model_vs_dummy.csv")
    group_df.to_csv(group_csv, index=False)
    print(f"\n   Tabla grupal guardada: {os.path.basename(group_csv)}")

    # Imprimir resumen en consola (Acá también traducimos para que se lea lindo)
    print("\n" + "─" * 60)
    for _, row in group_df.iterrows():
        dim_display = DIM_LABELS.get(row['Dimension'], row['Dimension']).upper()
        print(f"\n  {dim_display} | {row['Model_Type'].upper()}  (n={int(row['N_Subjects'])})")
        for _, _, label, better in METRICS:
            mm = row.get(f"{label}_Model_Mean", np.nan)
            dm = row.get(f"{label}_Dummy_Mean", np.nan)
            delta = row.get(f"{label}_Delta_Mean", np.nan)
            print(f"    {label:<6}: Modelo={mm:.4f}  Dummy={dm:.4f}  Δ={delta:+.4f}")

    # 3. Plots grupales
    print("\n🎨 Generando plots grupales...")
    plot_group_comparison(group_df, OUTPUT_DIR)

    # 4. Plots por sujeto
    print("\n🎨 Generando plots por sujeto...")
    plot_subjects_by_dimension(df_all, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"🎉 Finalizado. Todos los outputs en:\n   {OUTPUT_DIR}")
    print("=" * 60)