# ==============================
# LIBRERÍAS
# ==============================
import os
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ==============================
# CONFIGURACIÓN
# ==============================
INPUT_DIR  = r"C:\Users\jeror\Desktop\Cocuco\campeones_analysis\results\epoch_features"
OUTPUT_DIR = r"C:\Users\jeror\Desktop\Cocuco\campeones_analysis\results\analysis_models_obj2_stimulus"

VALENCE_CSV = os.path.join(INPUT_DIR, "valence_epochs_10s.csv")
AROUSAL_CSV = os.path.join(INPUT_DIR, "arousal_epochs_10s.csv")

EDA_FEATURES = [
    "EDA_Phasic_Mean",
    "EDA_SMNA_AUC",
    "EDA_Tonic_Mean",
    "EDA_SCR_Peaks_Count",
    "EDA_SMNA_Peaks_Count",
]

TARGET_VAR      = "Report_Mean"
TDE_HALF_WINDOW = 7
MIN_BLOCKS      = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# FUNCIÓN: VERIFICAR USABILIDAD
# ==============================
def check_subject_usability(df_subject: pd.DataFrame, subject_id) -> dict:
    df_subject = df_subject.copy()
    df_subject["block_id"] = (
        df_subject["Session"].str.upper() + "_" + df_subject["Block"].astype(str)
    )
    n_blocks = df_subject["block_id"].nunique()

    if n_blocks < MIN_BLOCKS:
        return {
            "usable":   False,
            "n_blocks": n_blocks,
            "reason":   f"Solo {n_blocks} bloque(s) únicos (mínimo requerido: {MIN_BLOCKS})",
        }
    return {"usable": True, "n_blocks": n_blocks, "reason": "OK"}


# ==============================
# FUNCIÓN: CONSTRUIR TDE
# ==============================
def build_tde(df: pd.DataFrame, features: list, half_window: int) -> pd.DataFrame:
    lags       = range(-half_window, half_window + 1)
    group_cols = ["Subject", "Session", "Block", "Stimulus"]
    lag_frames = []

    for lag in lags:
        shifted = (
            df.groupby(group_cols)[features]
            .shift(-lag)
            .rename(columns={f: f"{f}_lag{lag}" for f in features})
        )
        lag_frames.append(shifted)

    return pd.concat([df.drop(columns=features)] + lag_frames, axis=1)


# ==============================
# FUNCIÓN: LEAVE-ONE-STIMULUS-OUT CV
# ==============================
def run_loso_cv(
    df_subject: pd.DataFrame,
    subject_id,
    dimension_name: str,
    tde_cols: list,
) -> tuple:
    """
    Folds = estímulos únicos presentes en el sujeto.
    En cada fold:
      - test  = todas las épocas de ese estímulo (en cualquier sesión/bloque)
      - train = todas las épocas de los demás estímulos

    Lógica: como el mismo estímulo se repite en sesión A y B, el modelo
    nunca ve ese estímulo en train, lo que evalúa generalización
    a contenido emocional no visto.
    """
    df_subject = df_subject.copy()

    # fold_id = número de estímulo (puede repetirse en sesiones distintas,
    # pero lo dejamos juntos en test intencionalmente)
    folds = sorted(df_subject["Stimulus"].unique())
    fold_results = []

    for fold in folds:

        df_test  = df_subject[df_subject["Stimulus"] == fold]
        df_train = df_subject[df_subject["Stimulus"] != fold]

        cols_needed = [TARGET_VAR] + tde_cols
        train_clean = df_train[cols_needed].dropna()
        test_clean  = df_test[cols_needed].dropna()

        if len(train_clean) < 10 or len(test_clean) < 2:
            print(
                f"    ⚠️  Estímulo {fold}: datos insuficientes "
                f"(train={len(train_clean)}, test={len(test_clean)}), saltando"
            )
            continue

        X_train = train_clean[tde_cols].values
        y_train = train_clean[TARGET_VAR].values
        X_test  = test_clean[tde_cols].values
        y_test  = test_clean[TARGET_VAR].values

        # ── Escalado sin leakage ────────────────────────────────────────────
        scaler     = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        # ── 1. Ridge (sklearn) → métricas de predicción ────────────────────
        sk_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
        sk_model.fit(X_train_sc, y_train)
        preds = sk_model.predict(X_test_sc)

        rmse_val = np.sqrt(mean_squared_error(y_test, preds))
        mae_val  = mean_absolute_error(y_test, preds)
        r2_val   = r2_score(y_test, preds)

        # ── 2. OLS (statsmodels) → coeficientes e inferencia ───────────────
        X_train_sm = sm.add_constant(X_train_sc, has_constant="add")
        sm_model   = sm.OLS(y_train, X_train_sm).fit()

        coef_names = ["const"] + tde_cols
        ci         = sm_model.conf_int()
        ci_lower   = ci[:, 0] if isinstance(ci, np.ndarray) else ci.iloc[:, 0].values
        ci_upper   = ci[:, 1] if isinstance(ci, np.ndarray) else ci.iloc[:, 1].values

        coef_df = pd.DataFrame({
            "Feature":  coef_names,
            "Beta":     sm_model.params,
            "SE":       sm_model.bse,
            "t":        sm_model.tvalues,
            "P_Value":  sm_model.pvalues,
            "CI_Lower": ci_lower,
            "CI_Upper": ci_upper,
        })
        coef_df.insert(0, "Subject",   subject_id)
        coef_df.insert(1, "Dimension", dimension_name)
        coef_df.insert(2, "Fold",      f"stim_{fold}")

        print(
            f"    Estímulo {str(fold):<4} | "
            f"n_train={len(train_clean):>4} | n_test={len(test_clean):>3} | "
            f"RMSE={rmse_val:.4f} | MAE={mae_val:.4f} | R²={r2_val:.4f}"
        )

        fold_results.append({
            "Subject":     subject_id,
            "Dimension":   dimension_name,
            "Fold":        f"stim_{fold}",
            "Stimulus":    fold,
            "N_Train":     len(train_clean),
            "N_Test":      len(test_clean),
            "RMSE":        rmse_val,
            "MAE":         mae_val,
            "R2":          r2_val,
            "R2_train":    sm_model.rsquared,
            "AdjR2_train": sm_model.rsquared_adj,
            "F_stat":      sm_model.fvalue,
            "F_pvalue":    sm_model.f_pvalue,
            "Ridge_Alpha": sk_model.alpha_,
            "_coef_df":    coef_df,
        })

    if not fold_results:
        return None, None

    metrics_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "_coef_df"} for r in fold_results]
    )
    coefs_df = pd.concat([r["_coef_df"] for r in fold_results], ignore_index=True)

    return metrics_df, coefs_df


# ==============================
# FUNCIÓN PRINCIPAL: ANALIZAR UNA DIMENSIÓN
# ==============================
def analyze_dimension(filepath: str, dimension_name: str) -> dict | None:

    if not os.path.exists(filepath):
        print(f"⚠️  Archivo no encontrado: {filepath}")
        return None

    print("\n" + "=" * 60)
    print(f"📊 DIMENSIÓN: {dimension_name.upper()}")
    print("=" * 60)

    df_all = pd.read_csv(filepath)
    df_all["Session"] = df_all["Session"].str.upper()
    subjects = sorted(df_all["Subject"].unique())
    print(f"👥 Sujetos encontrados: {len(subjects)}")

    dim_output = os.path.join(OUTPUT_DIR, dimension_name)
    os.makedirs(dim_output, exist_ok=True)

    print(f"\n🔧 Construyendo TDE (half_window={TDE_HALF_WINDOW})...")
    df_tde = build_tde(df_all, EDA_FEATURES, TDE_HALF_WINDOW)

    tde_cols = [
        f"{feat}_lag{lag}"
        for feat in EDA_FEATURES
        for lag in range(-TDE_HALF_WINDOW, TDE_HALF_WINDOW + 1)
    ]

    usability_log = []
    all_metrics   = []
    all_coefs     = []

    for subj in subjects:
        print(f"\n👤 Sujeto: {subj}")
        df_subj = df_tde[df_tde["Subject"] == subj].copy()

        check = check_subject_usability(df_subj, subj)
        usability_log.append({
            "Subject":   subj,
            "Dimension": dimension_name,
            "N_Blocks":  check["n_blocks"],
            "Usable":    check["usable"],
            "Reason":    check["reason"],
        })

        if not check["usable"]:
            print(f"  ⛔ Descartado → {check['reason']}")
            continue

        print(f"  ✅ Sujeto apto ({check['n_blocks']} bloques) → corriendo LOSO-Stimulus CV...")
        metrics_df, coefs_df = run_loso_cv(df_subj, subj, dimension_name, tde_cols)

        if metrics_df is not None:
            all_metrics.append(metrics_df)
            all_coefs.append(coefs_df)

    # ── Guardar log de usabilidad ────────────────────────────────────────────
    usability_df = pd.DataFrame(usability_log)
    usability_df.to_csv(
        os.path.join(dim_output, f"usability_log_{dimension_name}.csv"), index=False
    )
    n_usable = usability_df["Usable"].sum()
    print(f"\n📋 Sujetos aptos:      {n_usable}")
    print(f"📋 Sujetos descartados: {len(subjects) - n_usable}")

    if not all_metrics:
        print(f"⚠️  Sin resultados para dimensión: {dimension_name}")
        return None

    # ── Resultados por fold ──────────────────────────────────────────────────
    results_df = pd.concat(all_metrics, ignore_index=True)
    results_df.to_csv(
        os.path.join(dim_output, f"fold_results_{dimension_name}.csv"), index=False
    )

    # ── Coeficientes por fold ────────────────────────────────────────────────
    coefs_df = pd.concat(all_coefs, ignore_index=True)
    coefs_df.to_csv(
        os.path.join(dim_output, f"coefficients_{dimension_name}.csv"), index=False
    )

    # ── Resumen por sujeto ───────────────────────────────────────────────────
    summary_df = (
        results_df.groupby(["Subject", "Dimension"])
        .agg(
            N_Folds   = ("Fold",  "count"),
            Mean_RMSE = ("RMSE",  "mean"),
            SD_RMSE   = ("RMSE",  "std"),
            Mean_MAE  = ("MAE",   "mean"),
            SD_MAE    = ("MAE",   "std"),
            Mean_R2   = ("R2",    "mean"),
            SD_R2     = ("R2",    "std"),
        )
        .reset_index()
    )
    summary_df.to_csv(
        os.path.join(dim_output, f"subject_summary_{dimension_name}.csv"), index=False
    )

    # ── Resumen global ───────────────────────────────────────────────────────
    global_summary = {
        "Dimension":       dimension_name,
        "N_Subjects":      len(summary_df),
        "Grand_Mean_RMSE": summary_df["Mean_RMSE"].mean(),
        "Grand_SD_RMSE":   summary_df["Mean_RMSE"].std(),
        "Grand_Mean_MAE":  summary_df["Mean_MAE"].mean(),
        "Grand_SD_MAE":    summary_df["Mean_MAE"].std(),
        "Grand_Mean_R2":   summary_df["Mean_R2"].mean(),
        "Grand_SD_R2":     summary_df["Mean_R2"].std(),
    }

    print(f"\n📈 RESUMEN GLOBAL ({dimension_name.upper()}):")
    for k, v in global_summary.items():
        print(f"   {k:<20}: {v}")

    print(f"\n💾 Resultados guardados en: {dim_output}")

    return {
        "results":   results_df,
        "coefs":     coefs_df,
        "summary":   summary_df,
        "global":    global_summary,
        "usability": usability_df,
    }


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    n_predictors = (2 * TDE_HALF_WINDOW + 1) * len(EDA_FEATURES)
    print("🚀 Iniciando análisis predictivo — Ridge + TDE + Leave-One-Stimulus-Out CV")
    print(f"   TDE half-window : {TDE_HALF_WINDOW}")
    print(f"   Lags            : {list(range(-TDE_HALF_WINDOW, TDE_HALF_WINDOW + 1))}")
    print(f"   Predictores     : {2 * TDE_HALF_WINDOW + 1} lags × {len(EDA_FEATURES)} features = {n_predictors}")
    print(f"   Mínimo bloques  : {MIN_BLOCKS}")

    res_arousal = analyze_dimension(AROUSAL_CSV, "arousal")
    res_valence = analyze_dimension(VALENCE_CSV, "valence")

    if res_arousal and res_valence:
        comparison = pd.DataFrame([res_arousal["global"], res_valence["global"]])
        comparison.to_csv(os.path.join(OUTPUT_DIR, "global_comparison.csv"), index=False)

        print("\n" + "=" * 60)
        print("🏁 COMPARACIÓN GLOBAL AROUSAL vs VALENCE")
        print("=" * 60)
        print(comparison.to_string(index=False))

    print("\n🎉 Finalizado.")