# ==============================
# LIBRERÍAS
# ==============================
from multiprocessing import dummy
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from itertools import product as iproduct

# Modelado
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

# ==============================
# CONFIGURACIÓN
# ==============================
INPUT_DIR  = r"C:\Users\jeror\Desktop\Cocuco\campeones_analysis\results\epoch_features"
OUTPUT_DIR = r"C:\Users\jeror\Desktop\Cocuco\campeones_analysis\results\analysis_models_obj2"

VALENCE_CSV = os.path.join(INPUT_DIR, "valence_epochs_10s.csv")
AROUSAL_CSV = os.path.join(INPUT_DIR, "arousal_epochs_10s.csv")

EDA_FEATURES = [
    "EDA_Phasic_Mean",
    "EDA_SMNA_AUC",
    "EDA_Tonic_Mean",
    "EDA_Phasic_Std",
    "EDA_Tonic_Std",
]

# Opciones: "ridge", "random_forest", "gradient_boosting", "svr"
MODEL_TYPE = "random_forest"

# Ventana TDE centrada: lags de -TDE_HALF_WINDOW a +TDE_HALF_WINDOW
# Ej: TDE_HALF_WINDOW=3 & TDE_TYPE="centered" → 7 lags × 5 features = 35 predictores
TDE_HALF_WINDOW = 0

TDE_TYPE = "past"

# --- NUEVO: Cálculo centralizado de lags ---
if TDE_TYPE == "centered":
    TDE_LAGS = list(range(-TDE_HALF_WINDOW, TDE_HALF_WINDOW + 1))
elif TDE_TYPE == "past":
    TDE_LAGS = list(range(-TDE_HALF_WINDOW, 1))
else:
    raise ValueError("TDE_TYPE debe ser 'centered' o 'past'")

# Mínimo de bloques únicos para que un sujeto sea utilizable
# (independiente de sesión — se cuenta sobre todo el CSV de esa dimensión)
MIN_BLOCKS = 3   # cambiá a 2 si querés criterio más permisivo

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# FUNCIÓN: FACTORY DE MODELOS
# ==============================
def get_model(model_type: str):
    """
    Devuelve un modelo sklearn listo para entrenar.
    Para modelos con hiperparámetros usa GridSearchCV interno
    sobre el train set → sin leakage.
    """
    if model_type == "ridge":
        return RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 50000.0], cv=5)
    
    elif model_type == "random_forest":
        base = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            "n_estimators": [100, 300],
            "max_depth":    [3, 5, None],
            "min_samples_leaf": [5, 10],
        }
        return GridSearchCV(base, param_grid, cv=5,
                            scoring="r2", n_jobs=-1, refit=True)

    elif model_type == "gradient_boosting":
        base = GradientBoostingRegressor(random_state=42)
        param_grid = {
            "n_estimators":  [100, 300],
            "max_depth":     [2, 3],
            "learning_rate": [0.05, 0.1],
        }
        return GridSearchCV(base, param_grid, cv=5,
                            scoring="r2", n_jobs=-1, refit=True)

    elif model_type == "svr":
        base = SVR()
        param_grid = {
            "C":       [0.1, 1.0, 10.0],
            "epsilon": [0.01, 0.1],
            "kernel":  ["rbf"],
        }
        return GridSearchCV(base, param_grid, cv=5,
                            scoring="r2", n_jobs=-1, refit=True)

    else:
        raise ValueError(f"MODEL_TYPE desconocido: '{model_type}'. "
                         f"Opciones: ridge, random_forest, gradient_boosting, svr")

# ==============================
# FUNCIÓN: VERIFICAR USABILIDAD
# ==============================
def check_subject_usability(df_subject: pd.DataFrame, subject_id) -> dict:
    """
    Criterio: cantidad de bloques únicos (Session_Block combinado)
    en el CSV de esta dimensión. No importa en qué sesión estén.
    """
    # Identificador único de bloque = Session + Block
    df_subject = df_subject.copy()
    df_subject["block_id"] = (
        df_subject["Session"].str.upper() + "_" + df_subject["Block"].astype(str)
    )
    n_blocks = df_subject["block_id"].nunique()

    if n_blocks < MIN_BLOCKS:
        return {
            "usable": False,
            "n_blocks": n_blocks,
            "reason": f"Solo {n_blocks} bloque(s) únicos (mínimo requerido: {MIN_BLOCKS})",
        }

    return {"usable": True, "n_blocks": n_blocks, "reason": "OK"}


# ==============================
# FUNCIÓN: CONSTRUIR TDE
# ==============================
def build_tde(df: pd.DataFrame, features: list, lags: list) -> pd.DataFrame:
    """
    Para cada feature EDA genera columnas  <feature>_lag<n>
    con n en [-half_window, ..., 0, ..., +half_window].

    El desplazamiento se hace DENTRO de cada grupo
    Subject-Session-Block-Stimulus para no contaminar series independientes.
    Los NAs en los bordes de ventana se dejan para eliminarlos luego en el CV.
    """
    group_cols = ["Subject", "Session", "Block", "Stimulus"]

    lag_frames = []

    for lag in lags:
        shifted = (
            df.groupby(group_cols)[features]
            .shift(-lag)   # shift(-lag): lag<0 → trae pasado; lag>0 → trae futuro
            .rename(columns={f: f"{f}_lag{lag}" for f in features})
        )
        lag_frames.append(shifted)

    df_tde = pd.concat([df.drop(columns=features)] + lag_frames, axis=1)
    return df_tde


# ==============================
# FUNCIÓN: LEAVE-ONE-BLOCK-OUT CV
# ==============================
def run_lobo_cv(
    df_subject: pd.DataFrame,
    subject_id,
    dimension_name: str,
    tde_cols: list,
) -> pd.DataFrame | None:
    """
    Folds = bloques únicos (Session_Block).
    En cada fold:
      - test  = ese bloque
      - train = todos los demás bloques del sujeto

    Escalado: media y SD del train, aplicados a test (sin leakage).

    Por fold se ajustan DOS modelos:
      1. sklearn LinearRegression  → métricas de predicción (RMSE, MAE, R²)
      2. statsmodels OLS           → coeficientes, p-valores, intervalos de confianza
    """
    df_subject = df_subject.copy()
    df_subject["fold_id"] = (
        df_subject["Session"].str.upper() + "_" + df_subject["Block"].astype(str)
    )

    folds = sorted(df_subject["fold_id"].unique())
    fold_results = []

    for fold in folds:
        df_test  = df_subject[df_subject["fold_id"] == fold]
        df_train = df_subject[df_subject["fold_id"] != fold]

        # Seleccionar columnas relevantes y eliminar NAs (bordes TDE)
        # Definir target variable según la dimensión
        target_var = "Report_Mean_Raw" if dimension_name == "valence" else "Report_Mean"

        # Seleccionar columnas relevantes y eliminar NAs (bordes TDE)
        cols_needed = [target_var] + tde_cols
        train_clean = df_train[cols_needed].dropna()
        test_clean  = df_test[cols_needed].dropna()

        if len(train_clean) < 10 or len(test_clean) < 2:
            print(
                f"    ⚠️  Fold {fold}: datos insuficientes "
                f"(train={len(train_clean)}, test={len(test_clean)}), saltando"
            )
            continue

        X_train = train_clean[tde_cols].values
        y_train = train_clean[target_var].values
        X_test  = test_clean[tde_cols].values
        y_test  = test_clean[target_var].values

        # ── Escalado (sin leakage) ──────────────────────────────────────────
        scaler  = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        # ── 1. sklearn: métricas de predicción ─────────────────────────────
        # RidgeCV elige automáticamente el mejor alpha por CV interna sobre el train set
        sk_model = get_model(MODEL_TYPE)
        sk_model.fit(X_train_sc, y_train)
        preds = sk_model.predict(X_test_sc)
        preds = np.clip(preds, -1.0, 1.0)

        rmse_val = np.sqrt(mean_squared_error(y_test, preds))
        mae_val  = mean_absolute_error(y_test, preds)
        r2_val   = r2_score(y_test, preds)

        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train_sc, y_train)
        dummy_preds = np.clip(dummy.predict(X_test_sc), -1.0, 1.0)

        dummy_rmse = np.sqrt(mean_squared_error(y_test, dummy_preds))
        dummy_mae  = mean_absolute_error(y_test, dummy_preds)
        dummy_r2   = r2_score(y_test, dummy_preds)

        # ── 2. statsmodels: coeficientes e inferencia (ajuste en train) ────
        X_train_sm = sm.add_constant(X_train_sc, has_constant="add")
        sm_model   = sm.OLS(y_train, X_train_sm).fit()

        # Extraer coeficientes (excluimos la constante para el resumen por fold)
        coef_names = ["const"] + tde_cols
        ci = sm_model.conf_int()
        ci_lower = ci[:, 0] if isinstance(ci, np.ndarray) else ci.iloc[:, 0].values
        ci_upper = ci[:, 1] if isinstance(ci, np.ndarray) else ci.iloc[:, 1].values

        coef_df = pd.DataFrame(
            {
                "Feature":  coef_names,
                "Beta":     sm_model.params,
                "SE":       sm_model.bse,
                "t":        sm_model.tvalues,
                "P_Value":  sm_model.pvalues,
                "CI_Lower": ci_lower,
                "CI_Upper": ci_upper,
            }
        )
        coef_df.insert(0, "Subject",   subject_id)
        coef_df.insert(1, "Dimension", dimension_name)
        coef_df.insert(2, "Fold",      fold)

        print(
            f"    Fold {fold:<6} | "
            f"n_train={len(train_clean):>4} | n_test={len(test_clean):>3} | "
            f"RMSE={rmse_val:.4f} | MAE={mae_val:.4f} | R²={r2_val:.4f}"
        )

        print(
            f"    {'[Dummy]':<12} | "
            f"{'':>12} | {'':>10} | "
            f"RMSE={dummy_rmse:.4f} | MAE={dummy_mae:.4f} | R²={dummy_r2:.4f}"
        )

        fold_results.append(
            {
                "Subject":         subject_id,
                "Dimension":       dimension_name,
                "Fold":            fold,
                "N_Train":         len(train_clean),
                "N_Test":          len(test_clean),
                "RMSE":            rmse_val,
                "MAE":             mae_val,
                "R2":              r2_val,
                "R2_train":        sm_model.rsquared,         # R² en train (statsmodels)
                "AdjR2_train":     sm_model.rsquared_adj,
                "F_stat":          sm_model.fvalue,
                "F_pvalue":        sm_model.f_pvalue,
                "Model_Type":  MODEL_TYPE,
                "Best_Params": str(sk_model.best_params_) if hasattr(sk_model, "best_params_") else str(MODEL_TYPE),
                "_coef_df":        coef_df,                   # guardamos para exportar aparte
                "Dummy_RMSE": dummy_rmse,
                "Dummy_MAE":  dummy_mae,
                "Dummy_R2":   dummy_r2,
            }
        )

    if not fold_results:
        return None, None

    # Separar métricas de coeficientes
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

    # ── Construir TDE sobre todos los datos ─────────────────────────────────
    print(f"\n🔧 Construyendo TDE (tipo={TDE_TYPE}, half_window={TDE_HALF_WINDOW})...")
    df_tde = build_tde(df_all, EDA_FEATURES, TDE_LAGS)

    tde_cols = [
        f"{feat}_lag{lag}"
        for feat in EDA_FEATURES
        for lag in TDE_LAGS
    ]

    # ── Iterar por sujeto ────────────────────────────────────────────────────
    usability_log  = []
    all_metrics    = []
    all_coefs      = []

    for subj in subjects:
        print(f"\n👤 Sujeto: {subj}")
        df_subj = df_tde[df_tde["Subject"] == subj].copy()

        check = check_subject_usability(df_subj, subj)
        usability_log.append(
            {
                "Subject":   subj,
                "Dimension": dimension_name,
                "N_Blocks":  check["n_blocks"],
                "Usable":    check["usable"],
                "Reason":    check["reason"],
            }
        )

        if not check["usable"]:
            print(f"  ⛔ Descartado → {check['reason']}")
            continue

        print(f"  ✅ Sujeto apto ({check['n_blocks']} bloques) → corriendo LOBO-CV...")
        metrics_df, coefs_df = run_lobo_cv(df_subj, subj, dimension_name, tde_cols)

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
            N_Folds       = ("Fold",       "count"),
            Mean_RMSE     = ("RMSE",       "mean"),
            SD_RMSE       = ("RMSE",       "std"),
            Mean_MAE      = ("MAE",        "mean"),
            SD_MAE        = ("MAE",        "std"),
            Mean_R2       = ("R2",         "mean"),
            SD_R2         = ("R2",         "std"),
            Mean_DummyRMSE = ("Dummy_RMSE", "mean"),
            Mean_DummyMAE  = ("Dummy_MAE",  "mean"),
            Mean_DummyR2   = ("Dummy_R2",   "mean"),
        )
        .reset_index()
    )
    summary_df.to_csv(
        os.path.join(dim_output, f"subject_summary_{dimension_name}_{MODEL_TYPE}.csv"), index=False
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
        "Grand_Mean_DummyRMSE": summary_df["Mean_DummyRMSE"].mean(),
        "Grand_Mean_DummyMAE":  summary_df["Mean_DummyMAE"].mean(),
        "Grand_Mean_DummyR2":   summary_df["Mean_DummyR2"].mean(),
    }

    print(f"\n📈 RESUMEN GLOBAL ({dimension_name.upper()}):")
    for k, v in global_summary.items():
        print(f"   {k:<20}: {v}")

    print(f"\n💾 Resultados guardados en: {dim_output}")

    return {
        "results":    results_df,
        "coefs":      coefs_df,
        "summary":    summary_df,
        "global":     global_summary,
        "usability":  usability_df,
    }


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    n_predictors = len(TDE_LAGS) * len(EDA_FEATURES)
    print(f"🚀 Iniciando análisis predictivo (Objetivo 2) — {MODEL_TYPE.upper()} + TDE ({TDE_TYPE})")
    print(f"   TDE half-window : {TDE_HALF_WINDOW}")
    print(f"   Lags calculados : {TDE_LAGS}")
    print(f"   Predictores     : {len(TDE_LAGS)} ventanas × {len(EDA_FEATURES)} features = {n_predictors}")
    print(f"   Mínimo bloques  : {MIN_BLOCKS}")

    res_arousal = analyze_dimension(AROUSAL_CSV, "arousal")
    res_valence = analyze_dimension(VALENCE_CSV, "valence")

    # ── Tabla comparativa final ──────────────────────────────────────────────
    if res_arousal and res_valence:
        comparison = pd.DataFrame([res_arousal["global"], res_valence["global"]])
        comparison.to_csv(os.path.join(OUTPUT_DIR, "global_comparison.csv"), index=False)

        print("\n" + "=" * 60)
        print("🏁 COMPARACIÓN GLOBAL AROUSAL vs VALENCE")
        print("=" * 60)
        print(comparison.to_string(index=False))

    print("\n🎉 Finalizado.")