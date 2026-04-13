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
from sklearn.model_selection import TimeSeriesSplit

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
    "EDA_Tonic_Mean",
    "EDA_Tonic_Std",
    "EDA_Phasic_Mean",
    "EDA_Phasic_Std",
    "EDA_SMNA_AUC",
]

# Opciones: "ridge", "random_forest", "svr"
MODEL_TYPE = "ridge"

# Ventana TDE centrada: lags de -TDE_HALF_WINDOW a +TDE_HALF_WINDOW
# Ej: TDE_HALF_WINDOW=3 & TDE_TYPE="centered" → 7 lags × 5 features = 35 predictores
TDE_HALF_WINDOW = 3

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
    cv = TimeSeriesSplit(n_splits=3)

    if model_type == "ridge":
        # Rango alto de alphas → busca regularización fuerte
        return RidgeCV(
            alphas=[1e2, 1e3, 1e4, 1e5, 1e6],
            cv=cv  # cv=3 es más estable con pocos datos
        )

    elif model_type == "random_forest":
        base = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            "n_estimators":     [200, 400],
            "max_depth":        [2, 3],        # sin None: evita árboles profundos
            "min_samples_leaf": [10, 20],      # mínimo de muestras por hoja
            "max_features":     [0.5, "sqrt"], # limita features por split
        }
        return GridSearchCV(base, param_grid, cv=cv,
                            scoring="neg_mean_squared_error",
                            n_jobs=-1, refit=True)

    elif model_type == "gradient_boosting":
        base = GradientBoostingRegressor(random_state=42)
        param_grid = {
            "n_estimators":     [200, 400],
            "max_depth":        [2, 3],
            "learning_rate":    [0.01, 0.05],  # más lento → menos overfitting
            "subsample":        [0.6, 0.8],    # bagging estocástico
            "min_samples_leaf": [10, 20],
        }
        return GridSearchCV(base, param_grid, cv=cv,
                            scoring="neg_mean_squared_error",
                            n_jobs=-1, refit=True)

    elif model_type == "svr":
        base = SVR()
        param_grid = {
            "C":       [0.001, 0.01, 0.1],  # C bajo → más regularización
            "epsilon": [0.1, 0.2],
            "kernel":  ["rbf"],
        }
        return GridSearchCV(base, param_grid, cv=cv,
                            scoring="neg_mean_squared_error",
                            n_jobs=-1, refit=True)

    else:
        raise ValueError(f"MODEL_TYPE desconocido: '{model_type}'.")

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


        for col in [target_var] + tde_cols:
            block_means = df_train.groupby("fold_id")[col].transform("mean")
            df_train = df_train.copy()
            df_train[col] = df_train[col] - block_means

        # Para el test: restar la media del propio bloque de test
        # (esto es válido porque no usamos info del train para centrar el test)
        for col in [target_var] + tde_cols:
            test_mean = df_test[col].mean()
            df_test = df_test.copy()
            df_test[col] = df_test[col] - test_mean
            
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

        # Dummies de bloque (fit en train, transform en test → sin leakage)
        block_dummies_train = pd.get_dummies(df_train["fold_id"], prefix="block", drop_first=True)
        block_dummies_test  = pd.get_dummies(df_test["fold_id"],  prefix="block", drop_first=True)

        # Alinear columnas: el bloque de test puede no tener todas las categorías del train
        block_dummies_test = block_dummies_test.reindex(
            columns=block_dummies_train.columns, fill_value=0
        )

        # Filtrar solo las filas que sobrevivieron al dropna anterior
        block_dummies_train = block_dummies_train.loc[train_clean.index]
        block_dummies_test  = block_dummies_test.loc[test_clean.index]

        X_train = np.hstack([train_clean[tde_cols].values, block_dummies_train.values])
        y_train = train_clean[target_var].values
        X_test  = np.hstack([test_clean[tde_cols].values,  block_dummies_test.values])
        y_test  = test_clean[target_var].values

        from scipy.stats import spearmanr

        def select_features_by_correlation(X_train, y_train, feature_names, threshold=0.1):
            """
            Filtra columnas con |correlación de Spearman| >= threshold con el target.
            Se calcula SOLO sobre train → sin leakage.
            """
            selected = []
            for i, name in enumerate(feature_names):
                corr, _ = spearmanr(X_train[:, i], y_train)
                if abs(corr) >= threshold:
                    selected.append(i)
            
            # Si ninguna pasa el umbral, conservar las top-3 para no quedar sin predictores
            if len(selected) == 0:
                corrs = [abs(spearmanr(X_train[:, i], y_train)[0]) for i in range(X_train.shape[1])]
                selected = sorted(range(len(corrs)), key=lambda i: corrs[i], reverse=True)[:3]
            
            return selected

        # ── Escalado (sin leakage) ─────────────────────────────────────────
        n_tde      = len(tde_cols)
        X_train_tde = X_train[:, :n_tde]
        X_train_dum = X_train[:, n_tde:]
        X_test_tde  = X_test[:, :n_tde]
        X_test_dum  = X_test[:, n_tde:]

        scaler = StandardScaler()
        X_train_tde_sc = scaler.fit_transform(X_train_tde)
        X_test_tde_sc  = scaler.transform(X_test_tde)

        # Dummies no se escalan (ya son 0/1)

        # ── Selección de features (solo sobre columnas TDE, no sobre dummies) ──
        selected_idx = select_features_by_correlation(
            X_train_tde_sc, y_train, tde_cols, threshold=0.10
        )
        X_train_tde_sel = X_train_tde_sc[:, selected_idx]
        X_test_tde_sel  = X_test_tde_sc[:, selected_idx]
        selected_cols   = [tde_cols[i] for i in selected_idx]

        # Recombinar: features TDE seleccionadas + dummies
        X_train_sc = np.hstack([X_train_tde_sel, X_train_dum])
        X_test_sc  = np.hstack([X_test_tde_sel,  X_test_dum])

        # ── 1. sklearn: métricas de predicción ────────────────────────────
        sk_model = get_model(MODEL_TYPE)
        sk_model.fit(X_train_sc, y_train)
        fold_results = []
        rf_importance_rows = []

        if MODEL_TYPE == "random_forest" and hasattr(sk_model, "best_estimator_"):
            rf_best = sk_model.best_estimator_
            dummy_col_names_rf = list(block_dummies_train.columns)
            all_feature_names_rf = selected_cols + dummy_col_names_rf
            importances = rf_best.feature_importances_
            n = min(len(all_feature_names_rf), len(importances))
            for fname, imp in zip(all_feature_names_rf[:n], importances[:n]):
                rf_importance_rows.append({
                    "Subject":    subject_id,
                    "Dimension":  dimension_name,
                    "Fold":       fold,
                    "Feature":    fname,
                    "Importance": imp,
                })
        preds = np.clip(sk_model.predict(X_test_sc), -1.0, 1.0)

        rmse_val = np.sqrt(mean_squared_error(y_test, preds))
        mae_val  = mean_absolute_error(y_test, preds)
        r2_val   = r2_score(y_test, preds)

        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train_sc, y_train)
        dummy_preds = np.clip(dummy.predict(X_test_sc), -1.0, 1.0)
        dummy_rmse  = np.sqrt(mean_squared_error(y_test, dummy_preds))
        dummy_mae   = mean_absolute_error(y_test, dummy_preds)
        dummy_r2    = r2_score(y_test, dummy_preds)

        # ── 2. statsmodels: coeficientes e inferencia ─────────────────────
        X_train_sm = sm.add_constant(X_train_sc, has_constant="add")
        sm_model   = sm.OLS(y_train, X_train_sm).fit()

        # coef_names ahora sí coincide en longitud con sm_model.params
        dummy_col_names = list(block_dummies_train.columns)
        coef_names = ["const"] + selected_cols + dummy_col_names

        ci = sm_model.conf_int()
        ci_lower = ci[:, 0] if isinstance(ci, np.ndarray) else ci.iloc[:, 0].values
        ci_upper = ci[:, 1] if isinstance(ci, np.ndarray) else ci.iloc[:, 1].values

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
                "N_Features_Selected": len(selected_idx),
                "Features_Selected":   str(selected_cols),
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

    if rf_importance_rows:
        rf_imp_df = pd.DataFrame(rf_importance_rows)
    else:
        rf_imp_df = pd.DataFrame(columns=["Subject", "Dimension", "Fold", "Feature", "Importance"])

    return metrics_df, coefs_df, rf_imp_df


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
    all_rf_importances = []   # ← agregar esta línea

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
        metrics_df, coefs_df, rf_imp_df = run_lobo_cv(df_subj, subj, dimension_name, tde_cols)

        if metrics_df is not None:
            all_metrics.append(metrics_df)
            all_coefs.append(coefs_df)
            if not rf_imp_df.empty:              # ← agregar estas dos líneas
                all_rf_importances.append(rf_imp_df)

    # ── Guardar log de usabilidad ────────────────────────────────────────────
    usability_df = pd.DataFrame(usability_log)
    usability_df.to_csv(
        os.path.join(dim_output, f"usability_log_{dimension_name}_{MODEL_TYPE}.csv"), index=False
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
        os.path.join(dim_output, f"fold_results_{dimension_name}_{MODEL_TYPE}.csv"), index=False
    )

    # ── Coeficientes por fold ────────────────────────────────────────────────
    coefs_df = pd.concat(all_coefs, ignore_index=True)
    coefs_df.to_csv(
        os.path.join(dim_output, f"coefficients_{dimension_name}_{MODEL_TYPE}.csv"), index=False
    )

    if all_rf_importances:
        rf_all_df = pd.concat(all_rf_importances, ignore_index=True)
        rf_all_df.to_csv(
            os.path.join(dim_output, f"rf_importances_{dimension_name}_{MODEL_TYPE}.csv"),
            index=False
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
        comparison.to_csv(os.path.join(OUTPUT_DIR, f"global_comparison_{MODEL_TYPE}.csv"), index=False)

        print("\n" + "=" * 60)
        print("🏁 COMPARACIÓN GLOBAL AROUSAL vs VALENCE")
        print("=" * 60)
        print(comparison.to_string(index=False))

    print("\n🎉 Finalizado.")