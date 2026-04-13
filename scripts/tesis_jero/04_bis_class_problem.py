# ==============================
# LIBRERÍAS
# ==============================
import os
import warnings
import numpy as np
import pandas as pd

# Modelado
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")


# ==============================
# CONFIGURACIÓN
# ==============================
INPUT_DIR  = r"C:\Users\jeror\Desktop\Cocuco\campeones_analysis\results\epoch_features"
OUTPUT_DIR = r"C:\Users\jeror\Desktop\Cocuco\campeones_analysis\results\analysis_models_obj2_classification"

VALENCE_CSV = os.path.join(INPUT_DIR, "valence_epochs_10s.csv")
AROUSAL_CSV = os.path.join(INPUT_DIR, "arousal_epochs_10s.csv")

EDA_FEATURES = [
    "EDA_Phasic_Mean",
    "EDA_SMNA_AUC",
    "EDA_Tonic_Mean",
    "EDA_SCR_Peaks_Count",
    "EDA_SMNA_Peaks_Count",
]

# Opciones: "logistic", "random_forest", "svm"
MODEL_TYPE = "logistic"

# Ventana TDE (igual que en regresión)
TDE_HALF_WINDOW = 8
TDE_TYPE = "past"   # "centered" o "past"

if TDE_TYPE == "centered":
    TDE_LAGS = list(range(-TDE_HALF_WINDOW, TDE_HALF_WINDOW + 1))
elif TDE_TYPE == "past":
    TDE_LAGS = list(range(-TDE_HALF_WINDOW, 1))
else:
    raise ValueError("TDE_TYPE debe ser 'centered' o 'past'")

# Mínimo de bloques únicos para que un sujeto sea utilizable
MIN_BLOCKS = 3

# Umbral de binarización: "subject_median" | "zero" | "global_median"
BINARIZATION = "subject_median"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# FUNCIÓN: FACTORY DE MODELOS
# ==============================
def get_model(model_type: str):
    """
    Devuelve un clasificador sklearn listo para entrenar.
    La selección de hiperparámetros se hace por CV interna sobre el train set
    → sin leakage.
    """
    if model_type == "logistic":
        # LogisticRegressionCV hace búsqueda de C internamente (equivalente a Ridge/L2)
        return LogisticRegressionCV(
            Cs=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            cv=5,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            random_state=42,
            scoring="balanced_accuracy",
        )

    elif model_type == "random_forest":
        base = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            "n_estimators":     [100, 300],
            "max_depth":        [3, 5, None],
            "min_samples_leaf": [5, 10],
        }
        return GridSearchCV(
            base, param_grid, cv=5,
            scoring="balanced_accuracy", n_jobs=-1, refit=True
        )

    elif model_type == "svm":
        base = SVC(probability=True, random_state=42)
        param_grid = {
            "C":      [0.1, 1.0, 10.0],
            "kernel": ["rbf"],
            "gamma":  ["scale", "auto"],
        }
        return GridSearchCV(
            base, param_grid, cv=5,
            scoring="balanced_accuracy", n_jobs=-1, refit=True
        )

    else:
        raise ValueError(
            f"MODEL_TYPE desconocido: '{model_type}'. "
            "Opciones: logistic, random_forest, svm"
        )


# ==============================
# FUNCIÓN: BINARIZAR TARGET
# ==============================
def binarize_target(
    y_train: np.ndarray,
    y_test: np.ndarray,
    strategy: str,
    global_median: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Convierte scores continuos (-1 a 1) en clases binarias (0=low, 1=high).

    Estrategias:
      - "subject_median" : umbral = mediana de y_train (adaptativo al sujeto/fold)
      - "zero"           : umbral fijo en 0
      - "global_median"  : umbral = mediana global del dataset (pasada como argumento)

    Retorna (y_train_bin, y_test_bin, threshold_used).
    """
    if strategy == "subject_median":
        threshold = np.median(y_train)
    elif strategy == "zero":
        threshold = 0.0
    elif strategy == "global_median":
        if global_median is None:
            raise ValueError("global_median debe proveerse para strategy='global_median'")
        threshold = global_median
    else:
        raise ValueError(f"BINARIZATION desconocida: '{strategy}'")

    y_train_bin = (y_train > threshold).astype(int)
    y_test_bin  = (y_test  > threshold).astype(int)

    return y_train_bin, y_test_bin, threshold


# ==============================
# FUNCIÓN: VERIFICAR USABILIDAD
# ==============================
def check_subject_usability(df_subject: pd.DataFrame, subject_id) -> dict:
    """
    Criterio: cantidad de bloques únicos (Session_Block combinado).
    Mínimo MIN_BLOCKS bloques para poder hacer LOBO-CV.
    """
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
    Genera columnas <feature>_lag<n> dentro de cada grupo
    Subject-Session-Block-Stimulus para no contaminar series independientes.
    """
    group_cols = ["Subject", "Session", "Block", "Stimulus"]
    lag_frames = []

    for lag in lags:
        shifted = (
            df.groupby(group_cols)[features]
            .shift(-lag)
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
    global_median: float | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Folds = bloques únicos (Session_Block).
    En cada fold:
      - test  = ese bloque
      - train = todos los demás bloques del sujeto

    Escalado: media y SD del train, aplicados a test (sin leakage).
    Binarización: umbral calculado SOLO sobre y_train (sin leakage).

    Métricas reportadas: Accuracy, Balanced Accuracy, F1 macro, ROC-AUC.
    """
    df_subject = df_subject.copy()
    df_subject["fold_id"] = (
        df_subject["Session"].str.upper() + "_" + df_subject["Block"].astype(str)
    )

    folds = sorted(df_subject["fold_id"].unique())
    fold_results  = []
    fold_cm_dfs   = []

    # Variable target según dimensión
    target_var = "Report_Mean_Raw" if dimension_name == "valence" else "Report_Mean"

    for fold in folds:
        df_test  = df_subject[df_subject["fold_id"] == fold]
        df_train = df_subject[df_subject["fold_id"] != fold]

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

        # ── Binarización (umbral calculado solo con y_train) ────────────────
        y_train_bin, y_test_bin, threshold = binarize_target(
            y_train, y_test, BINARIZATION, global_median
        )

        # Verificar que haya al menos 2 clases en train y en test
        if len(np.unique(y_train_bin)) < 2:
            print(
                f"    ⚠️  Fold {fold}: solo 1 clase en train "
                f"(umbral={threshold:.3f}), saltando"
            )
            continue
        if len(np.unique(y_test_bin)) < 2:
            # ROC-AUC no definido con 1 sola clase en test; se continúa con advertencia
            roc_auc_available = False
        else:
            roc_auc_available = True

        # ── Escalado (sin leakage) ───────────────────────────────────────────
        scaler      = StandardScaler()
        X_train_sc  = scaler.fit_transform(X_train)
        X_test_sc   = scaler.transform(X_test)

        # ── Entrenamiento ────────────────────────────────────────────────────
        model = get_model(MODEL_TYPE)
        model.fit(X_train_sc, y_train_bin)
        preds      = model.predict(X_test_sc)
        preds_prob = model.predict_proba(X_test_sc)[:, 1]

        # ── Métricas ─────────────────────────────────────────────────────────
        acc      = accuracy_score(y_test_bin, preds)
        bal_acc  = balanced_accuracy_score(y_test_bin, preds)
        f1_mac   = f1_score(y_test_bin, preds, average="macro", zero_division=0)
        roc_auc  = (
            roc_auc_score(y_test_bin, preds_prob)
            if roc_auc_available else np.nan
        )

        # ── Confusion matrix ─────────────────────────────────────────────────
        cm = confusion_matrix(y_test_bin, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        # ── Best params ──────────────────────────────────────────────────────
        best_params = (
            str(model.best_params_)
            if hasattr(model, "best_params_")
            else f"C={model.C_[0]:.4f}" if hasattr(model, "C_") else MODEL_TYPE
        )

        print(
            f"    Fold {fold:<6} | "
            f"n_train={len(train_clean):>4} | n_test={len(test_clean):>3} | "
            f"thr={threshold:+.3f} | "
            f"Acc={acc:.3f} | BalAcc={bal_acc:.3f} | "
            f"F1={f1_mac:.3f} | AUC={roc_auc:.3f}"
        )

        fold_results.append(
            {
                "Subject":        subject_id,
                "Dimension":      dimension_name,
                "Fold":           fold,
                "N_Train":        len(train_clean),
                "N_Test":         len(test_clean),
                "Threshold":      threshold,
                "N_Train_Low":    int((y_train_bin == 0).sum()),
                "N_Train_High":   int((y_train_bin == 1).sum()),
                "N_Test_Low":     int((y_test_bin  == 0).sum()),
                "N_Test_High":    int((y_test_bin  == 1).sum()),
                "Accuracy":       acc,
                "Balanced_Acc":   bal_acc,
                "F1_Macro":       f1_mac,
                "ROC_AUC":        roc_auc,
                "TN":             tn,
                "FP":             fp,
                "FN":             fn,
                "TP":             tp,
                "Model_Type":     MODEL_TYPE,
                "Best_Params":    best_params,
            }
        )

    if not fold_results:
        return None, None

    metrics_df = pd.DataFrame(fold_results)
    return metrics_df


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

    # ── Mediana global (por si se usa esa estrategia) ────────────────────────
    target_var    = "Report_Mean_Raw" if dimension_name == "valence" else "Report_Mean"
    global_median = df_all[target_var].median()
    print(f"   Mediana global ({target_var}): {global_median:.4f}")

    # ── TDE ──────────────────────────────────────────────────────────────────
    print(f"\n🔧 Construyendo TDE (tipo={TDE_TYPE}, half_window={TDE_HALF_WINDOW})...")
    df_tde = build_tde(df_all, EDA_FEATURES, TDE_LAGS)

    tde_cols = [
        f"{feat}_lag{lag}"
        for feat in EDA_FEATURES
        for lag in TDE_LAGS
    ]

    # ── Iterar por sujeto ────────────────────────────────────────────────────
    usability_log = []
    all_metrics   = []

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
        metrics_df = run_lobo_cv(
            df_subj, subj, dimension_name, tde_cols, global_median
        )

        if metrics_df is not None:
            all_metrics.append(metrics_df)

    # ── Guardar log de usabilidad ────────────────────────────────────────────
    usability_df = pd.DataFrame(usability_log)
    usability_df.to_csv(
        os.path.join(dim_output, f"usability_log_{dimension_name}.csv"), index=False
    )
    n_usable = usability_df["Usable"].sum()
    print(f"\n📋 Sujetos aptos:       {n_usable}")
    print(f"📋 Sujetos descartados: {len(subjects) - n_usable}")

    if not all_metrics:
        print(f"⚠️  Sin resultados para dimensión: {dimension_name}")
        return None

    # ── Resultados por fold ──────────────────────────────────────────────────
    results_df = pd.concat(all_metrics, ignore_index=True)
    results_df.to_csv(
        os.path.join(dim_output, f"fold_results_{dimension_name}.csv"), index=False
    )

    # ── Resumen por sujeto ───────────────────────────────────────────────────
    summary_df = (
        results_df.groupby(["Subject", "Dimension"])
        .agg(
            N_Folds       = ("Fold",         "count"),
            Mean_Acc      = ("Accuracy",     "mean"),
            SD_Acc        = ("Accuracy",     "std"),
            Mean_BalAcc   = ("Balanced_Acc", "mean"),
            SD_BalAcc     = ("Balanced_Acc", "std"),
            Mean_F1       = ("F1_Macro",     "mean"),
            SD_F1         = ("F1_Macro",     "std"),
            Mean_AUC      = ("ROC_AUC",      "mean"),
            SD_AUC        = ("ROC_AUC",      "std"),
            Mean_Threshold= ("Threshold",    "mean"),
        )
        .reset_index()
    )
    summary_df.to_csv(
        os.path.join(dim_output, f"subject_summary_{dimension_name}.csv"), index=False
    )

    # ── Resumen global ───────────────────────────────────────────────────────
    global_summary = {
        "Dimension":          dimension_name,
        "N_Subjects":         len(summary_df),
        "Binarization":       BINARIZATION,
        "Grand_Mean_Acc":     summary_df["Mean_Acc"].mean(),
        "Grand_SD_Acc":       summary_df["Mean_Acc"].std(),
        "Grand_Mean_BalAcc":  summary_df["Mean_BalAcc"].mean(),
        "Grand_SD_BalAcc":    summary_df["Mean_BalAcc"].std(),
        "Grand_Mean_F1":      summary_df["Mean_F1"].mean(),
        "Grand_SD_F1":        summary_df["Mean_F1"].std(),
        "Grand_Mean_AUC":     summary_df["Mean_AUC"].mean(),
        "Grand_SD_AUC":       summary_df["Mean_AUC"].std(),
    }

    print(f"\n📈 RESUMEN GLOBAL ({dimension_name.upper()}):")
    for k, v in global_summary.items():
        print(f"   {k:<25}: {v}")

    print(f"\n💾 Resultados guardados en: {dim_output}")

    return {
        "results":   results_df,
        "summary":   summary_df,
        "global":    global_summary,
        "usability": usability_df,
    }


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    n_predictors = len(TDE_LAGS) * len(EDA_FEATURES)
    print(f"🚀 Iniciando análisis de CLASIFICACIÓN BINARIA (Objetivo 2) — {MODEL_TYPE.upper()} + TDE ({TDE_TYPE})")
    print(f"   Binarización    : {BINARIZATION}")
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