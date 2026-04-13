# ==============================
# LIBRERÍAS
# ==============================
library(tidyverse)
library(lme4)
library(lmerTest)
library(performance)
library(ggeffects)
library(flextable)

set.seed(42)

# ==============================
# CONFIGURACIÓN
# ==============================
INPUT_DIR <- "C:/Users/jeror/Desktop/Cocuco/campeones_analysis/results/epoch_features"
OUTPUT_DIR <- "C:/Users/jeror/Desktop/Cocuco/campeones_analysis/results/analysis_cross_dimensions"

VALENCE_CSV <- file.path(INPUT_DIR, "valence_epochs_10s.csv")
AROUSAL_CSV <- file.path(INPUT_DIR, "arousal_epochs_10s.csv")

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ==============================
# FUNCIÓN: PREPARAR Y MERGEAR DATOS
# ==============================
prepare_and_merge_data <- function(val_path, aro_path) {
  
  cat("🔄 Cargando y alineando datos de Valencia y Arousal...\n")
  
  # Cargar Valencia
  df_val <- read_csv(val_path, show_col_types = FALSE) %>%
    select(Subject, Stimulus, Epoch_Index, 
           Session_Val = Session, 
           Valence = Report_Mean)
  
  # Cargar Arousal
  df_aro <- read_csv(aro_path, show_col_types = FALSE) %>%
    select(Subject, Stimulus, Epoch_Index, 
           Session_Aro = Session, 
           Arousal = Report_Mean)
  
  # Inner Join: Alineamos usando el Epoch_Index original
  df_merged <- inner_join(
    df_val, df_aro, 
    by = c("Subject", "Stimulus", "Epoch_Index")
  ) %>%
    mutate(
      Order = ifelse(Session_Val < Session_Aro, "Valence_First", "Arousal_First"),
      Subject = as.factor(Subject),
      Stimulus = as.factor(Stimulus),
      Order = as.factor(Order)
    ) %>%
    drop_na(Valence, Arousal)
  
  cat("✅ Datos combinados. Total de épocas alineadas:", nrow(df_merged), "\n")
  return(df_merged)
}

# ==============================
# FUNCIÓN: PLOTEAR RESULTADOS
# ==============================
# Modificado para recibir "df" crudo y graficar geom_jitter
plot_marginal_effects <- function(model, df, predictor, target, output_folder) {
  
  # ggpredict extrae el efecto principal fijando los efectos aleatorios a 0 (promedio)
  pred_data <- ggpredict(model, terms = predictor)
  
  p <- ggplot() +
    # 1. Capa de datos crudos (Scatterplot transparente con Jitter)
    geom_jitter(
      data = df, 
      aes(x = .data[[predictor]], y = .data[[target]]), 
      alpha = 0.01,         # 99% de transparencia para evitar overplotting
      width = 0.05,         # Leve dispersión horizontal
      height = 0.05,        # Leve dispersión vertical
      color = "gray30"      # Gris oscuro para que la línea destaque más
    ) +
    # 2. Capa predictiva (Línea de regresión y Cinta de confianza)
    geom_line(
      data = pred_data, 
      aes(x = x, y = predicted), 
      color = "darkred", 
      linewidth = 1.2
    ) +
    geom_ribbon(
      data = pred_data, 
      aes(x = x, ymin = conf.low, ymax = conf.high), 
      alpha = 0.2, 
      fill = "darkred"
    ) +
    labs(
      title = paste("Relación Marginal:", target, "~", predictor),
      subtitle = "Línea roja = Efecto Fijo (95% CI) | Puntos grises = Datos crudos",
      x = paste("Reporte de", predictor),
      y = paste("Reporte de", target)
    ) +
    theme_minimal(base_size = 14)
  
  file_path <- file.path(output_folder, paste0("marginal_plot_", target, "_from_", predictor, ".png"))
  ggsave(file_path, plot = p, width = 7, height = 5, dpi = 300)
  cat("📈 Gráfico marginal guardado en:", file_path, "\n")
}

# ==============================
# FUNCIÓN: CORRER MODELO MIXTO
# ==============================
run_cross_model <- function(df, target_var, predictor_var) {
  
  cat(sprintf("\n🚀 Corriendo modelo: %s ~ %s\n", target_var, predictor_var))
  
  # Chequeo de seguridad para el factor Order
  if (length(unique(df$Order)) < 2) {
    formula_str <- paste0(
      target_var, " ~ ", predictor_var, " + ",
      "(1 | Subject) + ",
      "(1 | Stimulus) + ",
      "(1 | Subject:Stimulus)"
    )
    cat("⚠️ Nota: 'Order' tiene un solo nivel, se excluyó del modelo.\n")
  } else {
    formula_str <- paste0(
      target_var, " ~ ", predictor_var, " + Order + ",
      "(1 | Subject) + ",
      "(1 | Stimulus) + ",
      "(1 | Subject:Stimulus)"
    )
  }
  
  model <- tryCatch({
    lmer(as.formula(formula_str), data = df,
         control = lmerControl(optimizer = "bobyqa"))
  }, error = function(e) {
    message("❌ Error en modelo: ", e$message)
    return(NULL)
  })
  
  if (is.null(model)) return(NULL)
  
  # Resumen del modelo
  summary_model <- summary(model)
  
  # EXTRACCIÓN DETALLADA DE ESTADÍSTICOS (Añadido SE, df, t y CI)
  coefs <- summary_model$coefficients
  beta   <- coefs[predictor_var, "Estimate"]
  se     <- coefs[predictor_var, "Std. Error"]
  df_val <- coefs[predictor_var, "df"]
  t_val  <- coefs[predictor_var, "t value"]
  pval   <- coefs[predictor_var, "Pr(>|t|)"]
  
  # Cálculo rápido de Intervalos de Confianza 95% (Aproximación Wald)
  ci_lower <- beta - 1.96 * se
  ci_upper <- beta + 1.96 * se
  
  # R cuadrados
  r2_vals <- tryCatch(r2(model), error = function(e) NULL)
  
  sig <- ifelse(pval < 0.001, "***", ifelse(pval < 0.01, "**", ifelse(pval < 0.05, "*", "ns")))
  
  # Imprimir Reporte en Consola
  cat(sprintf("   👉 β = %.4f | SE = %.4f | 95%% CI = [%.4f, %.4f]\n", beta, se, ci_lower, ci_upper))
  cat(sprintf("   👉 t(%.1f) = %.3f | p = %.4f %s\n", df_val, t_val, pval, sig))
  cat(sprintf("   👉 R2 Marginal (Fijos) = %.3f | R2 Condicional (Todo) = %.3f\n", 
              ifelse(is.null(r2_vals), NA, r2_vals$R2_marginal),
              ifelse(is.null(r2_vals), NA, r2_vals$R2_conditional)))
  
  cat("\n   [Anova Type III - Efectos Fijos]:\n")
  print(anova(model))
  
  # Plotear (Pasando el df original a la función)
  plot_marginal_effects(model, df, predictor_var, target_var, OUTPUT_DIR)
  
  return(model)
}

# ==============================
# MAIN
# ==============================
cat("🌟 INICIANDO ANÁLISIS CRUZADO AROUSAL <-> VALENCIA 🌟\n")

# 1. Preparar datos
df_merged <- prepare_and_merge_data(VALENCE_CSV, AROUSAL_CSV)

# Guardar el dataset cruzado
write_csv(df_merged, file.path(OUTPUT_DIR, "merged_arousal_valence_epochs.csv"))

# 2. Correr modelos
model_v_a <- run_cross_model(df_merged, target_var = "Valence", predictor_var = "Arousal")

# 3. Guardar resultados resumidos en texto
sink(file.path(OUTPUT_DIR, "models_summary.txt"))
cat("==============================\n")
cat("MODELO 1: VALENCIA ~ AROUSAL\n")
cat("==============================\n")
if(!is.null(model_v_a)) print(summary(model_v_a))
sink()

cat("\n🎉 Análisis finalizado. Resultados guardados en:", OUTPUT_DIR, "\n")