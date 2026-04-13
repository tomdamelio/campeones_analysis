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
# Directorios de entrada y salida (ajústalos si es necesario)
INPUT_DIR <- "C:/Users/jeror/Desktop/Cocuco/campeones_analysis/results/epoch_features"
OUTPUT_DIR <- "C:/Users/jeror/Desktop/Cocuco/campeones_analysis/results/analysis_models_obj3_valence_raw"

# Archivo de datos a utilizar
VALENCE_CSV <- file.path(INPUT_DIR, "valence_epochs_10s.csv")

# Variables predictoras (features de EDA)
EDA_FEATURES <- c(
  "EDA_Phasic_Mean",
  "EDA_SMNA_AUC",
  "EDA_Tonic_Mean",
  "EDA_SCR_Peaks_Count",
  "EDA_SMNA_Peaks_Count"
)

# Lags a probar: de -7 a +7 (0 incluido)
# Negativo: EDA de t+lag predice reporte en t (EDA ocurre ANTES del reporte)
# Positivo: EDA de t+lag predice reporte en t (EDA ocurre DESPUÉS del reporte)
LAGS_TO_TEST <- -7:7

# Crear directorio de salida si no existe
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ==============================
# FUNCIÓN: APLICAR DESPLAZAMIENTO (LAG)
# ==============================
# El reporte siempre queda fijo en t=0. Desplazamos las señales de EDA.
#
#   lag < 0 → EDA ANTERIOR al reporte
#             Ej: lag=-3 → emparejamos EDA de época t-3 con reporte de época t
#             Implementación: lag() con n=abs(lag_val) trae valores pasados de EDA
#
#   lag = 0 → Simultáneo, sin desplazamiento
#
#   lag > 0 → EDA POSTERIOR al reporte
#             Ej: lag=+3 → emparejamos EDA de época t+3 con reporte de época t
#             Implementación: lead() con n=lag_val trae valores futuros de EDA

apply_lag <- function(df, eda_features, lag_val) {
  if (lag_val == 0) return(df)
  
  df %>%
    arrange(Subject, Session, Block, Stimulus, Epoch_Index) %>%
    group_by(Subject, Session, Block, Stimulus) %>%
    mutate(across(
      all_of(eda_features),
      ~ if (lag_val < 0) {
        lag(.x, abs(lag_val))   # EDA pasada predice reporte actual
      } else {
        lead(.x, lag_val)       # EDA futura predice reporte actual
      }
    )) %>%
    ungroup()
}

# ==============================
# FUNCIÓN: CREAR TABLA DE RESULTADOS ESTILO APA
# ==============================
create_apa_table <- function(results_df, dimension_name, output_folder) {
  
  df <- results_df %>%
    mutate(
      Sig = case_when(
        P_Value < 0.001 ~ "***",
        P_Value < 0.01  ~ "**",
        P_Value < 0.05  ~ "*",
        TRUE            ~ ""
      ),
      Beta_Sig = sprintf("%.3f%s", Beta, Sig)
    ) %>%
    select(Predictor, Lag, Beta_Sig) %>%
    pivot_wider(
      names_from  = Lag,
      values_from = Beta_Sig
    ) %>%
    arrange(Predictor)
  
  # Reordenar columnas de lag de -7 a +7
  lag_cols <- as.character(sort(unique(results_df$Lag)))
  df <- df %>% select(Predictor, all_of(lag_cols))
  
  csv_file_path <- file.path(output_folder, paste0("APA_table_", dimension_name, ".csv"))
  write_csv(df, csv_file_path)
  
  ft <- flextable(df)
  ft <- autofit(ft)
  
  docx_file_path <- file.path(output_folder, paste0("APA_table_", dimension_name, ".docx"))
  save_as_docx(ft, path = docx_file_path)
  
  cat("📋 Tabla APA guardada en:", csv_file_path, "\n")
  
  return(df)
}

# ==============================
# FUNCIÓN: PLOTEAR HEATMAP DE COEFICIENTES BETA
# ==============================
create_heatmap <- function(results_df, dimension_name, output_folder) {
  
  df <- results_df %>%
    mutate(
      Sig = case_when(
        P_Value < 0.001 ~ "***",
        P_Value < 0.01  ~ "**",
        P_Value < 0.05  ~ "*",
        TRUE            ~ ""
      ),
      Label = ifelse(
        Sig != "",
        sprintf("%.3f\n%s", Beta, Sig),
        sprintf("%.3f", Beta)
      ),
      Lag_factor = factor(Lag, levels = sort(unique(Lag)))
    )
  
  p <- ggplot(df, aes(x = Lag_factor, y = Predictor, fill = Beta)) +
    geom_tile(color = "white", linewidth = 0.8) +
    geom_text(aes(label = Label), size = 2.8, lineheight = 0.9) +
    
    # Línea vertical en lag=0
    geom_vline(
      xintercept = which(levels(df$Lag_factor) == "0") - 0.5,
      color = "black", linewidth = 1, linetype = "dashed"
    ) +
    
    geom_vline(
      xintercept = which(levels(df$Lag_factor) == "1") - 0.5,
      color = "black", linewidth = 1, linetype = "dashed"
    ) +
    
    scale_fill_gradient2(
      low      = "blue",
      mid      = "white",
      high     = "red",
      midpoint = 0
    ) +
    
    scale_x_discrete(
      labels = function(x) {
        x_int <- as.integer(x)
        ifelse(x_int == 0,  "0\n(simultáneo)",
               ifelse(x_int  < 0,  paste0(x,    "\n(EDA antes)"),
                      paste0("+", x, "\n(EDA después)")))
      }
    ) +
    
    labs(
      title    = paste("Heatmap de Betas -", toupper(dimension_name)),
      subtitle = "Lags negativos: EDA precede al reporte | Lag 0: simultáneo | Lags positivos: EDA sucede al reporte",
      x        = "Lag (épocas de 10s)",
      y        = "Feature EDA",
      fill     = "Beta"
    ) +
    
    theme_minimal(base_size = 12) +
    theme(
      panel.grid    = element_blank(),
      axis.text.x   = element_text(size = 8),
      axis.text.y   = element_text(size = 10),
      plot.title    = element_text(hjust = 0.5, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 9, color = "gray40"),
      legend.position = "right"
    )
  
  file_path <- file.path(output_folder, paste0("heatmap_", dimension_name, ".png"))
  ggsave(file_path, plot = p, width = 16, height = 6, dpi = 300)
  
  cat("🔥 Heatmap guardado en:", file_path, "\n")
}

# ==============================
# FUNCIÓN: CORRER MODELO LINEAL MIXTO
# ==============================
run_mixed_model <- function(df, target_var, predictor_var) {
  
  df_clean <- df %>%
    drop_na(all_of(c(target_var, predictor_var, "Subject", "Session", "Block", "Stimulus")))
  
  if (nrow(df_clean) == 0) return(NULL)
  
  df_clean <- df_clean %>%
    mutate(
      Subject  = as.factor(Subject),
      Session  = as.factor(Session),
      Block    = as.factor(Block),
      Stimulus = as.factor(Stimulus)
    )
  
  formula_str <- paste0(
    target_var, " ~ ", predictor_var, " + Session + Block + ",
    "(1 | Subject:Session) + ",
    "(1 | Stimulus)"
  )
  
  model <- tryCatch({
    lmer(as.formula(formula_str), data = df_clean)
  }, error = function(e) {
    message("❌ Error en modelo: ", e$message)
    return(NULL)
  })
  
  if (is.null(model)) return(NULL)
  
  summary_model <- summary(model)
  
  beta <- summary_model$coefficients[predictor_var, "Estimate"]
  pval <- summary_model$coefficients[predictor_var, "Pr(>|t|)"]
  se   <- summary_model$coefficients[predictor_var, "Std. Error"]
  
  var_comp <- as.data.frame(VarCorr(model))
  
  var_subject  <- var_comp %>% filter(grp == "Subject") %>% pull(vcov)
  var_session  <- var_comp %>% filter(grp == "Subject:Session") %>% pull(vcov)
  var_block    <- var_comp %>% filter(grp == "Subject:Session:Block") %>% pull(vcov)
  var_stimulus <- var_comp %>% filter(grp == "Stimulus") %>% pull(vcov)
  var_residual <- attr(VarCorr(model), "sc")^2
  
  var_subject  <- ifelse(length(var_subject)  == 0, 0, var_subject)
  var_session  <- ifelse(length(var_session)  == 0, 0, var_session)
  var_block    <- ifelse(length(var_block)    == 0, 0, var_block)
  var_stimulus <- ifelse(length(var_stimulus) == 0, 0, var_stimulus)
  
  total_var <- var_subject + var_session + var_block + var_stimulus + var_residual
  
  r2_vals <- tryCatch({ r2(model) }, error = function(e) NULL)
  
  r2_marginal    <- ifelse(is.null(r2_vals), NA, r2_vals$R2_marginal)
  r2_conditional <- ifelse(is.null(r2_vals), NA, r2_vals$R2_conditional)
  
  return(list(
    beta           = beta,
    pval           = pval,
    se             = se,
    var_subject    = var_subject  / total_var * 100,
    var_session    = var_session  / total_var * 100,
    var_block      = var_block    / total_var * 100,
    var_stimulus   = var_stimulus / total_var * 100,
    var_residual   = var_residual / total_var * 100,
    r2_marginal    = r2_marginal,
    r2_conditional = r2_conditional,
    n              = nrow(df_clean),
    n_subjects     = n_distinct(df_clean$Subject),
    model          = model
  ))
}

# ==============================
# FUNCIÓN: ANALIZAR DIMENSIÓN (VALENCIA)
# ==============================
analyze_valence_raw <- function(filepath, dimension_name) {
  
  if (!file.exists(filepath)) {
    message("⚠️ Archivo no encontrado: ", filepath)
    return()
  }
  
  cat("\n📊 ANALIZANDO:", toupper(dimension_name), "CON Report_Mean_Raw\n")
  
  df <- read_csv(filepath)
  
  # ========================================================================
  # ÚNICO CAMBIO: Se especifica "Report_Mean_Raw" como variable dependiente
  # ========================================================================
  target_var <- "Report_Mean_Raw"
  
  # Verificar que la columna existe en el dataframe
  if (!target_var %in% names(df)) {
    stop(paste("Error: La columna '", target_var, "' no se encuentra en el archivo CSV. Columnas disponibles: ", paste(names(df), collapse=", ")))
  }
  
  dim_output <- file.path(OUTPUT_DIR, paste0(dimension_name, "_raw"))
  dir.create(dim_output, showWarnings = FALSE)
  
  results_all <- list()
  
  for (lag in LAGS_TO_TEST) {
    
    cat("\n⏱️ LAG", lag, "\n")
    
    df_lag <- apply_lag(df, EDA_FEATURES, lag)
    
    for (pred in EDA_FEATURES) {
      
      cat("  👉", pred, "\n")
      
      res <- run_mixed_model(df_lag, target_var, pred)
      
      if (!is.null(res)) {
        
        sig <- ifelse(res$pval < 0.001, "***",
                      ifelse(res$pval < 0.01,  "**",
                             ifelse(res$pval < 0.05,  "*", "ns")))
        
        cat(sprintf("     β=%.4f | p=%.4f %s\n", res$beta, res$pval, sig))
        cat(sprintf("     Var%% - Subj: %.1f | Sess: %.1f | Block: %.1f | Stim: %.1f | Resid: %.1f\n",
                    res$var_subject, res$var_session, res$var_block,
                    res$var_stimulus, res$var_residual))
        cat(sprintf("     R2_marg=%.3f | R2_cond=%.3f\n", res$r2_marginal, res$r2_conditional))
        
        results_all[[length(results_all) + 1]] <- data.frame(
          Lag            = lag,
          Predictor      = pred,
          Beta           = res$beta,
          P_Value        = res$pval,
          Std_Err        = res$se,
          Var_Subject    = res$var_subject,
          Var_Session    = res$var_session,
          Var_Block      = res$var_block,
          Var_Stimulus   = res$var_stimulus,
          Var_Residual   = res$var_residual,
          R2_Marginal    = res$r2_marginal,
          R2_Conditional = res$r2_conditional,
          N              = res$n,
          N_Subjects     = res$n_subjects
        )
      }
    }
  }
  
  if (length(results_all) > 0) {
    results_df           <- bind_rows(results_all)
    results_df$Dimension <- dimension_name
    
    create_apa_table(results_df, dimension_name, dim_output)
    create_heatmap(results_df, dimension_name, dim_output)
    
    output_path <- file.path(dim_output, paste0("lmm_results_", dimension_name, ".csv"))
    write_csv(results_df, output_path)
    
    cat("\n💾 Resultados guardados en el directorio:", dim_output, "\n")
  }
}

# ==============================
# SCRIPT PRINCIPAL
# ==============================
cat("🚀 Iniciando análisis LMM en R...\n")

# Ejecutar el análisis únicamente para valencia usando Report_Mean_Raw
analyze_valence_raw(VALENCE_CSV, "valence")

cat("\n🎉 Análisis finalizado.\n")
