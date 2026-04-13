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
OUTPUT_DIR <- "C:/Users/jeror/Desktop/Cocuco/campeones_analysis/results/analysis_models_obj3"

VALENCE_CSV <- file.path(INPUT_DIR, "valence_epochs_10s.csv")
AROUSAL_CSV <- file.path(INPUT_DIR, "arousal_epochs_10s.csv")

EDA_FEATURES <- c(
  "EDA_Tonic_Mean",
  "EDA_Phasic_Mean",
  "EDA_SMNA_AUC",
  "EDA_SCR_Peaks_Count",
  "EDA_SCR_Amplitude_Mean",
  "EDA_SMNA_Peaks_Count",
  "EDA_SMNA_Amplitude_Mean"
)

# Lags de -5 a +5 (0 incluido)
# Negativo: EDA de t+lag predice reporte en t (EDA anterior al reporte)
# Positivo: EDA de t+lag predice reporte en t (EDA posterior al reporte)
LAGS_TO_TEST <- -5:5

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ==============================
# FUNCIÓN: SHIFT POR LAG
# ==============================
apply_lag <- function(df, eda_features, lag_val) {
  if (lag_val == 0) return(df)
  
  df %>%
    arrange(Subject, Session, Block, Stimulus, Epoch_Index) %>%
    group_by(Subject, Session, Block, Stimulus) %>%
    mutate(across(
      all_of(eda_features),
      ~ if (lag_val < 0) {
        lag(.x, abs(lag_val))
      } else {
        lead(.x, lag_val)
      }
    )) %>%
    ungroup()
}

# ==============================
# FUNCIÓN: APA TABLE WITH RESULTS
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
      # Celda: β [CI_low, CI_high] (SE) df=X ***
      Cell = sprintf(
        "β=%.3f%s\n[%.3f, %.3f]\nSE=%.3f, df=%.0f",
        Beta, Sig, CI_Lower, CI_Upper, Std_Err, DF
      )
    ) %>%
    select(Predictor, Lag, Cell) %>%
    pivot_wider(
      names_from  = Lag,
      values_from = Cell
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
# FUNCIÓN: PLOTEAR HEATMAP DE BETAS
# Filtra solo predictores con al menos un lag significativo (p < .05).
# La intensidad del color se normaliza POR FILA (z-score del beta dentro
# de cada predictor), para que la variación temporal sea comparable
# entre features con betas en escalas distintas.
# ==============================
create_heatmap <- function(results_df, dimension_name, output_folder) {
  
  # 1. Identificar predictores con al menos un lag significativo
  sig_predictors <- results_df %>%
    group_by(Predictor) %>%
    summarise(any_sig = any(P_Value < 0.05), .groups = "drop") %>%
    filter(any_sig) %>%
    pull(Predictor)
  
  if (length(sig_predictors) == 0) {
    cat("⚠️  No hay predictores significativos para el heatmap de", dimension_name, "\n")
    return(invisible(NULL))
  }
  
  df <- results_df %>%
    filter(Predictor %in% sig_predictors) %>%
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
    ) %>%
    # 2. Normalizar por fila preservando signo y cero como referencia real.
    #    Se divide por el máximo valor absoluto de la fila → el valor más extremo
    #    llega a +1 o -1, el resto queda proporcional. El signo nunca se toca.
    #    - Betas positivos → siempre rojizos, más intenso el mayor
    #    - Betas negativos → siempre azulados, más intenso el más negativo
    #    - Cambio de signo en algún lag → cambio de color en ese lag
    group_by(Predictor) %>%
    mutate(
      Beta_z = Beta / max(abs(Beta), na.rm = TRUE)
    ) %>%
    ungroup()
  
  p <- ggplot(df, aes(x = Lag_factor, y = Predictor, fill = Beta_z)) +
    geom_tile(color = "white", linewidth = 0.8) +
    geom_text(aes(label = Label, fontface = ifelse(Sig != "", "bold", "plain")), size = 4, lineheight = 0.9) +
    
    # Líneas verticales delimitando lag=0
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
      midpoint = 0,
      name     = "Beta\n(z-score\npor fila)"
    ) +
    
    scale_x_discrete(
      labels = function(x) {
        x_int <- as.integer(x)
        ifelse(x_int == 0,  "0\nsimultáneo",
               ifelse(x_int  == -5,  paste0(x,    "\nEDA antes"),
                      ifelse(x_int == 5, paste0(x, "\nEDA después"), "") 
               )
        )
      }
    ) +
    
    labs(
      x = "Lag (épocas de 3s)",
      y = "Característica de EDA"
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
  ggsave(file_path, plot = p, width = 16, height = 3 + 1.2 * length(sig_predictors), dpi = 300)
  
  cat("🔥 Heatmap guardado en:", file_path, "\n")
  cat("   Predictores incluidos:", paste(sig_predictors, collapse = ", "), "\n")
}

# ==============================
# FUNCIÓN: CORRER MODELO MIXTO
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
    "(1 | Subject) + ",
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
  
  beta   <- summary_model$coefficients[predictor_var, "Estimate"]
  pval   <- summary_model$coefficients[predictor_var, "Pr(>|t|)"]
  se     <- summary_model$coefficients[predictor_var, "Std. Error"]
  # CAMBIO 1: extraer grados de libertad (Satterthwaite, provisto por lmerTest)
  df_val <- summary_model$coefficients[predictor_var, "df"]
  
  # CAMBIO 1: intervalos de confianza de Wald (rápidos y apropiados para LMM)
  ci <- tryCatch({
    confint(model, parm = predictor_var, method = "Wald")
  }, error = function(e) matrix(c(NA_real_, NA_real_), nrow = 1,
                                dimnames = list(predictor_var, c("2.5 %", "97.5 %"))))
  ci_lower <- ci[1, 1]
  ci_upper <- ci[1, 2]
  
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
    df_val         = df_val,   # CAMBIO 1
    ci_lower       = ci_lower, # CAMBIO 1
    ci_upper       = ci_upper, # CAMBIO 1
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
# FUNCIÓN: ANALIZAR DIMENSIÓN
# ==============================
analyze_dimension <- function(filepath, dimension_name) {
  
  if (!file.exists(filepath)) {
    message("⚠️ Archivo no encontrado: ", filepath)
    return()
  }
  
  cat("\n📊 ANALIZANDO:", toupper(dimension_name), "\n")
  
  df         <- read_csv(filepath)
  target_var <- "Report_Mean"
  
  dim_output <- file.path(OUTPUT_DIR, dimension_name)
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
        
        cat(sprintf("     β=%.4f [%.4f, %.4f] | SE=%.4f | df=%.1f | p=%.4f %s\n",
                    res$beta, res$ci_lower, res$ci_upper, res$se, res$df_val, res$pval, sig))
        cat(sprintf("     Var%% - Subj: %.1f | Sess: %.1f | Block: %.1f | Stim: %.1f | Resid: %.1f\n",
                    res$var_subject, res$var_session, res$var_block,
                    res$var_stimulus, res$var_residual))
        cat(sprintf("     R2_marg=%.3f | R2_cond=%.3f\n", res$r2_marginal, res$r2_conditional))
        
        results_all[[length(results_all) + 1]] <- data.frame(
          Lag            = lag,
          Predictor      = pred,
          Beta           = res$beta,
          CI_Lower       = res$ci_lower,  # CAMBIO 1
          CI_Upper       = res$ci_upper,  # CAMBIO 1
          Std_Err        = res$se,
          DF             = res$df_val,    # CAMBIO 1
          P_Value        = res$pval,
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
    
    cat("\n💾 Guardado en:", output_path, "\n")
  }
}

# ==============================
# MAIN
# ==============================
cat("🚀 Iniciando análisis LMM en R...\n")

analyze_dimension(AROUSAL_CSV, "arousal")
analyze_dimension(VALENCE_CSV, "valence")

cat("\n🎉 Finalizado.\n")