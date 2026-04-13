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
# ==============================
create_heatmap <- function(results_df, dimension_name, output_folder) {
  
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
    group_by(Predictor) %>%
    mutate(
      Beta_z = Beta / max(abs(Beta), na.rm = TRUE)
    ) %>%
    ungroup()
  
  p <- ggplot(df, aes(x = Lag_factor, y = Predictor, fill = Beta_z)) +
    geom_tile(color = "white", linewidth = 0.8) +
    geom_text(aes(label = Label), size = 2.8, lineheight = 0.9) +
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
        ifelse(x_int == 0,  "0\n(simultáneo)",
               ifelse(x_int  < 0,  paste0(x,    "\n(EDA antes)"),
                      paste0("+", x, "\n(EDA después)")))
      }
    ) +
    labs(
      title    = paste("Heatmap de Betas -", toupper(dimension_name),
                       "(solo predictores significativos)"),
      subtitle = paste(
        "Lags negativos: EDA precede al reporte | Lag 0: simultáneo | Lags positivos: EDA sucede al reporte\n",
        "Intensidad de color normalizada por fila (z-score); el valor numérico muestra β original"
      ),
      x = "Lag (épocas de 10s)",
      y = "Feature EDA"
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
# FUNCIÓN: DIAGNÓSTICO POR FEATURE
# Un PNG por feature con una fila de paneles por lag (QQ + Resid vs Fitted).
# ==============================
save_diagnostics_per_feature <- function(models_by_feature, dimension_name, output_folder) {
  
  diag_folder <- file.path(output_folder, "diagnostics_per_feature")
  dir.create(diag_folder, showWarnings = FALSE)
  
  for (pred in names(models_by_feature)) {
    
    model_list <- models_by_feature[[pred]]  # lista: lag -> model
    lags       <- sort(as.integer(names(model_list)))
    n_lags     <- length(lags)
    
    if (n_lags == 0) next
    
    # 2 paneles por lag (QQ + Resid vs Fitted), n_lags filas
    file_path <- file.path(diag_folder,
                           sprintf("diagnostics_%s_%s.png", dimension_name, pred))
    
    png(file_path, width = 900, height = 320 * n_lags, res = 130)
    par(mfrow = c(n_lags, 2), mar = c(4, 4, 3, 1))
    
    for (lag in lags) {
      model <- model_list[[as.character(lag)]]
      if (is.null(model)) next
      
      lag_label <- ifelse(lag == 0, "lag=0 (simultáneo)",
                          ifelse(lag < 0, sprintf("lag=%d (EDA antes)", lag),
                                 sprintf("lag=+%d (EDA después)", lag)))
      
      # QQ-plot
      qqnorm(resid(model),
             main = sprintf("%s | %s\nQQ-Plot residuos", pred, lag_label),
             cex.main = 0.85)
      qqline(resid(model), col = "red", lwd = 1.5)
      
      # Residuos vs Fitted
      plot(fitted(model), resid(model),
           main = sprintf("%s | %s\nResid vs Fitted", pred, lag_label),
           xlab = "Fitted", ylab = "Residuos",
           pch  = 16, cex = 0.5, col = rgb(0, 0, 0, 0.3),
           cex.main = 0.85)
      abline(h = 0, col = "red", lty = 2, lwd = 1.5)
      lines(lowess(fitted(model), resid(model)), col = "blue", lwd = 1.5)
    }
    
    dev.off()
    cat("🔬 Diagnóstico por feature guardado en:", file_path, "\n")
  }
}

# ==============================
# FUNCIÓN: DIAGNÓSTICO GLOBAL POR DIMENSIÓN
# Concatena los residuos de TODOS los modelos de la dimensión
# y produce un único panel de 4 gráficos.
# ==============================
save_diagnostics_global <- function(models_by_feature, dimension_name, output_folder) {
  
  # Recolectar todos los residuos y fitted de todos los modelos
  all_resid  <- c()
  all_fitted <- c()
  
  for (pred in names(models_by_feature)) {
    for (lag_key in names(models_by_feature[[pred]])) {
      model <- models_by_feature[[pred]][[lag_key]]
      if (!is.null(model)) {
        all_resid  <- c(all_resid,  resid(model))
        all_fitted <- c(all_fitted, fitted(model))
      }
    }
  }
  
  if (length(all_resid) == 0) {
    cat("⚠️  No hay residuos para el diagnóstico global de", dimension_name, "\n")
    return(invisible(NULL))
  }
  
  file_path <- file.path(output_folder,
                         sprintf("diagnostics_global_%s.png", dimension_name))
  
  png(file_path, width = 1200, height = 1100, res = 150)
  par(mfrow = c(2, 2), mar = c(4, 4, 4, 2))
  
  # 1. QQ-plot global
  qqnorm(all_resid,
         main = sprintf("QQ-Plot global — %s\n(todos los modelos, n=%s residuos)",
                        toupper(dimension_name), format(length(all_resid), big.mark = ",")),
         pch = 16, cex = 0.3, col = rgb(0, 0, 0, 0.15))
  qqline(all_resid, col = "red", lwd = 2)
  
  # 2. Residuos vs Fitted global
  plot(all_fitted, all_resid,
       main = sprintf("Resid vs Fitted global — %s", toupper(dimension_name)),
       xlab = "Fitted", ylab = "Residuos",
       pch  = 16, cex = 0.3, col = rgb(0, 0, 0, 0.15))
  abline(h = 0, col = "red", lty = 2, lwd = 2)
  lines(lowess(all_fitted, all_resid), col = "blue", lwd = 2)
  
  # 3. Histograma de residuos global
  hist(all_resid,
       main   = sprintf("Histograma residuos global — %s", toupper(dimension_name)),
       xlab   = "Residuos",
       breaks = 60,
       col    = "steelblue",
       border = "white")
  curve(dnorm(x, mean = mean(all_resid), sd = sd(all_resid)) * length(all_resid) *
          diff(hist(all_resid, plot = FALSE, breaks = 60)$breaks)[1],
        add = TRUE, col = "red", lwd = 2)
  
  # 4. Densidad de residuos vs normal teórica
  d <- density(all_resid)
  plot(d,
       main = sprintf("Densidad residuos vs Normal — %s", toupper(dimension_name)),
       xlab = "Residuos", ylab = "Densidad",
       col  = "steelblue", lwd = 2)
  curve(dnorm(x, mean = mean(all_resid), sd = sd(all_resid)),
        add = TRUE, col = "red", lwd = 2, lty = 2)
  legend("topright", legend = c("Residuos", "Normal teórica"),
         col = c("steelblue", "red"), lwd = 2, lty = c(1, 2), bty = "n")
  
  dev.off()
  
  # Estadísticos de resumen para consola
  sw_test <- tryCatch(shapiro.test(sample(all_resid, min(5000, length(all_resid)))),
                      error = function(e) NULL)
  
  cat("\n📊 DIAGNÓSTICO GLOBAL —", toupper(dimension_name), "\n")
  cat(sprintf("   Residuos totales: %s\n", format(length(all_resid), big.mark = ",")))
  cat(sprintf("   Media: %.4f | SD: %.4f | Skewness: %.4f\n",
              mean(all_resid), sd(all_resid),
              mean((all_resid - mean(all_resid))^3) / sd(all_resid)^3))
  if (!is.null(sw_test)) {
    cat(sprintf("   Shapiro-Wilk (muestra): W=%.4f, p=%.4f\n",
                sw_test$statistic, sw_test$p.value))
    cat("   ⚠️  Nota: con n grande Shapiro-Wilk es muy sensible; priorizá el QQ-plot visualmente.\n")
  }
  cat("🔬 Diagnóstico global guardado en:", file_path, "\n")
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
  df_val <- summary_model$coefficients[predictor_var, "df"]
  
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
    df_val         = df_val,
    ci_lower       = ci_lower,
    ci_upper       = ci_upper,
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
  
  results_all      <- list()
  # Estructura para guardar modelos: models_by_feature[[pred]][[lag]] = model
  models_by_feature <- setNames(
    lapply(EDA_FEATURES, function(p) list()),
    EDA_FEATURES
  )
  
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
          CI_Lower       = res$ci_lower,
          CI_Upper       = res$ci_upper,
          Std_Err        = res$se,
          DF             = res$df_val,
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
        
        # Guardar modelo para diagnósticos
        models_by_feature[[pred]][[as.character(lag)]] <- res$model
      }
    }
  }
  
  if (length(results_all) > 0) {
    results_df           <- bind_rows(results_all)
    results_df$Dimension <- dimension_name
    
    create_apa_table(results_df, dimension_name, dim_output)
    create_heatmap(results_df, dimension_name, dim_output)
    
    # ── DIAGNÓSTICOS ──────────────────────────────────────────────────────────
    cat("\n🔬 Generando diagnósticos de residuos...\n")
    save_diagnostics_per_feature(models_by_feature, dimension_name, dim_output)
    save_diagnostics_global(models_by_feature, dimension_name, dim_output)
    # ──────────────────────────────────────────────────────────────────────────
    
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
