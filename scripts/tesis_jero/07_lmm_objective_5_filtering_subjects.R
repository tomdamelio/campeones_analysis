# ==============================
# LIBRERÍAS
# ==============================
library(tidyverse)
library(lme4)
library(lmerTest)
library(emmeans)
library(performance)
library(effectsize)

# ==============================
# CONFIGURACIÓN
# ==============================
BASE_DIR   <- "C:/Users/jeror/Desktop/Cocuco/campeones_analysis/results/analysis_models_obj2"
OUTPUT_DIR <- file.path(BASE_DIR, "stats_objective_5_LMM")
dir.create(OUTPUT_DIR, showWarnings = FALSE)

# ==============================
# FUNCIÓN PARA CARGAR DATOS
# ==============================
load_dimension_data <- function(dimension) {
  
  dim_path <- file.path(BASE_DIR, dimension)
  files    <- list.files(dim_path, pattern = "subject_summary.*\\.csv", full.names = TRUE)
  
  df_list <- list()
  
  for (f in files) {
    df <- read.csv(f)
    
    if      (grepl("random_forest", f)) model <- "RandomForest"
    else if (grepl("svr",           f)) model <- "SVM"
    else                                model <- "Ridge"
    
    df_list[[length(df_list) + 1]] <- df %>%
      select(Subject, Mean_RMSE) %>%
      mutate(dimension = dimension, model = model)
  }
  
  bind_rows(df_list)
}

# ==============================
# CARGA DE DATOS
# ==============================
cat("📂 Cargando datos...\n")

df_arousal <- load_dimension_data("arousal")
df_valence <- load_dimension_data("valence")

# Filtrar solo sujetos con datos en AMBAS dimensiones
subjects_arousal <- unique(df_arousal$Subject)
subjects_valence <- unique(df_valence$Subject)
subjects_common  <- intersect(subjects_arousal, subjects_valence)

cat(sprintf("   Sujetos en arousal:  %d\n", length(subjects_arousal)))
cat(sprintf("   Sujetos en valence:  %d\n", length(subjects_valence)))
cat(sprintf("   Sujetos en ambas:    %d\n", length(subjects_common)))

if (length(subjects_common) < length(subjects_arousal) |
    length(subjects_common) < length(subjects_valence)) {
  cat("⚠️  Sujetos excluidos por no tener contraparte afectiva:\n")
  cat("   Arousal sin valence:", setdiff(subjects_arousal, subjects_valence), "\n")
  cat("   Valence sin arousal:", setdiff(subjects_valence, subjects_arousal), "\n")
}

df_all <- bind_rows(df_arousal, df_valence) %>%
  filter(Subject %in% subjects_common) %>%
  mutate(
    Subject   = as.factor(Subject),
    dimension = as.factor(dimension),
    model     = as.factor(model)
  )

cat("✅ Datos cargados\n")
print(table(df_all$dimension, df_all$model))

write.csv(df_all, file.path(OUTPUT_DIR, "dataset_long.csv"), row.names = FALSE)

# ==============================
# MODELO LINEAL MIXTO
# (slope de dimensión por sujeto)
# ==============================
cat("\n🧪 Ajustando modelo LMM...\n")

model_lmm <- lmer(Mean_RMSE ~ dimension * model + (1 + dimension | Subject),
                  data = df_all)

summary(model_lmm)

# ==============================
# MÉTRICAS DEL MODELO
# ==============================
cat("\n📊 Métricas del modelo\n")
print(r2(model_lmm))
print(model_performance(model_lmm))

# ==============================
# ANOVA (efectos principales)
# ==============================
cat("\n📈 ANOVA del modelo\n")

anova_results <- anova(model_lmm)
print(anova_results)
write.csv(as.data.frame(anova_results),
          file.path(OUTPUT_DIR, "anova_results.csv"))

# ==============================
# POST-HOC: dimensión por modelo
# ==============================
cat("\n🔍 Comparaciones post-hoc\n")

emm_by_model   <- emmeans(model_lmm, ~ dimension | model)
contrasts_model <- summary(contrast(emm_by_model, method = "pairwise"))
print(contrasts_model)
write.csv(as.data.frame(contrasts_model),
          file.path(OUTPUT_DIR, "posthoc_dimension_by_model.csv"))

# ==============================
# EFECTO GLOBAL DE DIMENSIÓN
# ==============================
emm_global      <- emmeans(model_lmm, ~ dimension)
contrast_global <- summary(contrast(emm_global, method = "pairwise"))
print(contrast_global)

# ==============================
# TAMAÑO DE EFECTO
# ==============================
cat("\n📏 Tamaño de efecto\n")

effect_sizes <- eff_size(emm_global,
                         sigma = sigma(model_lmm),
                         edf   = df.residual(model_lmm))
print(effect_sizes)

# ==============================
# SLOPES POR SUJETO
# ==============================
random_effects <- ranef(model_lmm)$Subject
fixed_valence  <- fixef(model_lmm)["dimensionvalence"]

slopes_df <- data.frame(
  Subject        = rownames(random_effects),
  intercept      = random_effects[, "(Intercept)"],
  slope_re       = random_effects[, "dimensionvalence"],
  true_slope     = fixed_valence + random_effects[, "dimensionvalence"]
)

write.csv(slopes_df,
          file.path(OUTPUT_DIR, "subject_slopes.csv"),
          row.names = FALSE)

# ==============================
# GRÁFICOS
# ==============================

# 1. Interacción modelo × dimensión
p_interaction <- ggplot(df_all, aes(x = dimension, y = Mean_RMSE, color = model)) +
  stat_summary(fun      = mean, geom = "point",    size = 3) +
  stat_summary(fun      = mean, geom = "line",     aes(group = model)) +
  stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.1) +
  theme_minimal() +
  labs(y = "RMSE", x = "Dimensión", color = "Modelo") +
  scale_x_discrete(labels = c("arousal" = "Arousal", "valence" = "Valencia"))

ggsave(file.path(OUTPUT_DIR, "interaction_plot.png"),
       p_interaction, width = 6, height = 5)

# 2. Efectos marginales de dimensión
emm_df <- as.data.frame(emm_global)

p_marginal <- ggplot(emm_df, aes(x = dimension, y = emmean)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL), width = 0.1) +
  theme_minimal() +
  labs(title = "Efecto marginal de la dimensión", y = "RMSE estimado")

ggsave(file.path(OUTPUT_DIR, "marginal_means.png"),
       p_marginal, width = 6, height = 5)

# 3. Variación inter-sujeto (spaghetti)
# AGREGADO: Filtrar solo las combinaciones de Sujeto y modelo que están en ambas dimensiones
df_spaghetti <- df_all %>%
  group_by(Subject, model) %>%
  filter(n_distinct(dimension) == 2) %>%
  ungroup()

# MODIFICADO: Usar df_spaghetti y agrupar por interaction(Subject, model)
p_spaghetti <- ggplot(df_spaghetti,
                      aes(x = dimension, y = Mean_RMSE, group = interaction(Subject, model))) +
  geom_line(alpha = 0.4, color = "gray50") +
  geom_point(aes(color = dimension), size = 2) +
  theme_minimal() +
  labs(y = "RMSE", x = "Dimensión") +
  scale_x_discrete(labels = c("arousal" = "Arousal", "valence" = "Valencia")) +
  theme(legend.position = "none") # Esta es la línea que elimina la leyenda

ggsave(file.path(OUTPUT_DIR, "spaghetti_subjects.png"),
       p_spaghetti, width = 6, height = 5)

# 4. Distribución de slopes totales por sujeto
p_slopes <- ggplot(slopes_df, aes(x = true_slope)) +
  geom_histogram(bins = 10, fill = "darkgreen", alpha = 0.7) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  theme_minimal() +
  labs(title = "Efecto total por sujeto (valence vs arousal)",
       x = "Slope total", y = "Frecuencia")

ggsave(file.path(OUTPUT_DIR, "slopes_per_subject_distribution.png"),
       p_slopes, width = 6, height = 5)

cat("\n✅ Análisis completo guardado en:\n", OUTPUT_DIR, "\n")