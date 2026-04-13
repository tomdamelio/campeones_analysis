import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr 
import random

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# --- CONFIGURACIÓN ---
INPUT_DIR = r"results/epoch_features"
OUTPUT_DIR = r"results/analysis_clusters"

VALENCE_CSV = os.path.join(INPUT_DIR, "valence_epochs_10s.csv")
AROUSAL_CSV = os.path.join(INPUT_DIR, "arousal_epochs_10s.csv")

# Definir las features para cada grupo
REPORT_FEATURES = ['Report_Mean', 'Report_Variance', 'Report_Deriv_Mean']
EDA_FEATURES = [
    'EDA_Clean_Mean', 'EDA_Phasic_Mean', 'EDA_Tonic_Mean', 
    'EDA_SMNA_AUC', 'EDA_SCR_Peaks_Count', 'EDA_SCR_Amplitude_Mean', 
    'EDA_SCR_RiseTime_Mean', 'EDA_SMNA_Peaks_Count', 'EDA_SMNA_Amplitude_Mean'
]

LAGS_TO_TEST = [0, 1, 2, 3] # 0 = misma época, 1 = reporte 1 época después, etc.

# Crear carpetas de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)


def cluster_1_descriptive(df, dimension_name, out_folder):
    """
    Análisis Descriptivo Agregado: Estadísticas centrales, de dispersión e histogramas.
    """
    print(f"\n📊 Iniciando Análisis Descriptivo para {dimension_name.upper()}...")
    
    features_to_describe = REPORT_FEATURES + EDA_FEATURES
    
    # 1. Tabla de estadísticas descriptivas (Media, Mediana, Desvío, Min, Max, etc.)
    desc_stats = df[features_to_describe].describe().T
    
    # Agregar la Mediana (que no viene por defecto en describe como 'median', sino como '50%')
    desc_stats.rename(columns={'50%': 'median'}, inplace=True)
    
    stats_csv_path = os.path.join(out_folder, f"{dimension_name}_descriptive_stats.csv")
    desc_stats.to_csv(stats_csv_path)
    print(f"   💾 Estadísticas guardadas en: {stats_csv_path}")
    
    # 2. Generar Histogramas con curva de densidad (KDE)
    num_features = len(features_to_describe)
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(16, 16))
    fig.suptitle(f"Distribución de Features - {dimension_name.upper()} (Datos Agregados)", fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, feature in enumerate(features_to_describe):
        sns.histplot(df[feature], kde=True, ax=axes[i], color='teal', bins=30)
        axes[i].set_title(feature)
        axes[i].set_xlabel("Valor")
        axes[i].set_ylabel("Frecuencia")
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    hist_path = os.path.join(out_folder, f"{dimension_name}_histograms.png")
    plt.savefig(hist_path, dpi=300)
    plt.show()
    plt.close()
    print(f"   🖼️ Histogramas guardados en: {hist_path}")


def cluster_2_correlation_lags(df, dimension_name, out_folder, CORR_METHOD = 'spearman'):
    """
    Análisis Correlativo con Lag: Cruza EDA(t) con Reporte(t + lag).
    Agrupa por Sujeto, Sesión, Bloque y Estímulo para no mezclar videos.
    Calcula p-valores y marca la significancia con asteriscos.
    """
    print(f"\n🔗 Iniciando Análisis de Correlación y Lags para {dimension_name.upper()}...")
    
    # Creamos un Excel con múltiples hojas para guardar todas las matrices
    excel_path = os.path.join(out_folder, f"{dimension_name}_correlations_lag.xlsx")
    
    with pd.ExcelWriter(excel_path) as writer:
        
        # Figura para los heatmaps (2x2 para acomodar los 4 lags)
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 14))
        fig.suptitle(f"Correlación EDA vs Reporte ({dimension_name.upper()}) con Retardos\n(EDA en t vs Reporte en t+Lag)\n* p<0.05, ** p<0.01, *** p<0.001", fontsize=16)
        axes = axes.flatten()
        
        for i, lag in enumerate(LAGS_TO_TEST):
            df_shifted = df.copy()
            
            if lag > 0:
                group_cols = ['Subject', 'Session', 'Block', 'Stimulus']
                df_shifted[REPORT_FEATURES] = df_shifted.groupby(group_cols)[REPORT_FEATURES].shift(-lag)
            
            df_clean = df_shifted.dropna(subset=REPORT_FEATURES + EDA_FEATURES)
            
            # --- NUEVA LÓGICA DE CORRELACIÓN CON P-VALORES ---
            # Matrices vacías para los coeficientes y las anotaciones
            crosstab_corr = pd.DataFrame(index=EDA_FEATURES, columns=REPORT_FEATURES, dtype=float)
            annot_matrix = pd.DataFrame(index=EDA_FEATURES, columns=REPORT_FEATURES, dtype=str)
            
            for eda in EDA_FEATURES:
                for rep in REPORT_FEATURES:
                    # Calcular correlación y p-valor
                    if CORR_METHOD == 'spearman':
                        r, p = spearmanr(df_clean[eda], df_clean[rep])
                    else:
                        r, p = pearsonr(df_clean[eda], df_clean[rep])
                    
                    crosstab_corr.loc[eda, rep] = r
                    
                    # Lógica de asteriscos basada en el p-valor
                    if p < 0.001:
                        stars = "***"
                    elif p < 0.01:
                        stars = "**"
                    elif p < 0.05:
                        stars = "*"
                    else:
                        stars = ""
                        
                    # Texto que irá en el cuadrito (ej. "0.456***")
                    annot_matrix.loc[eda, rep] = f"{r:.3f}{stars}"
            
            # Guardar en la hoja del Excel (guardamos la matriz con asteriscos para leerla fácil)
            annot_matrix.to_excel(writer, sheet_name=f"Lag_{lag}")
            
            # --- Graficar el Heatmap ---
            # Usamos crosstab_corr para los colores (vmin, vmax)
            # Usamos annot_matrix para el texto que se dibuja encima (fmt="")
            sns.heatmap(crosstab_corr, annot=annot_matrix.values, fmt="", cmap="coolwarm", 
                        vmin=-1, vmax=1, center=0, ax=axes[i], 
                        cbar_kws={'label': f'{CORR_METHOD.capitalize()} Rho'})
            
            axes[i].set_title(f"Lag {lag} épocas\n(EDA precede al reporte por {lag * 10}s)")
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].tick_params(axis='y', rotation=0)
            
            print(f"   ✔️ Correlación Lag {lag} calculada (N épocas usables: {len(df_clean)})")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        heatmap_path = os.path.join(out_folder, f"{dimension_name}_correlation_heatmaps.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
    print(f"   💾 Matrices de correlación guardadas en: {excel_path}")
    print(f"   🖼️ Heatmaps guardados en: {heatmap_path}")

def cluster_4_paired_distribution(df, dimension_name, out_folder):
    """
    Genera un boxplot pareado con scatterplot y líneas de conexión.
    Normaliza los datos (0-1) para que sean comparables en el mismo eje.
    """
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    
    print(f"\n🔗 Generando Boxplots Pareados para {dimension_name.upper()}...")
    
    eda_to_compare = ['EDA_Phasic_Mean', 'EDA_Tonic_Mean', 'EDA_SMNA_AUC']

    for eda_feat in eda_to_compare:
        # 1. Preparar datos y normalizar
        temp_df = df[['Report_Mean', eda_feat]].dropna()
        if temp_df.empty: continue
        
        # Escalamos para que ambas variables vayan de 0 a 1
        data_scaled = scaler.fit_transform(temp_df)
        df_scaled = pd.DataFrame(data_scaled, columns=['Report (Norm)', 'EDA (Norm)'])
        
        # 2. Configurar el plot
        plt.figure(figsize=(7, 6))
        
        # Dibujar las líneas que unen los puntos (el "pareo")
        # Para no saturar el gráfico, graficamos una muestra si hay demasiados puntos
        sample_n = min(len(df_scaled), 100) 
        df_sample = df_scaled.sample(sample_n, random_state=42)
        
        for i in df_sample.index:
            plt.plot([0, 1], [df_sample.loc[i, 'Report (Norm)'], df_sample.loc[i, 'EDA (Norm)']], 
                     color='gray', alpha=0.3, linewidth=0.5, zorder=1)

        # 3. Dibujar Boxplots y Stripplots (puntos)
        df_melted = df_scaled.melt(var_name='Variable', value_name='Valor (Normalizado)')
        
        sns.boxplot(data=df_melted, x='Variable', y='Valor (Normalizado)', 
                    palette='Set2', width=0.4, showfliers=False, zorder=2)
        
        sns.stripplot(data=df_melted, x='Variable', y='Valor (Normalizado)', 
                      color='black', alpha=0.5, size=4, zorder=3)

        plt.title(f"Distribución Pareada: Report vs {eda_feat}\n({dimension_name.upper()} - Datos Normalizados)")
        plt.ylim(-0.05, 1.05)
        
        plot_path = os.path.join(out_folder, f"{dimension_name}_paired_{eda_feat}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

def cluster_5_scatter_regression(df, dimension_name, out_folder, CORR_METHOD='spearman'):
    """
    Genera scatter plots con línea de regresión para features clave.
    Marca automáticamente si la relación es estadísticamente significativa.
    """
    print(f"\n📈 Generando Scatter Plots con Regresión para {dimension_name.upper()}...")
    
    # Las 3 que pediste específicamente
    target_eda = ['EDA_Phasic_Mean', 'EDA_Tonic_Mean', 'EDA_SMNA_AUC']
    
    for eda_feat in target_eda:
        temp_df = df[['Report_Mean', eda_feat]].dropna()
        if temp_df.empty: continue
        
        # Calcular correlación y p-valor para la etiqueta
        if CORR_METHOD == 'spearman':
            r, p = spearmanr(temp_df['Report_Mean'], temp_df[eda_feat])
        else:
            r, p = pearsonr(temp_df['Report_Mean'], temp_df[eda_feat])
            
        is_sig = p < 0.05
        sig_color = "firebrick" if is_sig else "slategray"
        stars = "*" if p < 0.05 else ""
        if p < 0.01: stars = "**"
        if p < 0.001: stars = "***"

        plt.figure(figsize=(8, 6))
        
        # Dibujar puntos y línea de regresión
        sns.regplot(data=temp_df, x='Report_Mean', y=eda_feat, 
                    scatter_kws={'alpha': 0.4, 'color': 'gray'}, 
                    line_kws={'color': sig_color, 'label': f'Regresión (p={p:.3f})'})
        
        # Formatear el título según significancia
        title_str = f"{dimension_name.upper()}: Report vs {eda_feat}\n"
        title_str += f"Rho: {r:.3f}{stars} " + ("(SIGNIFICATIVO)" if is_sig else "(No Sig.)")
        
        plt.title(title_str, fontweight='bold', color='black' if not is_sig else 'darkred')
        plt.xlabel(f"Reporte ({dimension_name})")
        plt.ylabel(eda_feat)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plot_path = os.path.join(out_folder, f"{dimension_name}_scatter_{eda_feat}.png")
        plt.show()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    print("Iniciando scripts de análisis...")
    
    datasets = {
        "valence": VALENCE_CSV,
        "arousal": AROUSAL_CSV
    }
    
    for dimension, path in datasets.items():
        if not os.path.exists(path):
            print(f"\n⚠️ No se encontró el archivo {path}. Saltando análisis de {dimension}.")
            continue
            
        # Crear subcarpeta específica para mantener organizado
        dim_out_folder = os.path.join(OUTPUT_DIR, dimension)
        os.makedirs(dim_out_folder, exist_ok=True)
        
        # Cargar datos
        df = pd.read_csv(path)
        print(f"\n=======================================================")
        print(f"Cargado dataset de {dimension.upper()} con {len(df)} épocas.")
        print(f"=======================================================")
        
        # Ejecutar los clústers
        cluster_1_descriptive(df, dimension, dim_out_folder)
        cluster_4_paired_distribution(df, dimension, dim_out_folder)
        cluster_2_correlation_lags(df, dimension, dim_out_folder)
        cluster_5_scatter_regression(df, dimension, dim_out_folder)
        
        
    print("\n🎉 Todos los análisis han finalizado correctamente.")

if __name__ == "__main__":
    main()