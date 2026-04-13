import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACIÓN ---
INPUT_DIR = r"results/epoch_features"
OUTPUT_DIR = r"results/inter_subject_correlation_summary"

FILES_TO_PROCESS = {
    "valence": os.path.join(INPUT_DIR, "valence_epochs_10s.csv"),
    "arousal": os.path.join(INPUT_DIR, "arousal_epochs_10s.csv")
}

# 1. Solo analizaremos el reporte
FEATURE_TO_ANALYZE = 'Report_Mean'
MIN_SUBJECTS_PER_STIMULUS = 5 
CORR_METHOD = 'spearman'

def analyze_isc_for_dimension(df, dimension):
    """
    Calcula el ISC promedio para cada estímulo en un dataframe dado.
    """
    print(f"\n  Analizando Dimensión: {dimension.upper()}")
    
    # Crear carpeta de salida específica
    output_folder = os.path.join(OUTPUT_DIR, dimension)
    os.makedirs(output_folder, exist_ok=True)

    unique_stimuli = sorted(df['Stimulus'].unique())
    isc_summary = []

    for stimulus_id in unique_stimuli:
        df_stimulus = df[df['Stimulus'] == stimulus_id].copy()
        
        n_subjects = df_stimulus['Subject'].nunique()
        if n_subjects < MIN_SUBJECTS_PER_STIMULUS:
            continue

        # 2. SOLUCIÓN AL ERROR: Usar pivot_table para manejar duplicados promediándolos
        try:
            wide_df = df_stimulus.pivot_table(
                index='Epoch_Index', 
                columns='Subject', 
                values=FEATURE_TO_ANALYZE,
                aggfunc='mean' # Si hay duplicados (ej. por sesiones A/B), los promedia.
            )
        except Exception as e:
            print(f"   ❌ Error irrecuperable al pivotar datos para estímulo {stimulus_id}. Error: {e}")
            continue

        common_epochs_df = wide_df.dropna()

        if len(common_epochs_df) < 3: # Necesitamos al menos 3 puntos para una correlación mínimamente fiable
            continue

        isc_matrix = common_epochs_df.corr(method=CORR_METHOD)

        # 3. CÁLCULO DE LA MÉTRICA ÚNICA: Promedio del triángulo inferior
        # Creamos una máscara para seleccionar solo los elementos bajo la diagonal
        mask = np.tril(np.ones_like(isc_matrix, dtype=bool), k=-1)
        # Extraemos esos valores y calculamos el promedio
        avg_isc = isc_matrix.where(mask).stack().mean()
        
        isc_summary.append({
            'Stimulus': stimulus_id,
            'Average_ISC': avg_isc,
            'Num_Subjects': n_subjects,
            'Num_Common_Epochs': len(common_epochs_df)
        })
        print(f"   ✅ Estímulo {stimulus_id}: ISC Promedio = {avg_isc:.4f}")

    if not isc_summary:
        print("   No se pudo generar un resumen de ISC para esta dimensión.")
        return

    # Convertir el resumen a un DataFrame y guardarlo
    summary_df = pd.DataFrame(isc_summary).sort_values(by='Average_ISC', ascending=False)
    summary_csv_path = os.path.join(output_folder, f'{dimension}_isc_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n   💾 Resumen de ISC guardado en: {summary_csv_path}")

    # Generar el gráfico de barras resumen
    plt.figure(figsize=(15, 8))
    sns.barplot(data=summary_df, x='Stimulus', y='Average_ISC', palette='viridis', order=summary_df['Stimulus'])
    plt.title(f"Sincronía Inter-Sujeto Promedio por Estímulo\nDimensión: {dimension.upper()} | Feature: {FEATURE_TO_ANALYZE}", fontsize=16)
    plt.ylabel(f"ISC Promedio (Correlación de {CORR_METHOD.capitalize()})")
    plt.xlabel("ID del Estímulo")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot_path = os.path.join(output_folder, f'{dimension}_isc_summary_barplot.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"   📊 Gráfico de barras resumen guardado en: {plot_path}")

def main():
    print("Iniciando análisis de Correlación Inter-Sujeto (ISC)...")

    for dimension, filepath in FILES_TO_PROCESS.items():
        if not os.path.exists(filepath):
            print(f"\n⚠️ Archivo no encontrado para '{dimension}': {filepath}. Saltando...")
            continue
        
        df = pd.read_csv(filepath)
        
        # Opcional: El código de diagnóstico que te pasé antes
        keys = ['Subject', 'Stimulus', 'Epoch_Index']
        duplicates = df[df.duplicated(subset=keys, keep=False)]
        if not duplicates.empty:
            print(f"\n🚨 ¡ATENCIÓN! Se encontraron {len(duplicates)} duplicados en '{dimension}'. Serán promediados.")
            print(duplicates.sort_values(by=keys).head(10))
            print(duplicates.sort_values(by=keys).tail(10))

        analyze_isc_for_dimension(df, dimension)
    
    print("\n🎉 Análisis de ISC completado.")

if __name__ == "__main__":
    main()