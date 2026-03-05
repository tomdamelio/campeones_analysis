import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIGURACIÓN ---
SUBJECT_ID = "27"
RESULTS_PATH = rf"results/eda_preproc_tests/sub-{SUBJECT_ID}/beh"
TSV_PATH = os.path.join(RESULTS_PATH, f"sub-{SUBJECT_ID}_desc-videoratings_beh.tsv")

def plot_affective_space_comparison(df):
    """
    Genera una figura con dos subplots lado a lado:
    1. Espacio Afectivo del Reporte Continuo (con barras de error).
    2. Espacio Afectivo del Reporte a Posteriori (puntos únicos).
    """
    
    # Configurar la figura con 1 fila y 2 columnas
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True, sharex=True)
    fig.suptitle('Comparación del Espacio Afectivo: Continuo vs. Posteriori', fontsize=18)

    # Identificar IDs para colorear
    video_ids = df.index
    mask_pos = (video_ids >= 1) & (video_ids <= 7)   # Videos 1-7 (Verdes)
    mask_neg = (video_ids >= 8) & (video_ids <= 14)  # Videos 8-14 (Rojos)

    # Definir la configuración de datos para cada subplot
    plots_config = [
        {
            "ax": axes[0],
            "title": "Reporte Continuo (Promedio)",
            "x": df['continua_valencia_media'],
            "y": df['continua_arousal_media'],
            "xerr": df['continua_valencia_desvio'],
            "yerr": df['continua_arousal_desvio'],
            "alpha_points": 0.8
        },
        {
            "ax": axes[1],
            "title": "Reporte a Posteriori",
            "x": df['post_valencia'],
            "y": df['post_arousal'],
            "xerr": None, # No hay desvío en reporte único
            "yerr": None,
            "alpha_points": 1.0
        }
    ]

    # Iterar para generar ambos gráficos con el mismo estilo
    for config in plots_config:
        ax = config["ax"]
        
        # 1. Dibujar líneas centrales (Neutral = 5)
        ax.axhline(5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(5, color='gray', linestyle='--', alpha=0.5)

        # 2. Graficar puntos (Verdes para Valencia Positiva)
        if mask_pos.any():
            ax.errorbar(
                config["x"][mask_pos], config["y"][mask_pos], 
                xerr=config["xerr"][mask_pos] if config["xerr"] is not None else None, 
                yerr=config["yerr"][mask_pos] if config["yerr"] is not None else None, 
                fmt='o', color='forestgreen', ecolor='lightgreen', 
                elinewidth=2, capsize=4, markersize=9, alpha=config["alpha_points"], 
                label='Videos 1-7 (Val. Positiva)'
            )

        # 3. Graficar puntos (Rojos para Valencia Negativa)
        if mask_neg.any():
            ax.errorbar(
                config["x"][mask_neg], config["y"][mask_neg], 
                xerr=config["xerr"][mask_neg] if config["xerr"] is not None else None, 
                yerr=config["yerr"][mask_neg] if config["yerr"] is not None else None, 
                fmt='o', color='crimson', ecolor='lightcoral', 
                elinewidth=2, capsize=4, markersize=9, alpha=config["alpha_points"], 
                label='Videos 8-14 (Val. Negativa)'
            )

        # 4. Anotar IDs de videos
        for i, vid in enumerate(video_ids):
            # Usamos iloc para acceder por posición ya que estamos iterando enumerates
            x_val = config["x"].iloc[i]
            y_val = config["y"].iloc[i]
            
            # Solo anotamos si no es NaN
            if pd.notna(x_val) and pd.notna(y_val):
                ax.annotate(str(vid), (x_val, y_val), textcoords="offset points", 
                            xytext=(5, 5), ha='left', fontsize=10, color='black', weight='bold')

        # 5. Estética del Subplot
        ax.set_title(config["title"], fontsize=14, pad=10)
        ax.set_xlim(0.5, 9.5)
        ax.set_ylim(0.5, 9.5)
        ax.set_xticks(range(1, 10))
        ax.set_yticks(range(1, 10))
        ax.grid(True, linestyle=':', alpha=0.4)
        
        # Etiquetas de ejes
        ax.set_xlabel('VALENCIA', fontsize=11, fontweight='bold')
        if ax == axes[0]: # Solo etiqueta Y en el primer gráfico para no ensuciar
            ax.set_ylabel('AROUSAL', fontsize=11, fontweight='bold')

        # 6. Etiquetas de cuadrantes (Contexto emocional)
        ax.text(0.6, 9.4, 'Alta Act. / Negativo', color='gray', fontsize=9, ha='left', va='top')
        ax.text(9.4, 9.4, 'Alta Act. / Positivo', color='gray', fontsize=9, ha='right', va='top')
        ax.text(0.6, 0.6, 'Baja Act. / Negativo', color='gray', fontsize=9, ha='left', va='bottom')
        ax.text(9.4, 0.6, 'Baja Act. / Positivo', color='gray', fontsize=9, ha='right', va='bottom')

    # Leyenda única
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.02), fontsize=12)

    plt.subplots_adjust(bottom=0.15) # Espacio para la leyenda
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Ajustar layout respetando la leyenda
    
    # Guardar figura en carpeta results con nomenclatura BIDS
    fig_name = f"sub-{SUBJECT_ID}_desc-affectivespace_fig.png"
    fig_path = os.path.join(RESULTS_PATH, fig_name)
    
    plt.savefig(fig_path, dpi=300)
    print(f"💾 Gráfico de Espacio Afectivo guardado en: {fig_path}")
    plt.show()

# --- SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    if not os.path.exists(TSV_PATH):
        print(f"ERROR: No se encontró el archivo '{TSV_PATH}'.")
        print("Asegúrate de haber ejecutado el script de extracción de reportes primero.")
    else:
        print(f"Cargando datos desde {TSV_PATH}...")
        # Leemos el TSV usando tabulación como separador
        df = pd.read_csv(TSV_PATH, sep='\t', index_col='video_id')
        
        print("Generando Gráfico Comparativo de Espacios Afectivos...")
        plot_affective_space_comparison(df)