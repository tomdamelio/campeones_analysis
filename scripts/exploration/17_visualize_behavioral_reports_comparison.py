import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import os

# --- CONFIGURACIÓN ---
SUBJECT_ID = "27"
RESULTS_PATH = rf"results/eda_preproc_tests/sub-{SUBJECT_ID}/beh"
TSV_PATH = os.path.join(RESULTS_PATH, f"sub-{SUBJECT_ID}_desc-videoratings_beh.tsv")

def plot_correlation_scatter(df):
    """
    Idea 1: Gráfico de dispersión (Post vs Continuo).
    Puntos verdes para videos 1-7 (Valencia positiva) y rojos para 8-14 (Valencia negativa).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Correlación entre Reporte a Posteriori y Promedio del Reporte Continuo', fontsize=16)

    dimensiones = [
        ('Valencia', 'post_valencia', 'continua_valencia_media', 'continua_valencia_desvio'),
        ('Arousal', 'post_arousal', 'continua_arousal_media', 'continua_arousal_desvio')
    ]

    for ax, (titulo, col_post, col_cont, col_std) in zip(axes, dimensiones):
        # Limpiar NaNs temporalmente para cálculos
        df_clean = df.dropna(subset=[col_post, col_cont])
        x = df_clean[col_post]
        y = df_clean[col_cont]
        yerr = df_clean[col_std]
        video_ids = df_clean.index

        # 1. Dibujar la línea diagonal perfecta (y = x)
        ax.plot([0, 10], [0, 10], color='gray', linestyle='--', alpha=0.7, label='Línea de Identidad')

        # 2. Separar por valencia usando máscaras booleanas
        mask_pos = (video_ids >= 1) & (video_ids <= 7)
        mask_neg = (video_ids >= 8) & (video_ids <= 14)

        # 3. Dibujar grupo de Valencia Positiva (Verdes)
        if mask_pos.any():
            ax.errorbar(x[mask_pos], y[mask_pos], yerr=yerr[mask_pos], fmt='o', 
                        color='forestgreen', ecolor='lightgreen', elinewidth=2, capsize=4, 
                        markersize=7, alpha=0.9, label='Valencia positiva')

        # 4. Dibujar grupo de Valencia Negativa (Rojos)
        if mask_neg.any():
            ax.errorbar(x[mask_neg], y[mask_neg], yerr=yerr[mask_neg], fmt='o', 
                        color='crimson', ecolor='lightcoral', elinewidth=2, capsize=4, 
                        markersize=7, alpha=0.9, label='Valencia negativa')

        # 5. Anotar cada punto con el ID del video
        for i, vid in enumerate(video_ids):
            ax.annotate(str(vid), (x.iloc[i], y.iloc[i]), textcoords="offset points", 
                        xytext=(8,-5), ha='left', fontsize=10, color='black')

        # 6. Calcular estadística de correlación (Pearson)
        if len(x) > 1:
            r, p_val = stats.pearsonr(x, y)
            ax.set_title(f'{titulo}\n(Pearson r = {r:.2f}, p = {p_val:.3f})')
        else:
            ax.set_title(f'{titulo}')

        # 7. Formato del gráfico
        ax.set_xlim(0.5, 9.5)
        ax.set_ylim(0, 10)
        ax.set_xlabel(f'Reporte a Posteriori (1-9)')
        ax.set_ylabel(f'Reporte Continuo Promedio (1-9)')
        ax.set_xticks(range(1, 10))
        ax.set_yticks(range(1, 10))
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left')

    plt.tight_layout()
    
    # Guardar figura en carpeta results con nomenclatura BIDS
    fig_name = f"sub-{SUBJECT_ID}_desc-scattercorrelation_fig.png"
    fig_path = os.path.join(RESULTS_PATH, fig_name)
    
    plt.savefig(fig_path, dpi=300)
    print(f"💾 Scatter Plot guardado en: {fig_path}")
    plt.show()


def plot_paired_differences(df):
    """
    Idea 2: Gráfico de líneas emparejadas para ver el cambio de Post a Continuo.
    Todas las líneas comparten el mismo color default.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Comparación de Medias: Reporte Post vs Continuo', fontsize=16)

    dimensiones = [
        ('Valencia', 'post_valencia', 'continua_valencia_media'),
        ('Arousal', 'post_arousal', 'continua_arousal_media')
    ]

    for ax, (titulo, col_post, col_cont) in zip(axes, dimensiones):
        df_clean = df.dropna(subset=[col_post, col_cont])
        post_vals = df_clean[col_post]
        cont_vals = df_clean[col_cont]
        video_ids = df_clean.index

        # Dibujar líneas conectando los puntos emparejados (mismo color para todos)
        for i in range(len(post_vals)):
            ax.plot([0, 1], [post_vals.iloc[i], cont_vals.iloc[i]], color='steelblue', alpha=0.7, marker='o')
            # Poner el ID del video a la izquierda del punto Post
            ax.text(-0.05, post_vals.iloc[i], str(video_ids[i]), ha='right', va='center', fontsize=9, color='dimgray')

        # Calcular diferencias estadísticas (Prueba de rangos con signo de Wilcoxon)
        try:
            stat, p_val = stats.wilcoxon(post_vals, cont_vals)
            stat_text = f"Wilcoxon p-value = {p_val:.3f}"
            if p_val < 0.05:
                stat_text += "\n(Diferencia Significativa *)"
            else:
                stat_text += "\n(Sin diferencia significativa)"
        except ValueError:
            stat_text = "N insuficiente para estadística"

        # Formato del gráfico
        ax.set_title(f'{titulo}\n{stat_text}')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Reporte Post', 'Reporte Continuo Prom.'])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0.5, 9.5)
        ax.set_yticks(range(1, 10))
        ax.set_ylabel('Puntaje en Escala (1-9)')
        ax.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    
    # Guardar figura en carpeta results con nomenclatura BIDS
    fig_name = f"sub-{SUBJECT_ID}_desc-pairedwilcoxon_fig.png"
    fig_path = os.path.join(RESULTS_PATH, fig_name)
    
    plt.savefig(fig_path, dpi=300)
    print(f"💾 Wilcoxon Plot guardado en: {fig_path}")
    plt.show()


# --- SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    if not os.path.exists(TSV_PATH):
        print(f"ERROR: No se encontró el archivo '{TSV_PATH}'.")
        print("Asegúrate de haber corrido el script anterior que genera el TSV en la carpeta results.")
    else:
        # Cargar los datos TSV (tab separated values)
        print(f"Cargando datos desde {TSV_PATH}...")
        df = pd.read_csv(TSV_PATH, sep='\t', index_col='video_id')
        
        print("Generando Gráfico 1: Dispersión y Correlación...")
        plot_correlation_scatter(df)
        
        print("Generando Gráfico 2: Líneas Emparejadas y Diferencias Estadísticas...")
        plot_paired_differences(df)