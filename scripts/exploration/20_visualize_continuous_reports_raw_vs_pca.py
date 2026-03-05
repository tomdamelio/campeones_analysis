import mne
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- CONFIGURACIÓN ---
SUBJECT_ID = "27"
BASE_PATH = rf"data/derivatives/campeones_preproc/sub-{SUBJECT_ID}/ses-vr/eeg"
SOURCEDATA_PATH = rf"data/sourcedata/xdf/sub-{SUBJECT_ID}"
RESULTS_PATH = rf"results/eda_preproc_tests/sub-{SUBJECT_ID}/beh/pca"
os.makedirs(RESULTS_PATH, exist_ok=True)

CHANNELS = ['joystick_x', 'joystick_y']

def process_pca_comparison_3plots(base_path, sourcedata_path):
    """
    Recorre los archivos, extrae X e Y, calcula PCA por video y grafica
    3 subplots: X, Y, y PCA vs X.
    """
    all_files = os.listdir(base_path)
    files_to_process = [f for f in all_files if f.endswith("_desc-preproc_eeg.vhdr")]
    files_to_process.sort()

    for filename in files_to_process:
        # 1. Metadatos del archivo
        parts = filename.split('_')
        subject_id = parts[0].split('-')[1]
        task_id = parts[2]
        acq_id = parts[3]
        task_num = int(task_id.split('-')[1])
        acq_char = acq_id.split('-')[1].upper()
        
        # 2. Cargar Excel de diseño
        excel_filename = f"order_matrix_{subject_id}_{acq_char}_block{task_num}_VR.xlsx"
        excel_path = os.path.join(sourcedata_path, excel_filename)
        
        try:
            excel_df = pd.read_excel(excel_path)
            dimension = "valence" if "valence" in excel_df["dimension"].dropna().tolist() else "arousal"
            video_ids = excel_df["video_id"].dropna().tolist()
            # Instrucción de inversión para el eje X
            inv_instr = excel_df['order_emojis_slider'].dropna().tolist()
            invert_x = True if (inv_instr and inv_instr[0] == 'inverse') else False
            
        except Exception as e:
            print(f"Skipping {filename}: No se pudo leer Excel ({e})")
            continue

        print(f"\nProcesando: Bloque {task_num} ({dimension}) - Archivo: {filename}")

        # 3. Cargar EEG
        try:
            eeg_path = os.path.join(base_path, filename)
            raw = mne.io.read_raw_brainvision(eeg_path, preload=True, verbose=False)
            
            if not all(ch in raw.ch_names for ch in CHANNELS):
                print(f"  -> Falta joystick_x o joystick_y en este archivo.")
                continue
                
            events_path = eeg_path.replace('_eeg.vhdr', '_events.tsv')
            events_df = pd.read_csv(events_path, sep='\t')
            
        except Exception as e:
            print(f"Error cargando EEG: {e}")
            continue

        # 4. Extraer datos continuos
        data, times = raw.get_data(picks=CHANNELS, return_times=True)
        raw_x = data[0]
        raw_y = data[1]

        # 5. Iterar por videos
        video_events = events_df[events_df['trial_type'] == 'video'].reset_index(drop=True)
        
        for i, vid_id in enumerate(video_ids):
            if i >= len(video_events): break
            
            vid_id = int(vid_id)
            row = video_events.iloc[i]
            onset = row['onset']
            offset = onset + row['duration']
            
            # Segmentar
            mask = (times >= onset) & (times <= offset)
            seg_x = raw_x[mask]
            seg_y = raw_y[mask]
            
            if len(seg_x) == 0: continue

            # --- APLICAR PCA ---
            X_combined = np.column_stack((seg_x, seg_y))
            pca = PCA(n_components=1)
            seg_pca = pca.fit_transform(X_combined).flatten()
            
            # --- ALINEACIÓN ---
            # Si el PCA está invertido respecto al movimiento principal de X, lo corregimos
            corr = np.corrcoef(seg_x, seg_pca)[0, 1]
            if corr < 0:
                seg_pca *= -1
            
            # --- INVERSIÓN EXPERIMENTAL ---
            if invert_x:
                seg_x *= -1
                seg_pca *= -1 
                # Nota: No invertimos Y porque Y es "ruido vertical", 
                # no tiene dirección emocional asignada en este diseño 1D.
            
            # Escalar a 1-9
            plot_x = (seg_x * 4) + 5
            plot_y = (seg_y * 4) + 5
            plot_pca = (seg_pca * 4) + 5
            
            expl_var = pca.explained_variance_ratio_[0] * 100
            
            # --- GRAFICAR (3 COLUMNAS) ---
            # sharey=True es clave para comparar amplitudes visualmente
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
            
            # --- PLOT 1: EJE X (TARGET) ---
            axes[0].plot(plot_x, color='steelblue', linewidth=1.5)
            axes[0].set_title(f"EJE X (Horizontal)\nLo que se usó para el análisis")
            axes[0].set_ylabel("Escala Afectiva (1-9)")
            axes[0].grid(True, alpha=0.3)
            axes[0].axhline(5, color='gray', linestyle='--')
            
            # --- PLOT 2: EJE Y (RUIDO VERTICAL) ---
            # Si esta línea oscila mucho, confirma movimiento diagonal
            axes[1].plot(plot_y, color='darksalmon', linewidth=1.5)
            axes[1].set_title(f"EJE Y (Vertical)\nDesplazamiento no deseado")
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(5, color='gray', linestyle='--')
            
            # --- PLOT 3: PCA (MOVIMIENTO REAL) ---
            axes[2].plot(plot_pca, color='purple', linewidth=2, label='PCA (Magnitud Real)', alpha=0.8)
            axes[2].plot(plot_x, color='gray', linewidth=1, linestyle='--', alpha=0.5, label='Eje X (Referencia)')
            
            # Calcular ganancia
            rango_x = plot_x.max() - plot_x.min()
            rango_pca = plot_pca.max() - plot_pca.min()
            ganancia = ((rango_pca - rango_x) / rango_x) * 100 if rango_x > 0 else 0
            
            axes[2].set_title(f"Comparación: PCA vs X\nGanancia de Amplitud: {ganancia:+.1f}%")
            axes[2].legend(loc='upper right')
            axes[2].grid(True, alpha=0.3)
            axes[2].axhline(5, color='gray', linestyle='--')

            # Título general
            plt.suptitle(f"Análisis de Trayectoria - Video {vid_id} - {dimension.upper()} - Var Explicada PC1: {expl_var:.1f}%", fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Guardar figura en carpeta results con nomenclatura BIDS
            fig_name = f"sub-{subject_id}_{acq_id}_{task_id}_desc-video{vid_id}pca_fig.png"
            fig_path = os.path.join(RESULTS_PATH, fig_name)
            
            plt.savefig(fig_path, dpi=300)
            print(f"💾 Gráfico PCA guardado en: {fig_path}")
            plt.show()

# --- MAIN ---
if __name__ == "__main__":
    print("Iniciando análisis de 3 ejes (X, Y, PCA)...")
    process_pca_comparison_3plots(BASE_PATH, SOURCEDATA_PATH)