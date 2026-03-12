import mne
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy import signal

# --- CONFIGURACIÓN ---
SUBJECT_ID = "30"
BASE_PATH = rf"data/derivatives/campeones_preproc/sub-{SUBJECT_ID}/ses-vr/eeg"
SOURCEDATA_PATH = rf"data/sourcedata/xdf/sub-{SUBJECT_ID}"
RESULTS_PATH = rf"results/eda_preproc_tests/sub-{SUBJECT_ID}/beh"
os.makedirs(RESULTS_PATH, exist_ok=True)

CHANNEL_TO_EXTRACT = 'joystick_x'
VIDEOS_TO_PLOT = [2, 14]  # Los videos solicitados

def get_raw_trace_for_video(video_id, target_dimension, base_path, sourcedata_path, channel):
    """
    Busca en todos los archivos EEG el bloque que contenga el video_id 
    Y que corresponda a la dimensión target ('valence' o 'arousal').
    Retorna el array de datos (escalado 1-9) o None si no lo encuentra.
    """
    all_files = os.listdir(base_path)
    files_to_process = [f for f in all_files if f.endswith("_desc-preproc_eeg.vhdr")]
    
    for filename in files_to_process:
        # Parsear nombres
        parts = filename.split('_')
        subject_id = parts[0].split('-')[1]
        task_id = parts[2]
        acq_id = parts[3]
        task_num = int(task_id.split('-')[1])
        acq_char = acq_id.split('-')[1].upper()
        
        # Leer Excel de diseño
        excel_filename = f"order_matrix_{subject_id}_{acq_char}_block{task_num}_VR.xlsx"
        excel_path = os.path.join(sourcedata_path, excel_filename)
        
        try:
            excel_df = pd.read_excel(excel_path)
            
            # Verificar dimensión del bloque
            dim_in_file = "valence" if "valence" in excel_df["dimension"].dropna().tolist() else "arousal"
            if dim_in_file != target_dimension:
                continue # Este archivo no tiene la dimensión que buscamos
            
            # Verificar si el video está en este bloque
            video_ids = excel_df["video_id"].dropna().tolist()
            if video_id not in video_ids:
                continue # El video no está en este bloque

            # Si llegamos acá, encontramos el archivo correcto. Cargamos EEG.
            eeg_path = os.path.join(base_path, filename)
            raw = mne.io.read_raw_brainvision(eeg_path, preload=True, verbose=False)
            events_path = eeg_path.replace('_eeg.vhdr', '_events.tsv')
            events_df = pd.read_csv(events_path, sep='\t')
            
            # Inversión?
            inv_instr = excel_df['order_emojis_slider'].dropna().tolist()
            invert = True if (inv_instr and inv_instr[0] == 'inverse') else False
            
            # Extraer datos
            data, times = raw.get_data(picks=[channel], return_times=True)
            signal_raw = data[0]
            if invert:
                signal_raw *= -1
            
            # Recortar el segmento del video
            # Necesitamos saber qué índice de aparición tiene el video en el excel para matchearlo con el TSV
            try:
                # Filtrar eventos que son solo 'video'
                video_events = events_df[events_df['trial_type'] == 'video'].reset_index(drop=True)
                # Encontrar el índice del video en la lista de IDs
                idx_video = video_ids.index(video_id)
                
                if idx_video >= len(video_events):
                    print(f"Advertencia: Índice fuera de rango para video {video_id}")
                    return None
                
                row = video_events.iloc[idx_video]
                onset = row['onset']
                offset = onset + row['duration']
                
                mask = (times >= onset) & (times <= offset)
                segment = signal_raw[mask]
                
                # Escalar a 1-9
                segment_scaled = (segment * 4) + 5
                
                return segment_scaled

            except ValueError:
                continue
                
        except Exception as e:
            print(f"Error procesando {filename}: {e}")
            continue
            
    return None

def plot_affective_trajectories(videos, base_path, source_path, channel):
    """
    Genera plots de trayectorias (Valencia vs Arousal) coloreadas por tiempo.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    if len(videos) == 1: axes = [axes] # Manejo por si es un solo video
    
    fig.suptitle(f'Trayectoria Afectiva Continua (Momento a Momento)', fontsize=16)

    for ax, vid_id in zip(axes, videos):
        print(f"Procesando Video {vid_id}...")
        
        # 1. Obtener trazas crudas
        val_trace = get_raw_trace_for_video(vid_id, 'valence', base_path, source_path, channel)
        aro_trace = get_raw_trace_for_video(vid_id, 'arousal', base_path, source_path, channel)
        
        if val_trace is None or aro_trace is None:
            ax.text(0.5, 0.5, "Datos incompletos", ha='center')
            print(f" -> Falta alguna dimensión para el video {vid_id}")
            continue
            
        # 2. Sincronización (Resample)
        # Como se grabaron en sesiones distintas, pueden diferir en unos pocos samples.
        # Ajustamos ambos al tamaño del más corto para evitar errores, o definimos un N fijo.
        n_samples = min(len(val_trace), len(aro_trace))
        
        # Usamos resample para ajustar suavemente ambas señales al mismo largo exacto
        val_resampled = signal.resample(val_trace, n_samples)
        aro_resampled = signal.resample(aro_trace, n_samples)
                
        # 3. Crear segmentos para la "LineCollection" multicolor
        # x = valencia, y = arousal
        points = np.array([val_resampled, aro_resampled]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Normalizar tiempo para el color (0 a 1)
        norm = plt.Normalize(0, 1)
        t = np.linspace(0, 1, len(val_resampled))
        
        # Usamos el colormap 'viridis' o 'plasma' (Amarillo al final suele indicar "llegada")
        lc = LineCollection(segments, cmap='turbo', norm=norm)
        lc.set_array(t)
        lc.set_linewidth(3) # Línea gruesa
        
        # 4. Plotear
        line = ax.add_collection(lc)
        
        # Marcadores de Inicio y Fin
        ax.scatter(val_resampled[0], aro_resampled[0], color='black', s=100, marker='o', label='Inicio', zorder=5)
        ax.text(val_resampled[0], aro_resampled[0], ' INICIO', color='black', fontsize=9, va='center', fontweight='bold')
        
        ax.scatter(val_resampled[-1], aro_resampled[-1], color='red', s=150, marker='X', label='Fin', zorder=5)
        ax.text(val_resampled[-1], aro_resampled[-1], ' FIN', color='red', fontsize=9, va='center', fontweight='bold')

        # 5. Configuración del Eje
        ax.set_title(f"Video {vid_id}", fontsize=14, weight='bold')
        ax.set_xlim(1, 9)
        ax.set_ylim(1, 9)
        ax.set_xlabel('VALENCIA')
        ax.set_ylabel('AROUSAL')
        
        # Líneas centrales
        ax.axhline(5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(5, color='gray', linestyle='--', alpha=0.5)
                
        ax.grid(True, linestyle=':', alpha=0.4)

    # Barra de color compartida
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    cbar = fig.colorbar(lc, cax=cbar_ax)
    cbar.set_label('Tiempo (Progreso del Video)', rotation=270, labelpad=15)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Inicio (0%)', 'Fin (100%)'])

    plt.subplots_adjust(right=0.9) # Espacio para la colorbar
    
    # Guardar figura en carpeta results con nomenclatura BIDS
    fig_name = f"sub-{SUBJECT_ID}_desc-affectivetrajectories_fig.png"
    fig_path = os.path.join(RESULTS_PATH, fig_name)
    
    plt.savefig(fig_path, dpi=300)
    print(f"💾 Trayectorias guardadas en: {fig_path}")
    plt.show()

# --- MAIN ---
if __name__ == "__main__":
    print(f"Extrayendo trazas continuas y generando trayectorias para videos: {VIDEOS_TO_PLOT}")
    plot_affective_trajectories(VIDEOS_TO_PLOT, BASE_PATH, SOURCEDATA_PATH, CHANNEL_TO_EXTRACT)