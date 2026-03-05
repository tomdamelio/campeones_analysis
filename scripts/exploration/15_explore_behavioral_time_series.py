import mne
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

# --- CONFIGURACIÓN ---
SUBJECT_ID = "27"
BASE_PATH = rf"data/derivatives/campeones_preproc/sub-{SUBJECT_ID}/ses-vr/eeg"
SOURCEDATA_PATH = rf"data/sourcedata/xdf/sub-{SUBJECT_ID}"

CHANNEL_TO_PLOT = 'joystick_x'

def plot_time_series_with_events(eeg_filepath, channel_to_plot):
    """
    Carga un archivo EEG y su .tsv de eventos. Lee un archivo .xlsx de diseño para decidir
    si invierte la señal completa del canal y luego genera el gráfico con etiquetas de video personalizadas.
    """
    # 1. Derivar nombres de archivos y extraer metadatos del nombre de archivo
    base_filename = os.path.basename(eeg_filepath)
    events_filepath = eeg_filepath.replace('_eeg.vhdr', '_events.tsv')
    
    parts = base_filename.split('_')
    subject_id = parts[0].split('-')[1]
    task_id = parts[2]
    acq_id = parts[3]

    print("-" * 50)
    print(f"Procesando: {base_filename}")

    # 2. Cargar datos de EEG y eventos
    try:
        raw = mne.io.read_raw_brainvision(eeg_filepath, preload=True, verbose=False)
        events_df = pd.read_csv(events_filepath, sep='\t')
    except Exception as e:
        print(f"  -> ¡ERROR! No se pudo cargar el archivo EEG o de eventos: {e}")
        return

    # 3. CONSTRUIR RUTA AL EXCEL Y EXTRAER INFO DE DISEÑO
    task_num = int(task_id.split('-')[1])
    acq_char = acq_id.split('-')[1].upper()
    excel_filename = f"order_matrix_{subject_id}_{acq_char}_block{task_num}_VR.xlsx"
    excel_filepath = os.path.join(SOURCEDATA_PATH, excel_filename)

    # Inicializar variables antes del try para evitar errores
    inversion_instruction = None 
    video_ids = []
    dimension = "unknown"

    try:
        print(f"  -> Buscando archivo de diseño: {excel_filename}")
        excel_df = pd.read_excel(excel_filepath)
        
        # Extraer instrucción de inversión
        valid_instructions = excel_df['order_emojis_slider'].dropna().tolist()
        if valid_instructions:
            inversion_instruction = valid_instructions[0]
            print(f"  -> Archivo encontrado. Instrucción de orden: '{inversion_instruction}'")
        else:
            print("  -> Archivo encontrado, pero la columna de orden está vacía.")
            
        # Extraer dimensión y lista de IDs de video
        dimension = "valence" if "valence" in excel_df["dimension"].dropna().tolist() else "arousal"
        video_ids = excel_df["video_id"].dropna().tolist()
            
    except FileNotFoundError:
        print(f"  -> AVISO: No se encontró el archivo Excel. La señal no será invertida y las etiquetas de video serán genéricas.")
    except KeyError as e:
        print(f"  -> AVISO: No se encontró la columna {e} en el Excel.")
    
    # 4. Preparar datos para el gráfico
    if channel_to_plot not in raw.ch_names:
        print(f"  -> AVISO: El canal '{channel_to_plot}' no existe en este archivo.")
        return

    data, times = raw.get_data(picks=[channel_to_plot], return_times=True)
    voltaje_modificado = data[0].copy()
    
    # 5. Lógica de inversión global
    plot_label = f'Canal {channel_to_plot}'
    if inversion_instruction == 'inverse':
        voltaje_modificado *= -1
        plot_label += ' (Invertido)'
        print("  -> La señal ha sido invertida globalmente.")
    else:
        print("  -> La señal se mantiene en su orientación original.")

    # 6. Crear el gráfico
    plt.figure(figsize=(20, 7))
    plt.plot(times, voltaje_modificado, label=plot_label, color='steelblue', linewidth=0.8)

    # 7. (MODIFICADO) Dibujar marcadores de eventos con etiquetas de video personalizadas
    video_counter = 0 # Inicializamos el contador para los videos
    for _, row in events_df.iterrows():
        onset = row['onset']
        offset = row['duration'] + onset
        trial_type = row['trial_type']
        
        label_to_plot = trial_type # Por defecto, la etiqueta es el tipo de trial
        
        # Si el evento es un video y tenemos IDs disponibles
        if trial_type == 'video' and video_counter < len(video_ids):
            # Creamos la etiqueta personalizada
            label_to_plot = f"video {int(video_ids[video_counter])}"
            video_counter += 1 # Incrementamos el contador para el próximo video

        text_y_position = np.max(np.abs(voltaje_modificado)) * 1.05
        color = 'gray' if trial_type == 'bad' else 'green'
        
        plt.axvline(x=onset, color=color, linestyle='--', alpha=0.7)
        # Usamos la nueva etiqueta 'label_to_plot'
        plt.text(onset, text_y_position, label_to_plot, color=color, rotation=45,
                 ha='left', va='bottom', fontsize=9)
        plt.axvline(x=offset, color="red", linestyle='--', alpha=0.7)

    # 8. Configurar y mostrar el gráfico
    title = f"Sujeto {subject_id} - Sesion {acq_char.upper()} - Bloque {task_num} - {dimension}"
    plt.title(title)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Coordenada normalizada del joystick")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# --- SCRIPT PRINCIPAL (Sin cambios) ---
if __name__ == "__main__":
    all_files = os.listdir(BASE_PATH)
    files_to_process = [
        os.path.join(BASE_PATH, f) for f in all_files 
        if f.endswith("_desc-preproc_eeg.vhdr")
    ]
    files_to_process.sort()

    if not files_to_process:
        print(f"No se encontraron archivos '_desc-preproc_eeg.vhdr' en la carpeta: {BASE_PATH}")
    else:
        print(f"Se encontraron {len(files_to_process)} archivos para visualizar.")
        for eeg_file in files_to_process:
            plot_time_series_with_events(eeg_file, CHANNEL_TO_PLOT)
    
    print("-" * 50)
    print("¡Proceso completado! Se han visualizado todos los archivos.")