import mne
import os
import pandas as pd
import numpy as np

# --- CONFIGURACIÓN ---
BASE_PATH = r"data/derivatives/campeones_preproc/sub-27/ses-vr/eeg"
SOURCEDATA_PATH = r"data/sourcedata/xdf/sub-27"
CHANNEL_TO_EXTRACT = 'joystick_x'

# 1. DATOS A POSTERIORI (Sujeto 27)
post_reports_raw = {
    3:  {'val': 8, 'aro': 7},
    12: {'val': 2, 'aro': 8},
    2:  {'val': 7, 'aro': 7},
    9:  {'val': 4, 'aro': 6},
    7:  {'val': 4, 'aro': 6},
    8:  {'val': 5, 'aro': 5},
    4:  {'val': 7, 'aro': 3},
    6:  {'val': 9, 'aro': 2},
    13: {'val': 4, 'aro': 5},
    14: {'val': 4, 'aro': 4},
    5:  {'val': 9, 'aro': 6},
    11: {'val': 5, 'aro': 7},
    10: {'val': 3, 'aro': 7},
    1:  {'val': 8, 'aro': 3}
}

def prepare_post_data():
    """Crea un DataFrame con los reportes post manteniendo la escala original 1 a 9."""
    df_post = pd.DataFrame.from_dict(post_reports_raw, orient='index')
    df_post.index.name = 'video_id'
    
    # Mantenemos los valores sin normalizar (ya están en 1-9)
    df_post['post_valencia'] = df_post['val']
    df_post['post_arousal'] = df_post['aro']
    
    # Descartamos las columnas originales
    df_post.drop(columns=['val', 'aro'], inplace=True)
    return df_post

def process_continuous_files(base_path, sourcedata_path, channel):
    """Itera sobre los archivos EEG, convierte la señal a escala 1-9 y calcula las métricas por video."""
    all_files = os.listdir(base_path)
    files_to_process = [
        os.path.join(base_path, f) for f in all_files 
        if f.endswith("_desc-preproc_eeg.vhdr")
    ]
    
    # Diccionario para guardar métricas: {video_id: {metricas...}}
    continuous_stats = {}

    for eeg_filepath in files_to_process:
        base_filename = os.path.basename(eeg_filepath)
        events_filepath = eeg_filepath.replace('_eeg.vhdr', '_events.tsv')
        
        parts = base_filename.split('_')
        subject_id = parts[0].split('-')[1]
        task_id = parts[2]
        acq_id = parts[3]
        task_num = int(task_id.split('-')[1])
        acq_char = acq_id.split('-')[1].upper()

        # Cargar Raw y Eventos
        try:
            raw = mne.io.read_raw_brainvision(eeg_filepath, preload=True, verbose=False)
            events_df = pd.read_csv(events_filepath, sep='\t')
        except Exception as e:
            print(f"Error cargando {base_filename}: {e}")
            continue

        if channel not in raw.ch_names:
            print(f"Canal {channel} no encontrado en {base_filename}")
            continue

        # Leer Excel de diseño
        excel_filename = f"order_matrix_{subject_id}_{acq_char}_block{task_num}_VR.xlsx"
        excel_filepath = os.path.join(sourcedata_path, excel_filename)
        
        try:
            excel_df = pd.read_excel(excel_filepath)
            
            inversion_instruction = excel_df['order_emojis_slider'].dropna().tolist()
            inversion_instruction = inversion_instruction[0] if inversion_instruction else None
            
            dimension_en = "valence" if "valence" in excel_df["dimension"].dropna().tolist() else "arousal"
            dim_es = "valencia" if dimension_en == "valence" else "arousal"
            
            video_ids = excel_df["video_id"].dropna().tolist()
        except Exception as e:
            print(f"Error leyendo Excel para {base_filename}: {e}")
            continue

        # Extraer señal del joystick
        data, times = raw.get_data(picks=[channel], return_times=True)
        joystick_signal = data[0].copy()
        
        # Invertir la señal si el Excel lo indica
        if inversion_instruction == 'inverse':
            joystick_signal *= -1
            
        # Segmentar señal por video
        video_counter = 0
        for _, row in events_df.iterrows():
            if row['trial_type'] == 'video' and video_counter < len(video_ids):
                vid = int(video_ids[video_counter])
                onset = row['onset']
                offset = onset + row['duration']
                
                # Crear máscara de tiempo para recortar la señal
                mask = (times >= onset) & (times <= offset)
                segment = joystick_signal[mask]
                
                # Transformar la señal continua de [-1, 1] a [1, 9] ANTES de sacar las métricas
                segment_scaled = (segment * 4) + 5
                
                # Calcular métricas si hay datos en el segmento
                if len(segment_scaled) > 0:
                    if vid not in continuous_stats:
                        continuous_stats[vid] = {}
                    
                    continuous_stats[vid][f'continua_{dim_es}_media'] = np.mean(segment_scaled)
                    continuous_stats[vid][f'continua_{dim_es}_mediana'] = np.median(segment_scaled)
                    continuous_stats[vid][f'continua_{dim_es}_desvio'] = np.std(segment_scaled)
                
                video_counter += 1

    # Convertir diccionario a DataFrame
    df_cont = pd.DataFrame.from_dict(continuous_stats, orient='index')
    df_cont.index.name = 'video_id'
    return df_cont

# --- SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    print("1. Procesando reportes a posteriori (manteniendo escala 1-9)...")
    df_post = prepare_post_data()
    
    print("2. Procesando reportes continuos (llevando de [-1, 1] a [1, 9])...")
    df_cont = process_continuous_files(BASE_PATH, SOURCEDATA_PATH, CHANNEL_TO_EXTRACT)
    
    print("3. Uniendo datos y formateando tabla final...")
    # Unir ambas tablas alineando por el 'video_id' (index)
    df_final = df_post.join(df_cont, how='outer')
    
    # Asegurar que todas las columnas existan, rellenando con NaN si falta alguna temporalmente
    columnas_deseadas = [
        "post_valencia", "post_arousal", 
        "continua_valencia_media", "continua_arousal_media", 
        "continua_valencia_mediana", "continua_arousal_mediana", 
        "continua_valencia_desvio", "continua_arousal_desvio"
    ]
    
    for col in columnas_deseadas:
        if col not in df_final.columns:
            df_final[col] = np.nan
            
    # Ordenar columnas según tu requerimiento
    df_final = df_final[columnas_deseadas]
    
    # Redondear para mejorar visualización
    df_final = df_final.round(4)
    
    print("-" * 60)
    print("TABLA FINAL RESULTANTE:")
    print("-" * 60)
    print(df_final)
    
    df_final.to_csv(f"sub-27_all_reports.csv")
    print("Guardado en 'sub-27_all_reports.csv'")