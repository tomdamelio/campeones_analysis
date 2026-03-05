import os
import pandas as pd
import numpy as np
import mne
import neurokit2 as nk
import biosppy
import matplotlib.pyplot as plt
import json

# --- CONFIGURACIÓN ---
SUBJECT_ID = "27"
BASE_PATH = rf"data/derivatives/campeones_preproc/sub-{SUBJECT_ID}/ses-vr/eeg"
OUTPUT_PATH = rf"data/derivatives/eda_preproc_tests/sub-{SUBJECT_ID}"
SOURCEDATA_PATH = rf"data/sourcedata/xdf/sub-{SUBJECT_ID}"

RESULTS_PATH = rf"results/eda_preproc_tests/sub-{SUBJECT_ID}/physio/features_timeseries"
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# Nombre del canal de EDA. ¡Asegúrate de que coincida con tus archivos VHDR!
EDA_CHANNEL = 'GSR' 

# =============================================================================
# FUNCIONES DE PREPROCESAMIENTO EDA (COPIADAS DEL SCRIPT ORIGINAL)
# =============================================================================

def process_eda_emotiphai(eda_signal, sampling_rate=250.0, min_amplitude=0.05):
    """
    Process EDA signal using BioSPPy emotiphai_eda method to extract SCR features.
    (Función original de preprocess_phys.py)
    """
    try:
        # BioSPPy para la señal limpia
        eda_result = biosppy.signals.eda.eda(
            signal=np.array(eda_signal), 
            sampling_rate=sampling_rate, 
            show=False
        )
        filtered_signal = eda_result['filtered']
        
        # Aplicar el método emotiphai
        emotiphai_result = biosppy.signals.eda.emotiphai_eda(
            signal=filtered_signal,
            sampling_rate=sampling_rate,
            min_amplitude=min_amplitude
        )
        
        # Extraer resultados (adaptado para manejar la salida de BioSPPy)
        if hasattr(emotiphai_result, 'onsets'):
            onsets = emotiphai_result.onsets
        else:
            onsets = emotiphai_result[0]
        
        print(f"   ✅ Emotiphai: {len(onsets)} SCRs detectados.")
        return emotiphai_result
        
    except Exception as e:
        print(f"   ❌ Emotiphai EDA processing failed: {e}")
        return None

def process_eda_cvx_decomposition(eda_signal, sampling_rate=250.0):
    """
    Process EDA signal using BioSPPy cvx_decomposition method to extract EDA components.
    (Función original de preprocess_phys.py)
    """
    try:
        # Aplicar el método CVX
        cvx_result = biosppy.signals.eda.cvx_decomposition(
            signal=np.array(eda_signal),
            sampling_rate=sampling_rate
        )
        
        # Extraer componentes (adaptado para manejar la salida de BioSPPy)
        if hasattr(cvx_result, 'edr'):
            edr, smna, edl = cvx_result.edr, cvx_result.smna, cvx_result.edl
        else:
            edr, smna, edl = cvx_result[0], cvx_result[1], cvx_result[2]

        # Crear DataFrame
        cvx_df = pd.DataFrame({'EDR': edr, 'SMNA': smna, 'EDL': edl})
        
        print(f"   ✅ CVX Decomposition: {len(cvx_df)} muestras procesadas.")
        return cvx_df
        
    except Exception as e:
        print(f"   ❌ CVX decomposition processing failed: {e}")
        # Retornar DataFrame con ceros para no detener el script
        empty_df = pd.DataFrame({
            'EDR': np.zeros(len(eda_signal)),
            'SMNA': np.zeros(len(eda_signal)),
            'EDL': np.zeros(len(eda_signal))
        })
        return empty_df

# =============================================================================
# FUNCIÓN PRINCIPAL DE PROCESAMIENTO Y VISUALIZACIÓN
# =============================================================================

def process_and_plot_eda_block(eeg_filepath, eda_channel_name=EDA_CHANNEL):
    """
    Carga, procesa (usando las funciones requeridas) y visualiza la señal EDA de un bloque.
    """
    base_filename = os.path.basename(eeg_filepath)
    events_filepath = eeg_filepath.replace('_eeg.vhdr', '_events.tsv')
    
    parts = base_filename.split('_')
    task_id = [p for p in parts if 'task-' in p][0].split('-')[1]
    acq_id = [p for p in parts if 'acq-' in p][0].split('-')[1].upper()

    print("\n" + "=" * 60)
    print(f"🔄 Procesando Sujeto {SUBJECT_ID} | Sesión: {acq_id} | Bloque: {task_id}")

    # 1. CARGAR DATOS RAW Y EVENTOS
    try:
        raw = mne.io.read_raw_brainvision(eeg_filepath, preload=True, verbose=False)
        events_df = pd.read_csv(events_filepath, sep='\t')
    except Exception as e:
        print(f"  ❌ ERROR: No se pudo cargar el EEG o el .tsv: {e}")
        return

    sfreq = raw.info['sfreq'] 
    print(f"  ✅ Señal extraída. Fs: {sfreq} Hz.")

    if eda_channel_name not in raw.ch_names:
        print(f"  ❌ ERROR: El canal '{eda_channel_name}' no existe. Canales disponibles: {raw.ch_names}")
        return

    sfreq = raw.info['sfreq']
    eda_data, times = raw.get_data(picks=[eda_channel_name], return_times=True)
    eda_raw = eda_data[0]
    print(f"  ✅ Señal extraída. Fs: {sfreq} Hz. Duración: {times[-1]:.2f} s")

    video_ids = []
    task_num = int(task_id) # Convierte '01' a 1 para que el nombre coincida
    excel_filename = f"order_matrix_{SUBJECT_ID}_{acq_id}_block{task_num}_VR.xlsx" 
    excel_filepath = os.path.join(SOURCEDATA_PATH, excel_filename)
    
    video_ids = []
    try:
        excel_df = pd.read_excel(excel_filepath)
        video_ids = excel_df["video_id"].dropna().tolist()
        print(f"  ✅ Archivo de diseño encontrado. IDs de video cargados.")
    except FileNotFoundError:
        print(f"  -> AVISO: No se encontró el Excel. Las etiquetas de 'video' serán genéricas.")
    except KeyError:
        print(f"  -> AVISO: No se encontró la columna 'video_id' en el Excel.")

# 2. PROCESAMIENTO EDA
    print("  🧠 Procesando con NeuroKit2 (Clean, Tonic, Phasic)...")
    nk_signals, _ = nk.eda_process(eda_raw, sampling_rate=sfreq)
    eda_clean = nk_signals['EDA_Clean'].values

    print("  ⚙️  Procesando cvx_decomposition por estímulo (con 5s de padding)...")
    
    # Inicializar vectores con ceros del tamaño de la señal original
    eda_smna = np.full_like(eda_raw, np.nan)
    eda_tonic = np.full_like(eda_raw, np.nan)
    eda_phasic = np.full_like(eda_raw, np.nan)
    
    # Inicializar columna de marcas con strings vacíos
    marks_col = np.full(len(eda_raw), "", dtype=object)
    
    # Iterar sobre los eventos para extraer, padear, procesar y descartar el padding
    for _, row in events_df.iterrows():
        trial_type = str(row['trial_type'])
        
        # Omitimos eventos que no sean estímulos
        if trial_type not in ['bad', 'fixation']:
            onset = row['onset']
            offset = onset + row['duration']
            
            # Definir tiempos con 5 segundos previos y posteriores (respetando los límites de la señal)
            onset_pad = max(0.0, onset - 5.0)
            offset_pad = min(times[-1], offset + 5.0)
            
            # Convertir tiempos a índices
            idx_start_pad = int(onset_pad * sfreq)
            idx_end_pad = int(offset_pad * sfreq)
            idx_start_stim = int(onset * sfreq)
            idx_end_stim = int(offset * sfreq)
            
            # --- LOGICA DE MARCAS (ONSET/OFFSET) ---
            # Aseguramos que los índices estén dentro de los límites del array
            if 0 <= idx_start_stim < len(marks_col):
                marks_col[idx_start_stim] = "onset"
            
            # Para el offset, usamos min() para evitar salirnos del array si el evento termina justo al final
            if 0 <= idx_end_stim < len(marks_col):
                marks_col[idx_end_stim] = "offset"
            elif idx_end_stim == len(marks_col): # Si cae justo en el último sample + 1
                marks_col[-1] = "offset"

            # --- EXTRACT & PROCESS CVX ---
            # Extraer segmento con padding
            eda_chunk = eda_raw[idx_start_pad:idx_end_pad]
            
            if len(eda_chunk) > 0:
                print(f"     -> CVX: Evento '{trial_type}' en {onset:.1f}s")
                cvx_df_chunk = process_eda_cvx_decomposition(eda_chunk, sampling_rate=sfreq)
                
                # Índices relativos dentro del chunk extraído para descartar el padding
                inner_start = idx_start_stim - idx_start_pad
                inner_end = idx_end_stim - idx_start_pad
                
                # Longitud real a insertar para evitar descuadres por redondeo
                len_to_insert = min(idx_end_stim - idx_start_stim, inner_end - inner_start)
                
                # Guardar el segmento SIN padding en la señal global
                # Verificamos que len_to_insert sea positivo para evitar errores
                if len_to_insert > 0:
                    eda_smna[idx_start_stim:idx_start_stim + len_to_insert] = cvx_df_chunk['SMNA'].values[inner_start:inner_start + len_to_insert]
                    eda_tonic[idx_start_stim:idx_start_stim + len_to_insert] = cvx_df_chunk['EDL'].values[inner_start:inner_start + len_to_insert]
                    eda_phasic[idx_start_stim:idx_start_stim + len_to_insert] = cvx_df_chunk['EDR'].values[inner_start:inner_start + len_to_insert]

    # --- GUARDAR DATAFRAME (FUERA DEL BUCLE) ---
    df_to_save = pd.DataFrame({
        'Time': times,
        'Raw': eda_raw,
        'Clean': eda_clean,
        'Tonic': eda_tonic,
        'Phasic': eda_phasic,
        'SMNA': eda_smna,
        'Marks': marks_col  # Nueva columna
    })
    
    bids_filename = f'sub-{SUBJECT_ID}_ses-{acq_id}_task-{task_id}_desc-edapreproc_physio'
    
    # 1. Guardar como TSV (separado por tabulaciones)
    output_tsv = os.path.join(OUTPUT_PATH, f'{bids_filename}.tsv')
    df_to_save.to_csv(output_tsv, sep='\t', index=False)
    
    # 2. Crear y guardar el diccionario JSON de metadatos (Sidecar)
    bids_metadata = {
        "SamplingFrequency": sfreq,
        "StartTime": 0.0,
        "Columns": list(df_to_save.columns),
        "Description": "Processed EDA signals including NeuroKit and CVX decomposition",
        "RawEDAChannel": eda_channel_name
    }
    
    output_json = os.path.join(OUTPUT_PATH, f'{bids_filename}.json')
    with open(output_json, 'w') as f:
        json.dump(bids_metadata, f, indent=4)
        
    print(f"  💾 Señales procesadas guardadas en BIDS: {output_tsv} (+ .json)")
    # Se ejecuta para cumplir el requisito, pero sus resultados no se grafican como señal
    _ = process_eda_emotiphai(eda_raw, sampling_rate=sfreq)

    # 3. VISUALIZACIÓN SECUENCIAL
    print("  📊 Generando visualización...")
    fig, axes = plt.subplots(5, 1, figsize=(18, 12), sharex=True)
    fig.suptitle(f"Actividad Electrodérmica (EDA) | Sujeto {SUBJECT_ID} - Sesión {acq_id} - Bloque {task_id}", fontsize=16)

    signals_to_plot = [
        (eda_raw, 'Raw', 'gray'),
        (eda_clean, 'Clean (NeuroKit)', 'black'),
        (eda_tonic, 'Tonic (cvx)', 'blue'),
        (eda_phasic, 'Phasic (cvx)', 'orange'),
        (eda_smna, 'SMNA', 'purple')
    ]

    for ax, (signal, title, color) in zip(axes, signals_to_plot):
            ax.plot(times, signal, color=color, linewidth=1)
            ax.set_ylabel(title, fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.6)
            
            video_counter = 0 # Reiniciamos el contador para cada gráfico
            for _, row in events_df.iterrows():
                onset = row['onset']
                offset = onset + row['duration']
                trial_type = str(row['trial_type'])
                
                # Sombreado para TODOS los eventos del .tsv (calm, luminance, video)
                if trial_type not in ['bad', 'fixation']:
                    if trial_type == 'video':
                        ax.axvspan(onset, offset, color='orange', alpha=0.1)

                    if trial_type == 'calm':
                        ax.axvspan(onset, offset, color='blue', alpha=0.1)

                    if trial_type == 'video_luminance':
                        ax.axvspan(onset, offset, color='green', alpha=0.1)
                
                if ax == axes[0]: # Poner etiqueta solo en el primer plot
                    label_to_plot = trial_type if trial_type not in ['bad', 'fixation'] else ''
                    
                    # Si el evento es un video, le agregamos el ID del Excel
                    if trial_type == 'video':
                        if video_counter < len(video_ids):
                            label_to_plot = f"video {int(video_ids[video_counter])}"
                        video_counter += 1
                    
                    ax.text(onset, np.max(signal), f" {label_to_plot}", rotation=45, ha='left', va='bottom', fontsize=9)

    axes[-1].set_xlabel("Tiempo (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajustar para el supertítulo
    
    print("  👁️  Mostrando gráfico. Cierra la ventana para procesar el siguiente bloque...")
    
    # Guardar figura en carpeta results con nomenclatura BIDS
    fig_bids_name = f"sub-{SUBJECT_ID}_ses-{acq_id}_task-{task_id}_desc-edatimeseries_fig.png"
    fig_path = os.path.join(RESULTS_PATH, fig_bids_name)
    
    plt.savefig(fig_path, dpi=300)
    print(f"  📊 Gráfico guardado en: {fig_path}")
    
    plt.show(block=True)

# --- SCRIPT DE EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    if not os.path.exists(BASE_PATH):
        print(f"❌ El directorio no existe: {BASE_PATH}")
    else:
        files_to_process = sorted([
            os.path.join(BASE_PATH, f) for f in os.listdir(BASE_PATH) 
            if f.endswith("_desc-preproc_eeg.vhdr")
        ])

        if not files_to_process:
            print(f"❌ No se encontraron archivos EEG preprocesados en: {BASE_PATH}")
        else:
            print(f"✅ Se encontraron {len(files_to_process)} bloques para el sujeto {SUBJECT_ID}.")
            for eeg_file in files_to_process:
                process_and_plot_eda_block(eeg_file)
        
        print("\n" + "=" * 60)
        print("🎉 ¡Proceso completado! Se han visualizado todos los bloques.")