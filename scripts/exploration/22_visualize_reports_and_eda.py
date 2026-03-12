import os
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
import warnings
from scipy.stats import spearmanr

# Ignorar warnings si el joystick no se mueve (varianza 0 produce NaN en la correlación)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- CONFIGURACIÓN ---
SUBJECT_ID = "28"
BASE_PATH = rf"data/derivatives/campeones_preproc/sub-{SUBJECT_ID}/ses-vr/eeg"
EDA_PREPROC_PATH = rf"data/derivatives/eda_preproc_tests/sub-{SUBJECT_ID}"
SOURCEDATA_PATH = rf"data/sourcedata/xdf/sub-{SUBJECT_ID}"
RESULTS_PATH = rf"results/eda_preproc_tests/sub-{SUBJECT_ID}/physio/multimodal_per_stimulus"
os.makedirs(RESULTS_PATH, exist_ok=True)

JOYSTICK_CHANNEL = 'joystick_x'
MAX_LAG_SECONDS = 5 # Rango máximo para buscar el lag óptimo
SMOOTHING_WINDOW_SECONDS = 1.0 

# =============================================================================
# FUNCIONES DE ANÁLISIS
# =============================================================================

def calculate_time_lagged_correlation(x, y, fs, max_lag_sec=15):
    """
    Calcula la correlación de Pearson entre x (Joystick/Derivada) e y (EDA) en distintos lags.
    Adapta el lag máximo si el video es muy corto.
    """
    # Protección: el lag máximo no puede ser mayor a 1/3 de la duración del video
    max_lag_sec = min(max_lag_sec, (len(x)/fs) / 3)
    max_shift = int(max_lag_sec * fs)
    
    if max_shift == 0:
        return 0, 0, 0 # El video es demasiado corto
        
    lags = range(-max_shift, max_shift + 1)
    corrs = []
    
    for lag in lags:
        if lag == 0:
            c = spearmanr(x, y)[0]
        elif lag > 0:
            c = spearmanr(x[:-lag], y[lag:])[0]
        else:
            c = spearmanr(x[-lag:], y[:lag])[0]
        
        corrs.append(0 if np.isnan(c) else c)
    
    corrs = np.array(corrs)
    lags_sec = np.array(lags) / fs
    
    corr_0 = corrs[max_shift]
    opt_idx = np.argmax(np.abs(corrs)) 
    opt_corr = corrs[opt_idx]
    opt_lag = lags_sec[opt_idx]
    
    return corr_0, opt_corr, opt_lag

# =============================================================================
# FUNCIÓN PRINCIPAL POR BLOQUE Y ESTÍMULO
# =============================================================================

def process_multimodal_per_stimulus(eeg_filepath):
    base_filename = os.path.basename(eeg_filepath)
    events_filepath = eeg_filepath.replace('_eeg.vhdr', '_events.tsv')
    
    parts = base_filename.split('_')
    task_id = [p for p in parts if 'task-' in p][0].split('-')[1]
    acq_id = [p for p in parts if 'acq-' in p][0].split('-')[1].upper()
    task_num = int(task_id)

    print("\n" + "=" * 80)
    print(f"🔄 Multimodal | Sujeto {SUBJECT_ID} | Sesión: {acq_id} | Bloque: {task_id}")

    # 1. CARGA DE DATOS EEG (JOYSTICK) Y EVENTOS
    try:
        raw = mne.io.read_raw_brainvision(eeg_filepath, preload=True, verbose=False)
        events_df = pd.read_csv(events_filepath, sep='\t')
        sfreq = raw.info['sfreq']
    except Exception as e:
        print(f"  ❌ ERROR: No se pudo cargar EEG o Eventos: {e}")
        return

    if JOYSTICK_CHANNEL not in raw.ch_names:
        print(f"  ❌ ERROR: Falta Joystick. Canales: {raw.ch_names}")
        return

    # 2. CARGA DE EDA PREPROCESADA (CSV)

    eda_tsv_filename = os.path.join(EDA_PREPROC_PATH, f'sub-{SUBJECT_ID}_ses-{acq_id}_task-{task_id}_desc-edapreproc_physio.tsv')
    if not os.path.exists(eda_tsv_filename):
        print(f"  ❌ ERROR: No se encontró el archivo BIDS de la EDA: {eda_tsv_filename}")
        print(f"     Asegúrate de haber ejecutado el script puro de EDA primero.")
        return
        
    # Importante: agregar sep='\t' porque ahora es un archivo TSV
    eda_df = pd.read_csv(eda_tsv_filename, sep='\t')
    eda_tonic = eda_df['Tonic'].values
    eda_phasic = eda_df['Phasic'].values
    times = eda_df['Time'].values # Usamos el vector de tiempo guardado

    # 3. LECTURA DE DISEÑO (EXCEL)
    excel_filename = f"order_matrix_{SUBJECT_ID}_{acq_id}_block{task_num}_VR.xlsx"
    excel_filepath = os.path.join(SOURCEDATA_PATH, excel_filename)
    
    inversion_instruction = None
    dimension = "Unknown"
    video_ids = []

    try:
        excel_df = pd.read_excel(excel_filepath)
        dimension = "Valence" if "valence" in excel_df["dimension"].dropna().tolist() else "Arousal"
        video_ids = excel_df["video_id"].dropna().tolist()
        
        valid_instructions = excel_df['order_emojis_slider'].dropna().tolist()
        if valid_instructions:
            inversion_instruction = valid_instructions[0]
            
        print(f"  ✅ Diseño: Dimensión '{dimension}' | Orden: '{inversion_instruction}'")
    except Exception as e:
        print(f"  ⚠️ AVISO: No se pudo leer el Excel. Error: {e}")

    # Extraer y preparar Joystick
    joy_data = raw.get_data(picks=[JOYSTICK_CHANNEL])[0]
    joy_label = f"Joystick ({dimension})"
    if inversion_instruction == 'inverse':
        joy_data = joy_data * -1
        joy_label += " [Invertido]"

    # 4. ITERACIÓN POR ESTÍMULO (Segmentación)
    video_counter = 0
    for _, row in events_df.iterrows():
        if row['trial_type'] == 'video':
            vid_id = int(video_ids[video_counter]) if video_counter < len(video_ids) else "N/A"
            video_counter += 1
            
            onset = row['onset']
            duration = row['duration']
            offset = onset + duration
            
            # Convertir segundos a índices (samples)
            start_idx = int(onset * sfreq)
            end_idx = int(offset * sfreq)
            
            # Recortar las señales
            t_slice = times[start_idx:end_idx]
            joy_slice = joy_data[start_idx:end_idx]
            tonic_slice = eda_tonic[start_idx:end_idx]
            phasic_slice = eda_phasic[start_idx:end_idx]
            
            window_samples = int(SMOOTHING_WINDOW_SECONDS * sfreq)
            # Aplicamos media móvil centrada. min_periods=1 evita que los bordes se vuelvan NaN
            joy_slice = pd.Series(joy_slice).rolling(window=window_samples, center=True, min_periods=1).mean().values
            
            # ---> CÁLCULO DE LA PRIMERA DERIVADA <---
            # Ahora la derivada se calcula sobre la señal ya suavizada, eliminando picos falsos o ruido
            joy_deriv_slice = np.gradient(joy_slice, 1.0 / sfreq)
            
            print(f"  🎬 Procesando Video {vid_id} (De {onset:.1f}s a {offset:.1f}s)...")

            # 5. CORRELACIONES ESPECÍFICAS DEL ESTÍMULO
            # 5.1 Correlaciones del reporte (absoluto)
            r0_ton, rOpt_ton, lagOpt_ton = calculate_time_lagged_correlation(joy_slice, tonic_slice, sfreq, MAX_LAG_SECONDS)
            r0_pha, rOpt_pha, lagOpt_pha = calculate_time_lagged_correlation(joy_slice, phasic_slice, sfreq, MAX_LAG_SECONDS)
            
            # 5.2 Correlaciones de la derivada del reporte (tasa de cambio)
            r0_deriv_ton, rOpt_deriv_ton, lagOpt_deriv_ton = calculate_time_lagged_correlation(joy_deriv_slice, tonic_slice, sfreq, MAX_LAG_SECONDS)
            r0_deriv_pha, rOpt_deriv_pha, lagOpt_deriv_pha = calculate_time_lagged_correlation(joy_deriv_slice, phasic_slice, sfreq, MAX_LAG_SECONDS)

            # 6. GENERAR GRÁFICO DEL ESTÍMULO
            # Aumentamos a 4 subplots y el tamaño a 15x12
            fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
            
            # Actualizamos el texto para mostrar ambos bloques de resultados
            corr_text = (
                f"Video {vid_id} | Análisis del Reporte ({dimension})\n"
                f"--- Señal Absoluta ---\n"
                f" vs Tónica: r(lag=0) = {r0_ton:.3f} | r máx = {rOpt_ton:.3f} (lag {lagOpt_ton:.1f}s)\n"
                f" vs Fásica: r(lag=0) = {r0_pha:.3f} | r máx = {rOpt_pha:.3f} (lag {lagOpt_pha:.1f}s)\n"
                f"--- Primera Derivada (Tasa de cambio) ---\n"
                f" vs Tónica: r(lag=0) = {r0_deriv_ton:.3f} | r máx = {rOpt_deriv_ton:.3f} (lag {lagOpt_deriv_ton:.1f}s)\n"
                f" vs Fásica: r(lag=0) = {r0_deriv_pha:.3f} | r máx = {rOpt_deriv_pha:.3f} (lag {lagOpt_deriv_pha:.1f}s)"
            )
            
            fig.suptitle(f"Sujeto {SUBJECT_ID} | Sesión {acq_id} | Bloque {task_num} | Estímulo: Video {vid_id}", fontsize=14, fontweight='bold')
            # Ajustamos la posición del texto porque ahora es más grande
            fig.text(0.5, 0.88, corr_text, ha='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

            # Configuramos los 4 ejes
            plot_configs = [
                (axes[0], joy_slice, f"{joy_label}\n[Suavizado {SMOOTHING_WINDOW_SECONDS}s]", 'steelblue'), # <-- ACTUALIZADO
                (axes[1], tonic_slice, 'EDA Tónica (EDL)', 'darkblue'),
                (axes[2], phasic_slice, 'EDA Fásica (EDR)', 'darkorange'),
                (axes[3], joy_deriv_slice, f'Derivada del Reporte\n(Velocidad de cambio)', 'purple')
            ]

            for ax, signal, title, color in plot_configs:
                ax.plot(t_slice, signal, color=color, linewidth=1.5)
                ax.set_ylabel(title, fontsize=10)
                ax.grid(True, linestyle=':', alpha=0.6)
                # Fondo sutil para indicar que es la zona del video
                ax.axvspan(onset, offset, color='gold', alpha=0.05)

            axes[-1].set_xlabel("Tiempo absoluto en el bloque (segundos)", fontsize=11)
            
            # Bajamos el inicio de las gráficas (top=0.81) para hacerle espacio al gran cuadro de texto de las correlaciones
            plt.tight_layout(rect=[0, 0, 1, 0.81]) 
            
            bids_fig_name = f"sub-{SUBJECT_ID}_ses-{acq_id}_task-{task_id}_desc-video_{vid_id}_multimodal_fig.png"
            out_filepath = os.path.join(RESULTS_PATH, bids_fig_name)
            
            plt.savefig(out_filepath, dpi=300)
            print(f"     💾 Gráfico guardado en: {out_filepath}")
            
            # Comenta esta línea si no quieres que el script se pause en cada video
            plt.show(block=True) 
            plt.close(fig) # Liberar memoria

# --- EJECUCIÓN ---
if __name__ == "__main__":
    if not os.path.exists(BASE_PATH):
        print(f"❌ El directorio no existe: {BASE_PATH}")
    else:
        files_to_process = sorted([
            os.path.join(BASE_PATH, f) for f in os.listdir(BASE_PATH) 
            if f.endswith("_desc-preproc_eeg.vhdr")
        ])

        if not files_to_process:
            print(f"❌ No se encontraron archivos en: {BASE_PATH}")
        else:
            print(f"✅ Se encontraron {len(files_to_process)} bloques para procesar.")
            for eeg_file in files_to_process:
                process_multimodal_per_stimulus(eeg_file)
        
        print("\n🎉 ¡Proceso completado! Se extrajeron todos los estímulos individuales.")