import os
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
import random
from scipy.stats import mannwhitneyu
from scipy.stats import zscore

# --- CONFIGURACIÓN ---
SUBJECT_ID = "27"
BASE_PATH = rf"data/derivatives/campeones_preproc/sub-{SUBJECT_ID}/ses-vr/eeg"
SOURCEDATA_PATH = rf"data/sourcedata/xdf/sub-{SUBJECT_ID}"
RESULTS_PATH = rf"results/eda_preproc_tests/sub-{SUBJECT_ID}/physio"
os.makedirs(RESULTS_PATH, exist_ok=True)

CHANNEL_TO_PLOT = 'joystick_x'
EPOCH_HALF_WINDOW = 5.0  # Segundos antes y después del pico
SMNA_PEAK_THRESHOLD = 0.5  # Umbral bajo para considerar un pico de SMNA (ajustable)
SMOOTHING_WINDOW_SECONDS = 1.0
ALLOW_OVERLAP = True  # True para permitir solapamiento, False para rechazarlo
RANDOM_SEED = 42

def get_non_overlapping_epochs(centers, duration, min_time, max_time, allow_overlap=False):
    """
    Filtra centros de épocas. Siempre chequea límites del video. 
    Solo chequea solapamiento entre sí si allow_overlap es False.
    """
    valid_epochs = []
    half_dur = duration / 2.0
    
    for c in centers:
        start, end = c - half_dur, c + half_dur
        
        # 1. Chequeo de límites (Siempre obligatorio)
        if start < min_time or end > max_time:
            continue
        
        # 2. Chequeo de solapamiento interno (Condicional)
        overlap = False
        if not allow_overlap:
            for v_start, v_end in valid_epochs:
                if not (end <= v_start or start >= v_end):
                    overlap = True
                    break
                
        if not overlap:
            valid_epochs.append((start, end))
            
    return valid_epochs

def find_no_peak_epochs(valid_peak_epochs, num_epochs, min_time, max_time, duration, allow_overlap=False, max_attempts=500):
    """
    Busca épocas de 'no activación'. 
    SIEMPRE evita solaparse con los picos (valid_peak_epochs).
    Solo evita solaparse con otras 'no_peak' si allow_overlap es False.
    """
    no_peak_epochs = []
    half_dur = duration / 2.0
    attempts = 0
    
    while len(no_peak_epochs) < num_epochs and attempts < max_attempts:
        attempts += 1
        c = random.uniform(min_time + half_dur, max_time - half_dur)
        start, end = c - half_dur, c + half_dur
        
        # 1. Chequear solapamiento con los PICOS (Siempre obligatorio)
        overlap_with_peaks = False
        for p_start, p_end in valid_peak_epochs:
            if not (end <= p_start or start >= p_end):
                overlap_with_peaks = True
                break
        
        if overlap_with_peaks:
            continue # Intentar otro random
            
        # 2. Chequear solapamiento con otras NO-PEAK (Condicional)
        overlap_self = False
        if not allow_overlap:
            for np_start, np_end in no_peak_epochs:
                if not (end <= np_start or start >= np_end):
                    overlap_self = True
                    break
        
        if not overlap_self:
            no_peak_epochs.append((start, end))
            
    return no_peak_epochs

def process_subject():
    all_results = []
    
    # 1. Buscar archivos preprocesados de EEG
    files_to_process = sorted([
        f for f in os.listdir(BASE_PATH) 
        if f.endswith("_desc-preproc_eeg.vhdr")
    ])
    
    if not files_to_process:
        print(f"❌ No se encontraron archivos en {BASE_PATH}")
        return
        
    for eeg_filename in files_to_process:
        eeg_filepath = os.path.join(BASE_PATH, eeg_filename)
        events_filepath = eeg_filepath.replace('_eeg.vhdr', '_events.tsv')
        
        parts = eeg_filename.split('_')
        task_id = parts[2].split('-')[1] # ej: '01'
        acq_id = parts[3].split('-')[1].upper() # ej: 'VR' o similar
        
        # Archivo EDA procesado correspondiente
        eda_filename = f'sub-{SUBJECT_ID}_ses-{acq_id}_task-{task_id}_desc-edapreproc_physio.tsv'
        
        # 💡 Opcional: Si en el primer script los guardaste en "data/derivatives/...", 
        # asegúrate de que la ruta aquí apunte a esa carpeta y no a os.getcwd(). 
        eda_filepath = os.path.join(rf"data/derivatives/eda_preproc_tests/sub-{SUBJECT_ID}", eda_filename)
        
        if not os.path.exists(eda_filepath):
            print(f"⚠️ Saltando bloque {task_id}: No se encontró el TSV BIDS de EDA procesado ({eda_filename})")
            continue
            
        print(f"\n🔄 Procesando Bloque {task_id} - Sesión {acq_id}")
        
        # Cargar datos
        raw = mne.io.read_raw_brainvision(eeg_filepath, preload=True, verbose=False)
        events_df = pd.read_csv(events_filepath, sep='\t')
        eda_df = pd.read_csv(eda_filepath, sep='\t') # Importante: agregar sep='\t'
        
        sfreq = raw.info['sfreq']
        
        # Extraer info del Excel de diseño
        excel_filename = f"order_matrix_{SUBJECT_ID}_{acq_id}_block{int(task_id)}_VR.xlsx"
        excel_filepath = os.path.join(SOURCEDATA_PATH, excel_filename)
        
        try:
            excel_df = pd.read_excel(excel_filepath)
            dimension = "valence" if "valence" in excel_df["dimension"].dropna().tolist() else "arousal"
            video_ids = excel_df["video_id"].dropna().tolist()
            inversion_instruction = excel_df['order_emojis_slider'].dropna().tolist()[0] if not excel_df['order_emojis_slider'].dropna().empty else None
        except Exception as e:
            print(f"  ❌ Error leyendo Excel {excel_filename}: {e}. Saltando bloque.")
            continue
            
        # Preparar señal de comportamiento (Joystick)
        joystick_data, times = raw.get_data(picks=[CHANNEL_TO_PLOT], return_times=True)
        raw_beh_signal = joystick_data[0].copy()
        
        # 2. Iterar sobre eventos de video
        video_counter = 0
        
        # 2. Iterar sobre eventos de video
        video_counter = 0
        for _, row in events_df.iterrows():
            if row['trial_type'] == 'video':
                if video_counter >= len(video_ids):
                    break
                
                vid_id = int(video_ids[video_counter])
                video_counter += 1
                
                onset = row['onset']
                offset = onset + row['duration']

                idx_onset = int(onset * sfreq)
                idx_offset = int(offset * sfreq)
                
                # Extraemos solo el segmento de la señal correspondiente a este video
                stim_beh_signal = raw_beh_signal[idx_onset:idx_offset].copy()
                
                if inversion_instruction == 'inverse':
                    stim_beh_signal *= -1
                    
                window_samples = int(SMOOTHING_WINDOW_SECONDS * sfreq)
                stim_beh_signal = pd.Series(stim_beh_signal).rolling(window=window_samples, center=True, min_periods=1).mean().values

                if dimension == "valence":
                    stim_beh_signal = np.abs(stim_beh_signal)
                    dimension_label = "valence_module"
                else:
                    dimension_label = "arousal"
                    
                stim_beh_derivative = np.gradient(stim_beh_signal) * sfreq
                stim_beh_signal_z = zscore(stim_beh_signal)
                stim_beh_derivative_z = np.gradient(stim_beh_signal_z) * sfreq
                
                # Mascara booleana para el tiempo del video en el DataFrame del EDA
                mask_video = (eda_df['Time'] >= onset) & (eda_df['Time'] <= offset)
                video_eda_df = eda_df[mask_video].copy()
                
                if video_eda_df.empty:
                    continue
                
                # Encontrar picos en SMNA
                smna_signal = video_eda_df['SMNA'].values
                # find_peaks devuelve índices relativos al segmento
                peaks_idx, _ = find_peaks(smna_signal, height=SMNA_PEAK_THRESHOLD, distance=int(sfreq*2)) 
                
                # Convertir índices a tiempos absolutos
                peak_times = video_eda_df['Time'].iloc[peaks_idx].values

                n_raw_peaks = len(peak_times)
                
                # Filtrar épocas de picos (para que no se solapen y entren en el video)
                valid_peak_epochs = get_non_overlapping_epochs(
                    centers=peak_times, 
                    duration=EPOCH_HALF_WINDOW * 2, 
                    min_time=onset, 
                    max_time=offset,
                    allow_overlap=ALLOW_OVERLAP
                )

                n_valid_peaks = len(valid_peak_epochs)
                if n_raw_peaks > 0:
                    print(f"   Video {vid_id}: {n_raw_peaks} picos detectados -> {n_valid_peaks} picos válidos (sin solapamiento).")
                
                # Generar épocas de NO picos
                num_peaks = len(valid_peak_epochs)
                no_peak_epochs = find_no_peak_epochs(
                    valid_peak_epochs, 
                    num_epochs=num_peaks, 
                    min_time=onset, 
                    max_time=offset, 
                    duration=EPOCH_HALF_WINDOW * 2,
                    allow_overlap=ALLOW_OVERLAP
                )
                
                # 3. Extraer métricas para ambas condiciones
                epoch_sets = [("peak", valid_peak_epochs), ("no_peak", no_peak_epochs)]
                
                for epoch_type, epochs in epoch_sets:
                    for e_start, e_end in epochs:
                        # --- MODIFICADO: Índices RELATIVOS al inicio del estímulo ---
                        # Se le resta el 'onset' para que coincida con nuestro array 'stim_beh_signal'
                        idx_start = int((e_start - onset) * sfreq)
                        idx_end = int((e_end - onset) * sfreq)
                        
                        beh_chunk = stim_beh_signal[idx_start:idx_end]
                        der_chunk = stim_beh_derivative[idx_start:idx_end]

                        beh_chunk_z = stim_beh_signal_z[idx_start:idx_end]
                        der_chunk_z = stim_beh_derivative_z[idx_start:idx_end]
                        
                        if len(beh_chunk) > 0:
                            all_results.append({
                                "epoch_type": epoch_type,
                                "stimuli": vid_id,
                                "dimension": dimension_label,
                                "mean": np.mean(beh_chunk),
                                "derivate_mean": np.mean(der_chunk),
                                "std": np.std(beh_chunk),
                                "z_mean": np.mean(beh_chunk_z),
                                "z_derivate_mean": np.mean(der_chunk_z),
                                "z_std": np.std(beh_chunk_z),
                            })

    # 4. Guardar resultados
    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        print("❌ No se generaron épocas válidas.")
        return
        
    out_filename = f'sub-{SUBJECT_ID}_desc-epochmetrics_stat.tsv'
    out_filepath = os.path.join(RESULTS_PATH, out_filename)
    
    results_df.to_csv(out_filepath, sep='\t', index=False)
    print(f"\n✅ Análisis completo. Datos guardados en {out_filepath}")
    
    # --- NUEVO: Resumen global de N ---
    print("\n📊 Resumen de Épocas Totales (N):")
    summary_counts = results_df.groupby(['dimension', 'epoch_type']).size()
    print(summary_counts)
    
    # 5. Visualización
    plot_results(results_df)

def plot_results(df):
    """Genera gráficos de Boxplot comparando Peak vs No Peak con Test Mann-Whitney U.
    Crea un plot para los valores Raw y otro para los Z-Scoreados."""
    sns.set_theme(style="whitegrid")
    
    def generate_figure(data_df, prefix_col, title_type):
        # Ahora usamos 4 filas (Mean, Der_Mean, Std) y 2 columnas (Arousal, Valencia)
        fig, axes = plt.subplots(3, 2, figsize=(14, 20))
        fig.suptitle(f"Comparación Comportamental ({title_type}): Peak SMNA vs No Peak (Sujeto {SUBJECT_ID})", fontsize=16)
        
        def custom_plot(data, y_col, ax, title_prefix, ylabel):
            if data.empty:
                ax.set_visible(False)
                return

            sns.boxplot(data=data, x="epoch_type", y=y_col, ax=ax, width=0.4, 
                        boxprops=dict(alpha=0.4), showfliers=False, palette="Set2", order=["peak", "no_peak"])
            sns.stripplot(data=data, x="epoch_type", y=y_col, ax=ax, 
                          size=4, color="black", alpha=0.5, jitter=True, order=["peak", "no_peak"])
            
            peak_vals = data[data['epoch_type'] == 'peak'][y_col].dropna()
            no_peak_vals = data[data['epoch_type'] == 'no_peak'][y_col].dropna()
            
            n_peak, n_no_peak = len(peak_vals), len(no_peak_vals)
            
            if n_peak > 1 and n_no_peak > 1:
                stat, p_val = mannwhitneyu(peak_vals, no_peak_vals, alternative='two-sided')
                if p_val < 0.001: sig = "***"
                elif p_val < 0.01: sig = "**"
                elif p_val < 0.05: sig = "*"
                else: sig = "ns"
                
                final_title = f"{title_prefix} ({sig})"
                stats_text = f"Mann-Whitney U\np = {p_val:.4f}\nn(peak) = {n_peak}\nn(no_peak) = {n_no_peak}"
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                        verticalalignment='top', fontsize=9, 
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
            else:
                final_title = f"{title_prefix} (Datos insuficientes)"
            
            ax.set_title(final_title, fontweight='bold')
            ax.set_ylabel(ylabel)
            ax.set_xlabel("")

        df_aro = data_df[data_df['dimension'] == 'arousal']
        df_val = data_df[data_df['dimension'] == 'valence_module']

        # Fila 1: Promedios
        custom_plot(df_aro, f'{prefix_col}mean', axes[0, 0], "Arousal - Promedio", "Mean")
        custom_plot(df_val, f'{prefix_col}mean', axes[0, 1], "|Valencia| - Promedio", "Mean")

        # Fila 2: Derivada Promedios
        custom_plot(df_aro, f'{prefix_col}derivate_mean', axes[1, 0], "Arousal - Derivada promedio", "Mean Derivative")
        custom_plot(df_val, f'{prefix_col}derivate_mean', axes[1, 1], "|Valencia| - Derivada promedio", "Mean Derivative")

        # Fila 3: Variabilidad (Desviación Estándar)
        custom_plot(df_aro, f'{prefix_col}std', axes[2, 0], "Arousal - Variabilidad (Std)", "Std Dev")
        custom_plot(df_val, f'{prefix_col}std', axes[2, 1], "|Valencia| - Variabilidad (Std)", "Std Dev")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Guardar figura en la carpeta results con nomenclatura BIDS
        fig_filename = f'sub-{SUBJECT_ID}_desc-boxplot_{title_type}_fig.png'
        fig_filepath = os.path.join(RESULTS_PATH, fig_filename)
        
        plt.savefig(fig_filepath, dpi=300)
        print(f"     💾 Gráfico {title_type} guardado en: {fig_filepath}")
        plt.show()

    # Generar los dos gráficos requeridos
    generate_figure(df, prefix_col="", title_type="RAW")
    generate_figure(df, prefix_col="z_", title_type="Z-SCORED")

if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    process_subject()