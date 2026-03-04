import os
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
import random
from scipy.stats import mannwhitneyu

# --- CONFIGURACIÓN ---
SUBJECT_ID = "27"
BASE_PATH = rf"data/derivatives/campeones_preproc/sub-{SUBJECT_ID}/ses-vr/eeg"
SOURCEDATA_PATH = rf"data/sourcedata/xdf/sub-{SUBJECT_ID}"

CHANNEL_TO_PLOT = 'joystick_x'
EPOCH_HALF_WINDOW = 3.0  # Segundos antes y después del pico
SMNA_PEAK_THRESHOLD = 0.01  # Umbral bajo para considerar un pico de SMNA (ajustable)
SMOOTHING_WINDOW_SECONDS = 1.0
ALLOW_OVERLAP = True  # True para permitir solapamiento, False para rechazarlo

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
        eda_filename = f'eda_processed_sub-{SUBJECT_ID}_ses-{acq_id.lower()}_task-{task_id}.csv'
        eda_filepath = os.path.join(os.getcwd(), eda_filename) # Asumiendo que se guardaron en la ruta de ejecución
        
        if not os.path.exists(eda_filepath):
            print(f"⚠️ Saltando bloque {task_id}: No se encontró el CSV de EDA procesado ({eda_filename})")
            continue
            
        print(f"\n🔄 Procesando Bloque {task_id} - Sesión {acq_id}")
        
        # Cargar datos
        raw = mne.io.read_raw_brainvision(eeg_filepath, preload=True, verbose=False)
        events_df = pd.read_csv(events_filepath, sep='\t')
        eda_df = pd.read_csv(eda_filepath)
        
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
        beh_signal = joystick_data[0].copy()
        
        if inversion_instruction == 'inverse':
            beh_signal *= -1
            
        window_samples = int(SMOOTHING_WINDOW_SECONDS * sfreq)
        # Usamos pd.Series para el rolling y .values para volver a numpy
        beh_signal = pd.Series(beh_signal).rolling(window=window_samples, center=True, min_periods=1).mean().values

        # Si es valencia, tomamos el módulo (valor absoluto)
        if dimension == "valence":
            beh_signal = np.abs(beh_signal)
            dimension_label = "valence_module"
        else:
            dimension_label = "arousal"
            
        # Derivada del comportamiento
        beh_derivative = np.gradient(beh_signal) * sfreq
        
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
                        # Encontrar índices en la señal de EEG (joystick)
                        idx_start = int(e_start * sfreq)
                        idx_end = int(e_end * sfreq)
                        
                        beh_chunk = beh_signal[idx_start:idx_end]
                        der_chunk = beh_derivative[idx_start:idx_end]
                        
                        if len(beh_chunk) > 0:
                            all_results.append({
                                "epoch_type": epoch_type,
                                "stimuli": vid_id,
                                "dimension": dimension_label,
                                "mean": np.mean(beh_chunk),
                                "derivate_mean": np.mean(der_chunk)
                            })

    # 4. Guardar resultados
    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        print("❌ No se generaron épocas válidas.")
        return
        
    out_csv = f'epoch_metrics_sub-{SUBJECT_ID}.csv'
    results_df.to_csv(out_csv, index=False)
    print(f"\n✅ Análisis completo. Datos guardados en {out_csv}")
    
    # --- NUEVO: Resumen global de N ---
    print("\n📊 Resumen de Épocas Totales (N):")
    summary_counts = results_df.groupby(['dimension', 'epoch_type']).size()
    print(summary_counts)
    
    # 5. Visualización
    plot_results(results_df)

def plot_results(df):
    """Genera gráficos de Boxplot comparando Peak vs No Peak con Test Mann-Whitney U."""
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Comparación Comportamental: Peak SMNA vs No Peak (Sujeto {SUBJECT_ID})", fontsize=16)
    
    # Aplanar los ejes
    ax_arousal_mean = axes[0, 0]
    ax_arousal_der = axes[1, 0]
    ax_valence_mean = axes[0, 1]
    ax_valence_der = axes[1, 1]
    
    def custom_plot(data, y_col, ax, title_prefix, ylabel):
        if data.empty:
            ax.set_visible(False)
            return

        # 1. Plotear
        sns.boxplot(data=data, x="epoch_type", y=y_col, ax=ax, width=0.4, 
                    boxprops=dict(alpha=0.4), showfliers=False, palette="Set2", order=["peak", "no_peak"])
        sns.stripplot(data=data, x="epoch_type", y=y_col, ax=ax, 
                      size=4, color="black", alpha=0.5, jitter=True, order=["peak", "no_peak"])
        
        # 2. Extraer datos para estadísticas
        peak_vals = data[data['epoch_type'] == 'peak'][y_col].dropna()
        no_peak_vals = data[data['epoch_type'] == 'no_peak'][y_col].dropna()
        
        n_peak = len(peak_vals)
        n_no_peak = len(no_peak_vals)
        
        # 3. Test de Hipótesis (Mann-Whitney U)
        if n_peak > 1 and n_no_peak > 1:
            stat, p_val = mannwhitneyu(peak_vals, no_peak_vals, alternative='two-sided')
            
            # Determinar asteriscos de significancia
            if p_val < 0.001: sig = "***"
            elif p_val < 0.01: sig = "**"
            elif p_val < 0.05: sig = "*"
            else: sig = "ns" # no significativo
            
            final_title = f"{title_prefix} ({sig})"
            
            # Texto con estadísticas dentro del plot
            stats_text = (f"Mann-Whitney U\n"
                          f"p = {p_val:.4f}\n"
                          f"n(peak) = {n_peak}\n"
                          f"n(no_peak) = {n_no_peak}")
            
            # Ubicar texto en la esquina superior izquierda
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', fontsize=9, 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
        else:
            final_title = f"{title_prefix} (Datos insuficientes)"
        
        ax.set_title(final_title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xlabel("") # Limpiamos etiqueta x para que no sea redundante

    # Filtrar datos por dimensión
    df_aro = df[df['dimension'] == 'arousal']
    df_val = df[df['dimension'] == 'valence_module']

    # Generar los 4 subplots
    custom_plot(df_aro, 'mean', ax_arousal_mean, "Arousal - Promedio", "Mean Arousal")
    custom_plot(df_aro, 'derivate_mean', ax_arousal_der, "Arousal - Derivada", "Mean Derivative")

    custom_plot(df_val, 'mean', ax_valence_mean, "Módulo Valencia - Promedio", "Mean |Valence|")
    custom_plot(df_val, 'derivate_mean', ax_valence_der, "Módulo Valencia - Derivada", "Mean Derivative")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Ajuste para el título principal
    plt.savefig(f'boxplot_metrics_stats_sub-{SUBJECT_ID}.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    process_subject()