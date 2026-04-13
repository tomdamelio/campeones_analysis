import os
import pandas as pd
import numpy as np
import mne
import neurokit2 as nk
import biosppy
import cvxeda  # El paquete standalone instalado por pip
import matplotlib.pyplot as plt
import json
import argparse
from scipy.interpolate import interp1d

# =============================================================================
# --- CONFIGURACIÓN Y ARGUMENTOS CLI ---
# =============================================================================
parser = argparse.ArgumentParser(description="Procesa datos de EDA con opciones avanzadas de cvxEDA.")
parser.add_argument("--subject", type=str, required=True, help="El ID del sujeto a procesar (ej: '38').")
parser.add_argument("--show", action="store_true", help="Mostrar gráficos de la señal EDA.")

# Nuevos argumentos para control del algoritmo
parser.add_argument("--use_clean", action="store_true", help="Si se incluye, pasa la señal Limpia (NeuroKit) a cvxEDA en lugar de la Cruda.")
parser.add_argument("--resample_freq", type=float, default=None, help="Frecuencia (Hz) para downsamplear antes de cvxEDA (ej. 10). Default: Ninguna (usa sfreq original).")
parser.add_argument("--alpha", type=float, default=8e-4, help="Parámetro alpha (curvatura tónica) para cvxEDA. Default: 8e-4")
parser.add_argument("--gamma", type=float, default=1e-2, help="Parámetro gamma (penalización SMNA) para cvxEDA. Default: 1e-2")

args = parser.parse_args()

SUBJECT_ID = args.subject
BASE_PATH = rf"data/derivatives/campeones_preproc/sub-{SUBJECT_ID}/ses-vr/eeg"
OUTPUT_PATH = rf"data/derivatives/eda_preproc_tests/sub-{SUBJECT_ID}"
SOURCEDATA_PATH = rf"data/sourcedata/xdf/sub-{SUBJECT_ID}"
RESULTS_PATH = rf"results/eda_preproc_tests/sub-{SUBJECT_ID}/physio/features_timeseries"

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

EDA_CHANNEL = 'GSR' 
PADDING_SEC = 15.0  # Aumentado a 15s para que el tónico tenga contexto base

print("\n" + "=" * 60)
print(f"🚀 INICIANDO PROCESAMIENTO EDA - SUJETO {SUBJECT_ID}")
print(f"   ➤ Señal origen para cvxEDA: {'Limpia (NeuroKit)' if args.use_clean else 'Cruda'}")
print(f"   ➤ Remuestreo cvxEDA: {f'{args.resample_freq} Hz' if args.resample_freq else 'Desactivado (Original)'}")
print(f"   ➤ Parámetros cvxEDA: Alpha = {args.alpha} | Gamma = {args.gamma}")
print("=" * 60)

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def resample_linear(signal_1d, original_len, target_len):
    """
    Remuestrea una señal 1D usando interpolación lineal.
    Evita los artefactos de oscilación (ringing) del remuestreo de Fourier en señales esparsas (SMNA).
    """
    if original_len == target_len:
        return signal_1d
    
    t_orig = np.linspace(0, 1, original_len)
    t_target = np.linspace(0, 1, target_len)
    
    f_interp = interp1d(t_orig, signal_1d, kind='linear', bounds_error=False, fill_value="extrapolate")
    return f_interp(t_target)

def process_eda_cvx_decomposition(eda_signal, sampling_rate, alpha, gamma):
    """
    Procesa un segmento de EDA usando el paquete standalone cvxeda.
    """
    try:
        delta = 1.0 / sampling_rate
        # Desempaquetamos los 3 primeros valores y descartamos el resto
        edr, smna, edl, _, _, _, _ = cvxeda.cvxEDA(
            np.array(eda_signal), 
            delta, 
            alpha=alpha, 
            gamma=gamma
        )

        return pd.DataFrame({
            'EDR': np.array(edr).flatten(),
            'SMNA': np.array(smna).flatten(),
            'EDL': np.array(edl).flatten()
        })
        
    except Exception as e:
        print(f"   ❌ Fallo en cvxEDA: {e}")
        return pd.DataFrame({
            'EDR': np.zeros(len(eda_signal)),
            'SMNA': np.zeros(len(eda_signal)),
            'EDL': np.zeros(len(eda_signal))
        })

# =============================================================================
# FUNCIÓN PRINCIPAL DE PROCESAMIENTO Y VISUALIZACIÓN
# =============================================================================

def process_and_plot_eda_block(eeg_filepath, eda_channel_name=EDA_CHANNEL):
    
    base_filename = os.path.basename(eeg_filepath)
    events_filepath = eeg_filepath.replace('_eeg.vhdr', '_events.tsv')
    
    parts = base_filename.split('_')
    task_id = [p for p in parts if 'task-' in p][0].split('-')[1]
    acq_id = [p for p in parts if 'acq-' in p][0].split('-')[1].upper()

    print(f"\n🔄 Sesión: {acq_id} | Bloque: {task_id}")

    # 1. CARGAR DATOS
    try:
        raw = mne.io.read_raw_brainvision(eeg_filepath, preload=True, verbose=False)
        events_df = pd.read_csv(events_filepath, sep='\t')
    except Exception as e:
        print(f"  ❌ ERROR al cargar EEG o TSV: {e}")
        return

    if eda_channel_name not in raw.ch_names:
        print(f"  ❌ ERROR: El canal '{eda_channel_name}' no existe.")
        return

    sfreq_orig = raw.info['sfreq']
    eda_data, times = raw.get_data(picks=[eda_channel_name], return_times=True)
    eda_raw = eda_data[0]
    print(f"  ✅ Señal extraída. Fs original: {sfreq_orig} Hz. Duración: {times[-1]:.2f} s")

    # Intentar cargar Excel
    task_num = int(task_id)
    excel_filename = f"order_matrix_{SUBJECT_ID}_{acq_id}_block{task_num}_VR.xlsx" 
    excel_filepath = os.path.join(SOURCEDATA_PATH, excel_filename)
    
    video_ids = []
    if os.path.exists(excel_filepath):
        try:
            excel_df = pd.read_excel(excel_filepath)
            video_ids = excel_df["video_id"].dropna().tolist()
        except KeyError:
            pass

    # 2. PROCESAMIENTO GLOBAL (NeuroKit)
    print("  🧠 Procesando limpieza con NeuroKit2...")
    nk_signals, _ = nk.eda_process(eda_raw, sampling_rate=sfreq_orig)
    eda_clean = nk_signals['EDA_Clean'].values

    # 3. PROCESAMIENTO POR CHUNKS (cvxEDA)
    print(f"  ⚙️ Procesando cvxEDA por estímulo (Padding: {PADDING_SEC}s)...")
    
    eda_smna = np.full_like(eda_raw, np.nan)
    eda_tonic = np.full_like(eda_raw, np.nan)
    eda_phasic = np.full_like(eda_raw, np.nan)
    marks_col = np.full(len(eda_raw), "", dtype=object)
    
    # Decidir qué señal usar como base para extraer los chunks
    base_signal_for_cvx = eda_clean if args.use_clean else eda_raw

    for _, row in events_df.iterrows():
        trial_type = str(row['trial_type'])
        
        if trial_type not in ['bad', 'fixation']:
            onset = row['onset']
            offset = onset + row['duration']
            
            # Límites con padding
            onset_pad = max(0.0, onset - PADDING_SEC)
            offset_pad = min(times[-1], offset + PADDING_SEC)
            
            idx_start_pad = int(onset_pad * sfreq_orig)
            idx_end_pad = int(offset_pad * sfreq_orig)
            idx_start_stim = int(onset * sfreq_orig)
            idx_end_stim = int(offset * sfreq_orig)
            
            # Lógica de marcas
            if 0 <= idx_start_stim < len(marks_col):
                marks_col[idx_start_stim] = "onset"
            if 0 <= idx_end_stim < len(marks_col):
                marks_col[idx_end_stim] = "offset"
            elif idx_end_stim == len(marks_col):
                marks_col[-1] = "offset"

            # Extraer chunk (puede ser crudo o limpio según los args)
            chunk_orig = base_signal_for_cvx[idx_start_pad:idx_end_pad]
            len_chunk_orig = len(chunk_orig)
            
            if len_chunk_orig > 0:
                # --- LÓGICA DE REMUESTREO ---
                sfreq_cvx = args.resample_freq if args.resample_freq else sfreq_orig
                
                if args.resample_freq and args.resample_freq != sfreq_orig:
                    len_chunk_down = int(len_chunk_orig * (sfreq_cvx / sfreq_orig))
                    chunk_to_process = resample_linear(chunk_orig, len_chunk_orig, len_chunk_down)
                else:
                    chunk_to_process = chunk_orig
                
                # Procesar con cvxEDA
                print(f"     -> Evento '{trial_type}' | Duración total chunk: {len_chunk_orig/sfreq_orig:.1f}s")
                cvx_df = process_eda_cvx_decomposition(chunk_to_process, sfreq_cvx, args.alpha, args.gamma)
                
                # --- LÓGICA DE UPSAMPLING (Volver a la frecuencia original) ---
                if args.resample_freq and args.resample_freq != sfreq_orig:
                    tonic_chunk = resample_linear(cvx_df['EDL'].values, len(cvx_df), len_chunk_orig)
                    phasic_chunk = resample_linear(cvx_df['EDR'].values, len(cvx_df), len_chunk_orig)
                    smna_chunk = resample_linear(cvx_df['SMNA'].values, len(cvx_df), len_chunk_orig)
                else:
                    tonic_chunk = cvx_df['EDL'].values
                    phasic_chunk = cvx_df['EDR'].values
                    smna_chunk = cvx_df['SMNA'].values

                # --- QUITAR PADDING E INSERTAR EN SEÑAL GLOBAL ---
                inner_start = idx_start_stim - idx_start_pad
                inner_end = idx_end_stim - idx_start_pad
                len_to_insert = min(idx_end_stim - idx_start_stim, inner_end - inner_start)
                
                if len_to_insert > 0:
                    eda_smna[idx_start_stim:idx_start_stim + len_to_insert] = smna_chunk[inner_start:inner_start + len_to_insert]
                    eda_tonic[idx_start_stim:idx_start_stim + len_to_insert] = tonic_chunk[inner_start:inner_start + len_to_insert]
                    eda_phasic[idx_start_stim:idx_start_stim + len_to_insert] = phasic_chunk[inner_start:inner_start + len_to_insert]

    # --- GUARDAR RESULTADOS ---
    df_to_save = pd.DataFrame({
        'Time': times,
        'Raw': eda_raw,
        'Clean': eda_clean,
        'Tonic': eda_tonic,
        'Phasic': eda_phasic,
        'SMNA': eda_smna,
        'Marks': marks_col
    })
    
    bids_filename = f'sub-{SUBJECT_ID}_ses-{acq_id}_task-{task_id}_desc-edapreproc_physio'
    
    output_tsv = os.path.join(OUTPUT_PATH, f'{bids_filename}.tsv')
    df_to_save.to_csv(output_tsv, sep='\t', index=False)
    
    bids_metadata = {
        "SamplingFrequency": sfreq_orig,
        "StartTime": 0.0,
        "Columns": list(df_to_save.columns),
        "Description": "Processed EDA signals",
        "RawEDAChannel": eda_channel_name,
        "cvxEDA_Parameters": {
            "alpha": args.alpha,
            "gamma": args.gamma,
            "used_clean_signal": args.use_clean,
            "resampled_freq_hz": args.resample_freq if args.resample_freq else sfreq_orig,
            "padding_sec": PADDING_SEC
        }
    }
    
    output_json = os.path.join(OUTPUT_PATH, f'{bids_filename}.json')
    with open(output_json, 'w') as f:
        json.dump(bids_metadata, f, indent=4)
        
    print(f"  💾 Señales procesadas guardadas en BIDS (+ JSON con parámetros).")

    # --- VISUALIZACIÓN ---
    print("  📊 Generando visualización...")
    fig, axes = plt.subplots(5, 1, figsize=(18, 12), sharex=True)
    
    title_params = f"cvxEDA: Alpha={args.alpha}, Gamma={args.gamma} | Input={'Limpio' if args.use_clean else 'Crudo'} | Fs={args.resample_freq if args.resample_freq else sfreq_orig}Hz"
    fig.suptitle(f"EDA | Sujeto {SUBJECT_ID} - Sesión {acq_id} - Bloque {task_id}\n[{title_params}]", fontsize=14)

    signals_to_plot = [
        (eda_raw, 'Raw', 'gray'),
        (eda_clean, 'Clean (NeuroKit)', 'black'),
        (eda_tonic, 'Tonic (cvx)', 'blue'),
        (eda_phasic, 'Phasic (cvx)', 'orange'),
        (eda_smna, 'SMNA', 'purple')
    ]

    for ax, (signal_array, title, color) in zip(axes, signals_to_plot):
        ax.plot(times, signal_array, color=color, linewidth=1)
        ax.set_ylabel(title, fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        video_counter = 0
        for _, row in events_df.iterrows():
            onset = row['onset']
            offset = onset + row['duration']
            trial_type = str(row['trial_type'])
            
            if trial_type not in ['bad', 'fixation']:
                if trial_type == 'video':
                    ax.axvspan(onset, offset, color='orange', alpha=0.1)
                elif trial_type == 'calm':
                    ax.axvspan(onset, offset, color='blue', alpha=0.1)
                elif trial_type == 'video_luminance':
                    ax.axvspan(onset, offset, color='green', alpha=0.1)
            
            if ax == axes[0]:
                label_to_plot = trial_type if trial_type not in ['bad', 'fixation'] else ''
                if trial_type == 'video':
                    if video_counter < len(video_ids):
                        label_to_plot = f"video {int(video_ids[video_counter])}"
                    video_counter += 1
                
                ax.text(onset, np.max(signal_array), f" {label_to_plot}", rotation=45, ha='left', va='bottom', fontsize=9)

    axes[-1].set_xlabel("Tiempo (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    fig_bids_name = f"sub-{SUBJECT_ID}_ses-{acq_id}_task-{task_id}_desc-edatimeseries_fig.png"
    fig_path = os.path.join(RESULTS_PATH, fig_bids_name)
    plt.savefig(fig_path, dpi=300)
    print(f"  📊 Gráfico guardado en: {fig_path}")
    
    if args.show:
        print("  👁️  Mostrando gráfico. Cierra la ventana para continuar...")
        plt.show(block=True)
    else:
        plt.close(fig)

# =============================================================================
# EJECUCIÓN
# =============================================================================
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
            for eeg_file in files_to_process:
                process_and_plot_eda_block(eeg_file)
        
        print("\n" + "=" * 60)
        print("🎉 ¡Proceso completado!")