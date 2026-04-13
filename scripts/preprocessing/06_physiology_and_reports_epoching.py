import os
import pandas as pd
import numpy as np
import mne
from scipy.signal import find_peaks
import json 
import re
import warnings
import neurokit2 as nk

warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN ---
SUBJECT_IDS = ["19", "20", "21", "22", "23", "24", "25", "26", "27", "28",
               "29", "30", "31", "32", "33", "34", "35", "36", "37", "38",
               "39", "40", "42", "43", "46"]

GLOBAL_RESULTS_PATH = r"results/epoch_features" 
os.makedirs(GLOBAL_RESULTS_PATH, exist_ok=True)

VALIDATION_LOG_PATH = r"results/physio_validation_log.json" 
MARKERS_VALIDATION_LOG_PATH = r"results/markers_validation_log.json"

CHANNEL_TO_PLOT = 'joystick_x'
SMOOTHING_WINDOW_SECONDS = 1.0
EPOCH_WINDOW = 3.0

SCR_PEAK_THRESHOLD = 0.05  
SMNA_PEAK_THRESHOLD = 0.5  

def get_column_name(df, possible_names):
    for name in possible_names:
        if name in df.columns: return name
        for col in df.columns:
            if col.lower() == name.lower(): return col
    return None

# ─────────────────────────────────────────────────────────────
# NUEVA FUNCIÓN: calcula stats de normalización por sesión
# ─────────────────────────────────────────────────────────────
def compute_session_stats(subject_id, session_eda_files, eda_base_path):
    """
    Concatena todos los TSV de EDA de una sesión y devuelve
    un dict {col_name: (mean, std)} calculado sobre la sesión completa.
    Retorna también los nombres de columna detectados (del primer archivo válido).
    """
    col_names = {}  # se rellena con el primer archivo
    all_data = {c: [] for c in ['clean', 'phasic', 'tonic', 'smna']}

    for eda_filepath in session_eda_files:
        if not os.path.exists(eda_filepath):
            continue
        df = pd.read_csv(eda_filepath, sep='\t')

        # Detectar nombres de columnas solo una vez
        if not col_names:
            col_names['clean']  = get_column_name(df, ['EDA_Clean', 'Clean'])
            col_names['phasic'] = get_column_name(df, ['EDA_Phasic', 'Phasic'])
            col_names['tonic']  = get_column_name(df, ['EDA_Tonic',  'Tonic'])
            col_names['smna']   = get_column_name(df, ['SMNA', 'smna'])

        for key, col in col_names.items():
            if col and col in df.columns:
                all_data[key].append(df[col].values)

    session_stats = {}
    for key, col in col_names.items():
        if all_data[key]:
            concatenated = np.concatenate(all_data[key])
            mean_val = np.nanmean(concatenated)
            std_val  = np.nanstd(concatenated)
            session_stats[col] = (mean_val, std_val)
        else:
            session_stats[col] = (0.0, 1.0)  # fallback neutro

    return session_stats, col_names


def process_all_subjects():
    valence_results = []
    arousal_results = []
    skipped_blocks = []
    
    validation_data = {}
    if os.path.exists(VALIDATION_LOG_PATH):
        with open(VALIDATION_LOG_PATH, 'r', encoding='utf-8') as f:
            validation_data = json.load(f).get("subjects", {})

    markers_validation_data = {}
    if os.path.exists(MARKERS_VALIDATION_LOG_PATH):
        with open(MARKERS_VALIDATION_LOG_PATH, 'r', encoding='utf-8') as f:
            markers_validation_data = json.load(f).get("subjects", {})

    print(f"Iniciando extracción de épocas para {len(SUBJECT_IDS)} sujetos...\n")
    
    for subject_id in SUBJECT_IDS:
        print(f"👉 Sujeto {subject_id}")
        
        base_path = rf"data/derivatives/campeones_preproc/sub-{subject_id}/ses-vr/eeg"
        sourcedata_path = rf"data/sourcedata/xdf/sub-{subject_id}"
        eda_base_path = rf"data/derivatives/eda_preproc_tests/sub-{subject_id}"
        
        if not os.path.exists(base_path): continue

        files_to_process = sorted([f for f in os.listdir(base_path) if f.endswith("_desc-preproc_eeg.vhdr")])

        # ─────────────────────────────────────────────────────
        # NUEVO: determinar qué bloques son bad ANTES de normalizar
        # ─────────────────────────────────────────────────────
        bad_blocks = set()  # guarda eeg_filenames que hay que saltear
        block_bad_stimuli = {}  # eeg_filename → lista de stim bad (si aplica)

        for eeg_filename in files_to_process:
            skip_this_block = False
            bad_stimuli_for_block = []

            if subject_id in validation_data:
                for val_key, val_info in validation_data[subject_id].items():
                    if val_key in eeg_filename:
                        eda_cat = (val_info.get("eda") or val_info.get("gsr") or {}).get("category", "good").lower()
                        if eda_cat in ["bad", "maybe"]:
                            skip_this_block = True
                            skipped_blocks.append(f"Sub-{subject_id} | {val_key} | EDA={eda_cat}")
                        break

            if subject_id in markers_validation_data and not skip_this_block:
                for val_key, note in markers_validation_data[subject_id].items():
                    if val_key in eeg_filename:
                        note_up = note.strip().upper()
                        is_bad_stim = "ESTIMULO BAD" in note_up or "BAD ESTIMULO" in note_up
                        is_bad_block = (note_up.startswith("BAD") and not is_bad_stim) or "NO SE TOMO" in note_up or "NO TOMADO" in note_up
                        if is_bad_block:
                            skip_this_block = True
                            skipped_blocks.append(f"Sub-{subject_id} | {val_key} | Reporte={note[:20]}")
                        elif is_bad_stim:
                            match = re.search(r'\[(.*?)\]', note)
                            if match: bad_stimuli_for_block.append(match.group(1).strip().lower())
                        break

            if skip_this_block:
                bad_blocks.add(eeg_filename)
            else:
                block_bad_stimuli[eeg_filename] = bad_stimuli_for_block

        # ─────────────────────────────────────────────────────
        # Agrupar por sesión SOLO los bloques buenos
        # ─────────────────────────────────────────────────────
        session_files_map = {}
        for eeg_filename in files_to_process:
            if eeg_filename in bad_blocks:
                continue  # excluir bad de la normalización
            parts   = eeg_filename.split('_')
            task_id = parts[2].split('-')[1]
            acq_id  = parts[3].split('-')[1].upper()
            eda_filepath = os.path.join(
                eda_base_path,
                f'sub-{subject_id}_ses-{acq_id}_task-{task_id}_desc-edapreproc_physio.tsv'
            )
            session_files_map.setdefault(acq_id, []).append(eda_filepath)

        # Calcular stats de normalización una vez por sesión (solo bloques buenos)
        session_stats_cache = {}
        session_col_cache   = {}
        for acq_id, eda_files in session_files_map.items():
            stats, col_names = compute_session_stats(subject_id, eda_files, eda_base_path)
            session_stats_cache[acq_id] = stats
            session_col_cache[acq_id]   = col_names
            print(f"   Sesión {acq_id}: stats calculadas sobre {len(eda_files)} bloque(s) buenos.")

        # ─────────────────────────────────────────────────────
        # Loop de extracción (igual que antes, usando bad_blocks y block_bad_stimuli)
        # ─────────────────────────────────────────────────────
        for eeg_filename in files_to_process:
            if eeg_filename in bad_blocks:
                continue

            bad_stimuli_for_block = block_bad_stimuli.get(eeg_filename, [])

            parts   = eeg_filename.split('_')
            task_id = parts[2].split('-')[1] 
            acq_id  = parts[3].split('-')[1].upper() 
            
            eda_filepath = os.path.join(eda_base_path, f'sub-{subject_id}_ses-{acq_id}_task-{task_id}_desc-edapreproc_physio.tsv')
            if not os.path.exists(eda_filepath): continue
            
            raw    = mne.io.read_raw_brainvision(os.path.join(base_path, eeg_filename), preload=True, verbose=False)
            eda_df = pd.read_csv(eda_filepath, sep='\t')
            
            col_names  = session_col_cache[acq_id]
            col_time   = get_column_name(eda_df, ['Time', 'time'])
            col_clean  = col_names['clean']
            col_phasic = col_names['phasic']
            col_tonic  = col_names['tonic']
            col_smna   = col_names['smna']

            col_phasic_raw = col_phasic + "_raw"
            col_smna_raw   = col_smna   + "_raw"
            col_clean_raw  = col_clean  + "_raw"
            col_tonic_raw  = col_tonic  + "_raw"
            eda_df[col_phasic_raw] = eda_df[col_phasic]
            eda_df[col_smna_raw]   = eda_df[col_smna]
            eda_df[col_clean_raw]  = eda_df[col_clean]
            eda_df[col_tonic_raw]  = eda_df[col_tonic]

            session_stats = session_stats_cache[acq_id]
            std_dict = {}

            for col in [col_clean, col_phasic, col_tonic, col_smna]:
                mean_val, std_val = session_stats.get(col, (0.0, 1.0))
                std_dict[col] = std_val
                if std_val > 0:
                    eda_df[col] = (eda_df[col] - mean_val) / std_val
                else:
                    eda_df[col] = 0.0

            excel_path = os.path.join(sourcedata_path, f"order_matrix_{subject_id}_{acq_id}_block{int(task_id)}_VR.xlsx")
            try:
                excel_df = pd.read_excel(excel_path)
                dim = "valence" if "valence" in excel_df["dimension"].dropna().tolist() else "arousal"
                video_ids = excel_df["video_id"].dropna().tolist()
                inv_instr = excel_df['order_emojis_slider'].dropna().iloc[0] if 'order_emojis_slider' in excel_df else None
            except: continue
                
            eeg_sfreq = raw.info['sfreq']
            eda_sfreq = 1.0 / np.nanmean(np.diff(eda_df[col_time]))
            joystick_data, _ = raw.get_data(picks=[CHANNEL_TO_PLOT], return_times=True)
            raw_beh_signal = joystick_data[0].copy()
            
            events_df = pd.read_csv(os.path.join(base_path, eeg_filename.replace('_eeg.vhdr', '_events.tsv')), sep='\t')
            video_counter = 0
            
            for _, row in events_df.iterrows():
                if row['trial_type'] == 'video':
                    if video_counter >= len(video_ids): break
                    
                    vid_str = str(video_ids[video_counter]).strip().lower()
                    try: vid_str = str(int(float(vid_str)))
                    except: pass
                    
                    if vid_str in bad_stimuli_for_block:
                        video_counter += 1
                        continue
                    
                    vid_id = int(vid_str) if vid_str.isdigit() else vid_str
                    video_counter += 1
                    
                    onset, offset = row['onset'], row['onset'] + row['duration']
                    
                    stim_beh = raw_beh_signal[int(onset*eeg_sfreq):int(offset*eeg_sfreq)].copy()
                    if inv_instr in ['inverse', 'indirect']: stim_beh *= -1
                    
                    stim_beh_raw = stim_beh.copy()
                    
                    if dim == "valence": stim_beh = np.abs(stim_beh)
                    
                    win_smp = int(SMOOTHING_WINDOW_SECONDS * eeg_sfreq)
                    stim_beh     = pd.Series(stim_beh).rolling(window=win_smp, center=True, min_periods=1).mean().values
                    stim_beh_raw = pd.Series(stim_beh_raw).rolling(window=win_smp, center=True, min_periods=1).mean().values
                    stim_der     = np.gradient(stim_beh) * eeg_sfreq

                    curr   = onset
                    idx_ep = 1
                    while curr + EPOCH_WINDOW <= offset:
                        t0, t1 = curr, curr + EPOCH_WINDOW
                        
                        b_idx0, b_idx1 = int((t0-onset)*eeg_sfreq), int((t1-onset)*eeg_sfreq)
                        b_chunk, d_chunk = stim_beh[b_idx0:b_idx1], stim_der[b_idx0:b_idx1]
                        b_chunk_raw = stim_beh_raw[b_idx0:b_idx1]
                        
                        eda_chunk = eda_df[(eda_df[col_time] >= t0) & (eda_df[col_time] < t1)]
                        
                        if len(b_chunk) > 0 and not eda_chunk.empty:
                            p_raw = np.nan_to_num(eda_chunk[col_phasic_raw].values)
                            s_raw = np.nan_to_num(eda_chunk[col_smna_raw].values)
                            t_chunk = eda_chunk[col_time].values
                            dist = int(eda_sfreq * 1.5)
                            
                            try:
                                _, scr_info = nk.eda_peaks(p_raw, sampling_rate=eda_sfreq, amplitude_min=SCR_PEAK_THRESHOLD)
                                scr_count = len(scr_info['SCR_Peaks'])
                                scr_amp_mean = np.nanmean(scr_info['SCR_Amplitude']) if scr_count > 0 else 0.0
                                if std_dict.get(col_phasic, 0) > 0: scr_amp_mean /= std_dict[col_phasic]
                                scr_risetime_mean = np.nanmean(scr_info['SCR_RiseTime']) if scr_count > 0 else 0.0
                            except:
                                scr_count, scr_amp_mean, scr_risetime_mean = 0, 0.0, 0.0

                            smna_p, smna_props = find_peaks(s_raw, height=SMNA_PEAK_THRESHOLD, distance=dist)
                            smna_count = len(smna_p)
                            smna_amp_mean = np.nanmean(smna_props['peak_heights']) if smna_count > 0 else 0.0

                            c_raw   = np.nan_to_num(eda_chunk[col_clean_raw].values)
                            p_raw_z = np.nan_to_num(eda_chunk[col_phasic].values)
                            t_raw   = np.nan_to_num(eda_chunk[col_tonic_raw].values)

                            eda_clean_std  = np.std(c_raw)
                            eda_range      = np.ptp(c_raw)
                            eda_phasic_std = np.std(p_raw_z)
                            eda_tonic_std  = np.std(t_raw)

                            phasic_centered = p_raw - np.mean(p_raw)
                            eda_phasic_zc   = np.sum(np.diff(np.sign(phasic_centered)) != 0)

                            if std_dict.get(col_smna, 0) > 0: smna_amp_mean /= std_dict[col_smna]
                            
                            valid_idx = ~np.isnan(eda_chunk[col_smna].values)
                            if np.sum(valid_idx) > 1:
                                smna_auc = np.trapz(eda_chunk[col_smna].values[valid_idx], x=t_chunk[valid_idx])
                            else:
                                smna_auc = 0.0
                            
                            res = {
                                "Subject": subject_id, "Session": acq_id, "Block": task_id, "Stimulus": vid_id,
                                "Epoch_Index": idx_ep, "Time_Start": round(t0-onset, 2), "Time_End": round(t1-onset, 2),
                                "Report_Mean": np.mean(b_chunk), "Report_Mean_Raw": np.mean(b_chunk_raw),
                                "Report_Variance": np.var(b_chunk), "Report_Deriv_Mean": np.mean(d_chunk),
                                "EDA_Clean_Mean": eda_chunk[col_clean].mean(),
                                "EDA_Phasic_Mean": eda_chunk[col_phasic].mean(),
                                "EDA_Tonic_Mean": eda_chunk[col_tonic].mean(),
                                "EDA_SMNA_AUC": smna_auc,
                                "EDA_SCR_Peaks_Count": scr_count, 
                                "EDA_SCR_Amplitude_Mean": scr_amp_mean,
                                "EDA_SCR_RiseTime_Mean": scr_risetime_mean,
                                "EDA_SMNA_Peaks_Count": smna_count,
                                "EDA_SMNA_Amplitude_Mean": smna_amp_mean,
                                "EDA_Clean_Std": eda_clean_std,
                                "EDA_Range": eda_range,
                                "EDA_Phasic_Std": eda_phasic_std,
                                "EDA_Tonic_Std": eda_tonic_std,
                                "EDA_Phasic_ZeroCrossings": eda_phasic_zc,
                            }

                            if dim == "valence": valence_results.append(res)
                            else: arousal_results.append(res)
                                
                        curr += EPOCH_WINDOW
                        idx_ep += 1

    for name, data in [("valence", valence_results), ("arousal", arousal_results)]:
        if data:
            df   = pd.DataFrame(data)
            path = os.path.join(GLOBAL_RESULTS_PATH, f'{name}_epochs_10s.csv')
            df.to_csv(path, index=False)
            print(f"✅ {name.upper()} guardado: {len(df)} épocas.")

if __name__ == "__main__":
    process_all_subjects()