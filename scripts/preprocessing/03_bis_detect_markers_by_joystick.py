#!/usr/bin/env python
"""
Script alternativo para la Fase C del proceso de análisis:
- Detecta automáticamente marcadores basándose en la AMPLITUD (puntaje bruto) del canal JOYSTICK.
- Útil para sujetos donde los marcadores AUDIO/PHOTO fallaron o son irrecuperables.
- Visualiza las señales con las anotaciones para EDICIÓN MANUAL.
- Fusiona las anotaciones finales con los eventos originales y actualiza onsets y duraciones,
  respetando la nomenclatura original (desc-merged_events).

Uso desde la terminal:
    python scripts/preprocessing/03_detect_joystick_markers.py --subject 14 --task 01 --acq b
    
Parámetros importantes:
    --joystick-channel: Nombre del canal (default: 'joystick_x')
    --joystick-threshold: Factor de umbral sobre el puntaje bruto (default: 3.0)
    --joystick-min-distance: Distancia mínima entre ensayos en segundos (default: 15.0)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
import mne
mne.viz.set_browser_backend('qt')
from mne_bids import BIDSPath, read_raw_bids
import matplotlib.pyplot as plt
from scipy import stats, signal

# Buscar la raíz del repositorio
script_path = Path(__file__).resolve()
repo_root = script_path
while not (repo_root / ".git").exists() and repo_root != repo_root.parent:
    repo_root = repo_root.parent

sys.path.insert(0, str(repo_root))

def parse_args():
    parser = argparse.ArgumentParser(description="Detecta marcadores basándose en el puntaje bruto del Joystick")
    parser.add_argument("--subject", type=str, required=True, help="ID del sujeto (e.g., '14')")
    parser.add_argument("--session", type=str, default="vr", help="ID de la sesión (default: 'vr')")
    parser.add_argument("--task", type=str, default=None, help="ID de la tarea (e.g., '01')")
    parser.add_argument("--acq", type=str, default=None, help="Parámetro de adquisición (e.g., 'b')")
    parser.add_argument("--run", type=str, default=None, help="ID del run")
    
    # Parámetros específicos de Joystick
    parser.add_argument("--joystick-channel", type=str, default="joystick_x", help="Canal del joystick")
    parser.add_argument("--joystick-threshold", type=float, default=0.1, help="Umbral (Z-score de la amplitud) para inicio de movimiento")
    parser.add_argument("--joystick-min-distance", type=float, default=15.0, help="Distancia mínima entre ensayos (segundos)")
    parser.add_argument("--default-duration", type=float, default=10.0, help="Duración por defecto de la anotación generada")
    
    # Parámetros heredados útiles
    parser.add_argument("--no-zscore", action="store_true", help="No aplicar z-score a las señales visualizadas")
    parser.add_argument("--events-dir", type=str, default="events", help="Directorio de eventos originales")
    parser.add_argument("--events-desc", type=str, default=None)
    parser.add_argument("--merged-save-dir", type=str, default="merged_events")
    parser.add_argument("--merged-desc", type=str, default="merged", help="Descripción para guardar (igual al original)")
    parser.add_argument("--force-merge", action="store_true", help="Forzar fusión aunque no coincidan las cantidades")
    
    return parser.parse_args()

def load_raw_data(subject, session, task, acq=None, run=None):
    bids_root = repo_root / 'data' / 'raw'
    if task and task.isdigit(): task = task.zfill(2)
    if acq: acq = acq.lower()
    
    import glob
    run_str = f"run-{run}" if run else "run-*"
    pattern = f"sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{task}_acq-{acq}_{run_str}_eeg.vhdr"
    search_path = bids_root / pattern
    matching_files = glob.glob(str(search_path))
    
    if not matching_files:
        raise FileNotFoundError(f"No se encontró archivo EEG para: {search_path}")
    
    eeg_file = Path(matching_files[0])
    import re
    run_match = re.search(r'_run-(\d+)_', eeg_file.name)
    run_detected = run_match.group(1)
    
    bids_path = BIDSPath(subject=subject, session=session, task=task, run=run_detected, acquisition=acq, datatype='eeg', root=bids_root, extension='.vhdr')
    raw = read_raw_bids(bids_path, verbose=False)
    return raw, bids_path, run_detected

def load_events_file(subject, session, task, run, acq=None, events_dir="events", desc=None):
    events_root = repo_root / 'data' / 'derivatives' / events_dir
    if task and task.isdigit(): task = task.zfill(2)
    if run and run.isdigit(): run = run.zfill(3)
    if acq: acq = acq.lower()
    
    events_path = BIDSPath(subject=subject, session=session, task=task, run=run, acquisition=acq, datatype='eeg', suffix='events', description=desc, extension='.tsv', root=events_root, check=False)
    
    if not events_path.fpath.exists():
        alt_path = BIDSPath(subject=subject, session=session, task=task, run=run, acquisition=acq, datatype='eeg', suffix='events', extension='.tsv', root=events_root, check=False)
        if alt_path.fpath.exists():
            events_path = alt_path
        else:
            return None
    
    return pd.read_csv(events_path.fpath, sep='\t')

def apply_zscore_to_raw(raw):
    raw_zscore = raw.copy()
    data = raw_zscore.get_data()
    for i in range(data.shape[0]):
        if np.var(data[i]) == 0 or np.isnan(data[i]).any(): continue
        try:
            data[i] = stats.zscore(data[i], nan_policy='omit')
        except: continue
    raw_zscore = mne.io.RawArray(data, raw.info)
    raw_zscore.set_annotations(raw.annotations)
    return raw_zscore

def detect_joystick_onsets(raw, channel='joystick_x', threshold_factor=3.0, min_distance_sec=15.0):
    """
    Detecta el inicio de los movimientos analizando el puntaje bruto (amplitud) del joystick.
    """
    print(f"\n=== Detectando movimientos por puntaje bruto en el canal {channel} ===\n")
    
    if channel not in raw.ch_names:
        similar = [ch for ch in raw.ch_names if 'joy' in ch.lower()]
        if similar:
            channel = similar[0]
            print(f"Canal original no encontrado, usando alternativo: {channel}")
        else:
            print(f"Error: No hay canales de joystick disponibles.")
            return np.array([])
            
    # Obtener datos de la señal original
    data = raw.get_data(picks=channel)[0]
    sfreq = raw.info['sfreq']
    
    # --- LÍNEAS MODIFICADAS ---
    # 1. Calcular la diferencia absoluta punto a punto (derivada simple)
    # Usamos prepend para mantener el mismo largo que la señal original
    diff_data = np.abs(np.diff(data, prepend=data[0]))
    
    # 2. Definir el umbral fijo solicitado
    threshold = 0.001
    
    print(f"Buscando inicios con diferencia punto a punto > {threshold}")
    
    # 3. Encontrar los picos donde la diferencia supera el umbral
    min_distance_samples = int(min_distance_sec * sfreq)
    
    peaks, _ = signal.find_peaks(
        diff_data, 
        height=threshold, 
        distance=min_distance_samples
    )
    
    peak_times = peaks / sfreq
    print(f"Se detectaron {len(peak_times)} movimientos basados en el puntaje bruto.")
    
    return peak_times

def visualize_signals_with_annotations(raw, annotations, apply_zscore=True, target_channel="joystick_x"):
    print("\n=== Visualizando señales para EDICIÓN MANUAL ===\n")
    
    channels_to_pick = [target_channel] if target_channel in raw.ch_names else []
    if not channels_to_pick:
        similar = [ch for ch in raw.ch_names if 'joy' in ch.lower()]
        if similar: channels_to_pick.append(similar[0])
            
    for ch in ['AUDIO', 'PHOTO']:
        if ch in raw.ch_names and ch not in channels_to_pick:
            channels_to_pick.append(ch)
            
    if not channels_to_pick: channels_to_pick = raw.ch_names[:3]

    raw_plot = raw.copy().pick_channels(channels_to_pick)
    raw_plot.set_annotations(annotations)
    
    if apply_zscore:
        raw_plot = apply_zscore_to_raw(raw_plot)
    
    if raw_plot.info['sfreq'] > 1000:
        raw_plot.resample(1000.0)
    
    print("INSTRUCCIONES:")
    print("1. Ajusta los onsets arrastrando la anotación.")
    print("2. AJUSTA LA DURACIÓN arrastrando el borde derecho (esto se guardará en tu TSV).")
    print("3. Presiona 'a' para crear nuevas anotaciones si faltan.")
    print("4. Clic derecho para borrar las anotaciones sobrantes.")
    print("-> CIERRA LA VENTANA CUANDO HAYAS TERMINADO PARA GUARDAR.\n")
    
    # 'auto' asegura que el joystick se vea bien proporcionado
    fig = raw_plot.plot(
        title="Edición Manual basada en Puntaje Bruto del Joystick",
        scalings='auto', 
        duration=30,
        start=0,
        show=True,
        block=True
    )
    
    updated_annotations = raw_plot.annotations
    return updated_annotations

def merge_events_with_annotations(original_events_df, annotations, force_merge=False):
    print("\n=== Sincronizando con archivo original de eventos ===\n")
    
    merged_df = original_events_df.copy().reset_index(drop=True)
    
    # Identificar qué eventos debemos comparar y actualizar (ignorando calm y fixation)
    if 'trial_type' in merged_df.columns:
        mask_valid = ~merged_df['trial_type'].isin(['calm', 'fixation'])
    else:
        mask_valid = pd.Series(True, index=merged_df.index)
        
    valid_count = mask_valid.sum()
    
    if valid_count != len(annotations):
        print(f"❌ ¡ADVERTENCIA! Cantidades distintas: Eventos a marcar en TSV ({valid_count}) vs Marcas en pantalla ({len(annotations)}).")
        print(f"   (Nota: Se están ignorando {len(merged_df) - valid_count} eventos 'calm'/'fixation')")
        
        if not force_merge:
            print("Deben coincidir para poder fusionar correctamente. Volveremos a abrir la ventana.")
            return None, True
        else:
            print("⚠️ Forzando fusión a pesar de la diferencia.")
            limit = min(valid_count, len(annotations))
            valid_indices = merged_df[mask_valid].index[:limit]
            merged_df.loc[valid_indices, 'onset'] = annotations.onset[:limit]
            merged_df.loc[valid_indices, 'duration'] = annotations.duration[:limit]
            return merged_df, False
            
    # ACTUALIZACIÓN DE DURACIONES Y ONSETS SOLO PARA LOS EVENTOS VÁLIDOS
    merged_df.loc[mask_valid, 'onset'] = annotations.onset
    merged_df.loc[mask_valid, 'duration'] = annotations.duration  
    
    print(f"✅ ¡Fusión exitosa! Se actualizaron {valid_count} eventos (conservando calm/fixation originales).")
    return merged_df, False

def save_merged_events(merged_df, bids_path, merged_save_dir="merged_events", merged_desc="merged"):
    merged_root = repo_root / 'data' / 'derivatives' / merged_save_dir
    os.makedirs(merged_root, exist_ok=True)
    
    output_path = BIDSPath(
        subject=bids_path.subject, session=bids_path.session, task=bids_path.task,
        run=bids_path.run, acquisition=bids_path.acquisition, datatype='eeg',
        suffix='events', description=merged_desc, extension='.tsv',
        root=merged_root, check=False
    )
    
    merged_df.to_csv(output_path.fpath, sep='\t', index=False)
    print(f"\n📁 Archivo final guardado idéntico al proceso estándar en:\n   {output_path.fpath}")
    return str(output_path.fpath)

def display_events_in_order(events_df):
    """Muestra los eventos en su orden original en la terminal."""
    if events_df is None or len(events_df) == 0:
        return
    print("\n=== EVENTOS EN ORDEN ORIGINAL (Desde el TSV/Excel) ===")
    for idx, row in events_df.iterrows():
        onset = row.get('onset', 'N/A')
        dur = row.get('duration', 'N/A')
        tipo = row.get('trial_type', 'N/A')
        print(f"Evento #{idx + 1}: Onset: {onset:.2f} s | Duración: {dur:.2f} s | Tipo: {tipo}")
    print("=====================================================\n")

def process_single_run(args, task, acq):
    try:
        raw, bids_path, run = load_raw_data(args.subject, args.session, task, acq, args.run)
        
        # 1. Cargar eventos originales (TSV/Excel)
        original_events_df = load_events_file(args.subject, args.session, task, run, acq, args.events_dir, args.events_desc)
        if original_events_df is None:
            print("❌ No se encontró archivo TSV original para este bloque. Cancelando.")
            return 1
        
        display_events_in_order(original_events_df)

        # 2. Detectar onsets automáticos basados en el puntaje bruto del Joystick
        joystick_onsets = detect_joystick_onsets(
            raw, 
            channel=args.joystick_channel, 
            threshold_factor=args.joystick_threshold, 
            min_distance_sec=args.joystick_min_distance
        )
        
        # Intentar extraer duraciones promedio del original si existen
        default_dur = args.default_duration

        # 3. Crear Anotaciones Iniciales
        annotations = mne.Annotations(
            onset=joystick_onsets,
            duration=np.ones_like(joystick_onsets) * default_dur,
            description=["auto_JOY_RAW"] * len(joystick_onsets)
        )
        
        # 4. BUCLE DE EDICIÓN MANUAL
        needs_edit = True
        while needs_edit:
            valid_expected = sum(~original_events_df['trial_type'].isin(['calm', 'fixation'])) if 'trial_type' in original_events_df.columns else len(original_events_df)
            print(f"\nEsperados a marcar (sin calm/fixation): {valid_expected} | Detectados en pantalla: {len(annotations)}")
            
            # Abrir visualizador interactivo
            annotations = visualize_signals_with_annotations(
                raw, annotations, apply_zscore=not args.no_zscore, target_channel=args.joystick_channel
            )
            
            # Intentar fusionar
            merged_df, needs_edit = merge_events_with_annotations(original_events_df, annotations, args.force_merge)
            
            if needs_edit:
                respuesta = input("Las cantidades no coinciden. ¿Reabrir para corregir? (s/n): ").lower()
                if respuesta == 'n':
                    print("Operación cancelada por el usuario.")
                    return 1

        # 5. Guardar resultado final con nomenclatura estándar ('merged')
        save_merged_events(merged_df, bids_path, args.merged_save_dir, args.merged_desc)
        print("\n🎉 Proceso completado exitosamente para este bloque.")
        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    args = parse_args()
    tasks_to_process = [args.task] if args.task else ['01', '02', '03', '04']
    acqs_to_process = [args.acq] if args.acq else ['a', 'b']
    
    for task in tasks_to_process:
        for acq in acqs_to_process:
            print(f"\n{'-'*60}\nIniciando sub-{args.subject} task-{task} acq-{acq}\n{'-'*60}")
            process_single_run(args, task, acq)

if __name__ == "__main__":
    sys.exit(main())