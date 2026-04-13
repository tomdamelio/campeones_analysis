#!/usr/bin/env python
"""
Este script calcula el tiempo de reacción (tiempo hasta el primer reporte) 
para los videos utilizando el canal 'joystick_x' de los registros fisiológicos.

1. Localiza el archivo _desc-merged_events.tsv (con los onsets reales).
2. Localiza y carga el archivo raw de BrainVision (.vhdr).
3. Itera sobre los eventos de tipo video en el TSV.
4. Para cada video, busca el primer cambio significativo en el canal 'joystick_x'.
5. Exporta los resultados a un archivo Excel (.xlsx).
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import mne
import matplotlib.pyplot as plt

# Buscar la raíz del repositorio
script_path = Path(__file__).resolve()
repo_root = script_path
while not (repo_root / ".git").exists() and repo_root != repo_root.parent:
    repo_root = repo_root.parent

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calcula el tiempo hasta el primer movimiento del joystick tras el inicio del video."
    )
    parser.add_argument("--subjects", type=str, nargs='+', required=True,
                       help="Lista de IDs de sujetos a procesar (ej: 34)")
    parser.add_argument("--session", type=str, default="vr",
                       help="ID de la sesión (default: 'vr')")
    parser.add_argument("--task", type=str, required=True,
                       help="ID de la tarea específica (ej: '01', '02')")
    parser.add_argument("--acq", type=str, required=True,
                       help="Parámetro de adquisición (ej: 'a', 'b')")
    parser.add_argument("--joystick-channel", type=str, default="joystick_x",
                       help="Nombre del canal del joystick (default: 'joystick_x')")
    parser.add_argument("--threshold", type=float, default=0.001,
                       help="Umbral de cambio en la derivada para considerar movimiento (default: 0.02)")
    parser.add_argument("--plot", action="store_true", 
                       help="Muestra gráficos de la señal del joystick para cada video")
    
    return parser.parse_args()

def find_files(subject, session, task, acq):
    """
    Encuentra el TSV de merged_events y el VHDR crudo.
    """
    task_fmt = str(int(task)).zfill(2)
    acq_fmt = acq.lower()
    
    # Rutas base
    deriv_dir = repo_root / 'data' / 'derivatives' / 'merged_events' / f"sub-{subject}" / f"ses-{session}" / "eeg"
    raw_dir = repo_root / 'data' / 'raw' / f"sub-{subject}" / f"ses-{session}" / "eeg"
    
    # Patrones de búsqueda
    pattern_base = f"sub-{subject}_ses-{session}_task-{task_fmt}_acq-{acq_fmt}_run-*"
    
    tsv_pattern = pattern_base + "_desc-merged_events.tsv"
    vhdr_pattern = pattern_base + "_eeg.vhdr"
    
    tsv_files = list(deriv_dir.glob(tsv_pattern))
    vhdr_files = list(raw_dir.glob(vhdr_pattern))
    
    tsv_file = tsv_files[0] if tsv_files else None
    vhdr_file = vhdr_files[0] if vhdr_files else None
    
    return tsv_file, vhdr_file

def calculate_reaction_times(tsv_file, vhdr_file, joystick_ch, threshold, plot=False):
    """
    Cruza los tiempos del TSV con el EEG para detectar el primer movimiento.
    """
    print(f"Cargando eventos reales desde:\n  -> {tsv_file.name}")
    events_df = pd.read_csv(tsv_file, sep='\t')
    
    print(f"Cargando registro EEG desde:\n  -> {vhdr_file.name}")
    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True, verbose='ERROR')
    
    if joystick_ch not in raw.ch_names:
        raise ValueError(f"El canal '{joystick_ch}' no se encuentra. Canales disponibles: {raw.ch_names}")
    
    joy_idx = raw.ch_names.index(joystick_ch)
    joy_data = raw.get_data(picks=[joy_idx])[0]
    sfreq = raw.info['sfreq']
    
    print(f"\nINFO JOYSTICK: Rango global de la señal: [{joy_data.min():.3f} a {joy_data.max():.3f}]")
    print("-" * 50)
    
    results = []
    
    # Filtrar solo los eventos que son videos (incluye 'video' y 'video_luminance')
    video_events = events_df[events_df['trial_type'].str.contains('video', na=False)]
    
    if video_events.empty:
        print("No se encontraron eventos de tipo 'video' en el TSV.")
        return pd.DataFrame()
    
    for _, row in video_events.iterrows():
        video_id = row['stim_id']
        onset = row['onset']
        duration = row['duration']
        cond = row.get('condition', 'unknown')
        
        # Convertir segundos a muestras
        start_sample = int(onset * sfreq)
        end_sample = int((onset + duration) * sfreq)
        
        # Evitar desbordamiento de índice si el video dura más que el registro
        end_sample = min(end_sample, len(joy_data))
        
        # Aislar el segmento del video
        segment = joy_data[start_sample:end_sample]
        
        if len(segment) == 0:
            print(f"Video {video_id}: Datos vacíos (Onset {onset}s fuera de rango)")
            continue
            
        # Calcular los cambios entre muestras (derivada absoluta)
        diffs = np.abs(np.diff(segment))
        
        # Buscar el primer momento que supera el threshold
        movements = np.where(diffs > threshold)[0]
        
        if len(movements) > 0:
            # Dividir por sfreq da el tiempo en segundos DESDE EL ONSET
            time_until_report = movements[0] / sfreq
            msg = f"RT = {time_until_report:.3f} s"
        else:
            time_until_report = np.nan
            msg = "Sin movimiento detectado"
            
        print(f"Video {video_id:3} ({cond[:4]:4}) | Onset: {onset:7.2f}s | {msg}")
        
        results.append({
            'video_id': video_id,
            'condition': cond,
            'onset_time': onset,
            'time_until_report': time_until_report
        })

        # Graficar si se habilitó el flag --plot
        if plot:
            plt.figure(figsize=(10, 4))
            # Eje X en segundos desde el inicio del video
            time_axis = np.arange(len(segment)) / sfreq
            plt.plot(time_axis, segment, label=f'Canal: {joystick_ch}', color='steelblue')
            
            if not np.isnan(time_until_report):
                plt.axvline(x=time_until_report, color='red', linestyle='--', 
                            label=f'Primer mov: {time_until_report:.2f}s', linewidth=2)
                
            plt.title(f"Video ID: {video_id} | Condición: {cond} | Onset real: {onset:.2f}s")
            plt.xlabel("Tiempo transcurrido desde inicio del video (segundos)")
            plt.ylabel("Amplitud del Joystick (-1 a 1)")
            plt.ylim(-1.1, 1.1)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            print("  >> Mostrando gráfico... (Cierra la ventana para continuar con el siguiente video)")
            plt.show()
            
    return pd.DataFrame(results)

def main():
    args = parse_args()
    
    # Directorio de salida
    output_dir = repo_root / 'data' / 'derivatives' / 'reaction_times'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for subject in args.subjects:
        print(f"\n{'='*60}")
        print(f" Procesando Sub-{subject} | Ses-{args.session} | Task-{args.task} | Acq-{args.acq}")
        print(f"{'='*60}")
        
        try:
            tsv_file, vhdr_file = find_files(subject, args.session, args.task, args.acq)
            
            if not tsv_file:
                print(f"ERROR: No se encontró el archivo *_desc-merged_events.tsv")
                continue
            if not vhdr_file:
                print(f"ERROR: No se encontró el archivo *_eeg.vhdr")
                continue
                
            # Procesar y calcular RTs
            results_df = calculate_reaction_times(
                tsv_file=tsv_file,
                vhdr_file=vhdr_file,
                joystick_ch=args.joystick_channel,
                threshold=args.threshold,
                plot=args.plot
            )
            
            if results_df.empty:
                print("\nEl dataframe de resultados está vacío. No se guardará el Excel.")
                continue
                
            # Extraer el run del nombre del TSV (ej: run-006)
            run_str = tsv_file.stem.split('_run-')[1].split('_')[0]
            
            # Nombre del archivo final
            task_fmt = str(int(args.task)).zfill(2)
            out_filename = f"sub-{subject}_ses-{args.session}_task-{task_fmt}_acq-{args.acq.lower()}_run-{run_str}_reaction_times.xlsx"
            out_path = output_dir / out_filename
            
            # Exportar a Excel
            results_df.to_excel(out_path, index=False)
            print(f"\n{'='*60}")
            print(f" ÉXITO: Resultados guardados correctamente en:\n {out_path}")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"\nERROR INESPERADO al procesar sujeto {subject}: {e}")

if __name__ == "__main__":
    main()