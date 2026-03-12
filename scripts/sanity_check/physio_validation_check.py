# %%
import mne
from mne_bids import BIDSPath, read_raw_bids
import neurokit2 as nk
import pandas as pd
import matplotlib
try:
    matplotlib.use('Qt5Agg')
except ImportError:
    print("Qt5Agg backend not found, trying TkAgg...")
    matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
# Inferimos la raíz del proyecto asumiendo que el script está en una subcarpeta (ej: src/scripts/)
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent # Sube dos niveles, ajusta si es necesario
bids_root = project_root / "data" / "raw"
results_dir = project_root / "results"
results_dir.mkdir(parents=True, exist_ok=True) # Crea la carpeta results si no existe

log_file_path = results_dir / "validation_log.json"

# --- LISTAS PARA ITERAR ---
# Modifica estas listas según los datos que necesites validar
subjects = ["30"]
sessions = ["vr"]
tasks = ["01", "02", "03", "04"]
runs = ["002", "003", "004", "005", "006", "007", "008", "009"]
acqs = ["a", "b"]

# --- FUNCIONES DE AYUDA ---
def load_validation_log(path):
    """Carga el log de validación desde un archivo JSON. Si no existe, crea uno nuevo."""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            print("📋 Log de validación cargado.")
            return json.load(f)
    else:
        print("📋 Creando nuevo log de validación.")
        return {"subjects": {}}

def save_validation_log(path, data):
    """Guarda el log de validación en un archivo JSON."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_manual_validation_input(validation_data, subject, file_key, signal_type):
    """Pide al usuario que evalúe la calidad de una señal."""
    print(f"\n--- VALIDACIÓN MANUAL: {signal_type.upper()} ---")
    
    previous_entry = validation_data.get("subjects", {}).get(subject, {}).get(file_key, {}).get(signal_type, {})
    default_category = previous_entry.get("category", "")
    default_notes = previous_entry.get("notes", "")

    if default_category:
        print(f"🗂️  Valores previos: Categoría='{default_category}', Notas='{default_notes}'")
        print("   (Puedes escribir 'ok' para mantenerlos)")

    while True:
        prompt = "➡️ Categoría (good/acceptable/bad/maybe): "
        category = input(prompt).strip().lower()
        if category == 'ok' and default_category:
            category = default_category
            break
        if category in ['good', 'acceptable', 'bad', 'maybe']:
            break
        print("❌ Opción inválida. Inténtalo de nuevo.")
    
    prompt_notes = "➡️ Notas (ej: 'mucho ruido de movimiento', 'q' para omitir): "
    notes = input(prompt_notes).strip()
    if notes.lower() == 'ok' and default_notes:
        notes = default_notes
    elif notes.lower() == 'q':
        notes = ""

    return category, notes

# --- BUCLE PRINCIPAL ---
validation_log = load_validation_log(log_file_path)

for subject in subjects:
    if subject not in validation_log["subjects"]:
        validation_log["subjects"][subject] = {}

    for session in sessions:
        for task in tasks:
            for run in runs:
                for acq in acqs:
                    # Crear una clave única para este bloque/run/sesión
                    file_key = f"ses-{session}_task-{task}_acq-{acq}_run-{run}"

                    # Construir la ruta BIDS para BrainVision
                    bids_path = BIDSPath(
                        subject=subject,
                        session=session,
                        task=task,
                        run=run,
                        acquisition=acq,
                        datatype="eeg",
                        extension=".vhdr",
                        root=bids_root,
                        check=False
                    )

                    if not bids_path.fpath.exists():
                        print(f"  [AVISO] Archivo no encontrado: {bids_path.fpath.name}. Saltando.")
                        continue

                    if file_key not in validation_log["subjects"][subject]:
                        validation_log["subjects"][subject][file_key] = {}
                    
                    print("-" * 70)
                    print(f"Procesando: Sujeto={subject}, Ses={session}, Task={task}, Run={run}, Acq={acq}")


                    try:
                        # Leer archivo raw usando mne-bids
                        raw = read_raw_bids(bids_path, verbose=False)
                        raw.load_data(verbose=False)
                        sfreq = raw.info['sfreq']

                        # Extraer eventos desde las anotaciones de BrainVision (.vmrk)
                        events, event_id = mne.events_from_annotations(raw, verbose=False)
                        
                        # Convertir eventos a tiempos en segundos para el recorte
                        event_times = events[:, 0] / sfreq

                        # Extraer señales DESPUÉS de recortar
                        df_physio = pd.DataFrame()
                        for ch_name in ['ECG', 'GSR', 'RESP']:
                            if ch_name in raw.ch_names:
                                df_physio[ch_name] = raw.get_data(picks=[ch_name])[0]

                        if df_physio.empty:
                            print("  [AVISO] No se encontraron canales fisiológicos. Saltando.")
                            continue

                        # El tiempo ahora empieza en 0 para los datos recortados
                        tiempo_segundos = raw.times 

                        # Generar gráfico unificado
                        fig, axes = plt.subplots(len(df_physio.columns), 1, figsize=(20, 10), sharex=True)
                        if len(df_physio.columns) == 1: axes = [axes]
                        
                        fig.suptitle(f"Validación [{file_key}] - Sujeto: {subject}", fontsize=16)
                        
                        plot_idx = 0
                        for signal_name in df_physio.columns:
                            ax = axes[plot_idx]
                            signal_data = df_physio[signal_name]
                            
                            if signal_name == 'GSR':
                                processed, _ = nk.eda_process(signal_data, sampling_rate=sfreq)
                                ax.plot(tiempo_segundos, processed["EDA_Clean"], label="EDA/GSR (Limpia)")
                                ax.set_ylabel("EDA (µS)")
                            elif signal_name == 'ECG':
                                processed, _ = nk.ecg_process(signal_data, sampling_rate=sfreq)
                                ax.plot(tiempo_segundos, processed["ECG_Clean"], label="ECG (Limpio)")
                                ax.set_ylabel("ECG (mV)")
                            elif signal_name == 'RESP':
                                processed, _ = nk.rsp_process(signal_data, sampling_rate=sfreq)
                                ax.plot(tiempo_segundos, processed["RSP_Clean"], label="RESP (Limpia)")
                                ax.set_ylabel("Respiración")
                            
                            ax.legend(loc='upper right')
                            ax.grid(True, alpha=0.4)
                            plot_idx += 1
                        
                        plt.xlabel(f"Tiempo (segundos)")
                        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                        
                        # Mostrar el gráfico de manera interactiva
                        plt.show(block=False) 
                        plt.pause(1)

                        # Bucle de validación manual
                        for signal_name in df_physio.columns:
                            signal_type_map = {'GSR': 'eda', 'ECG': 'ecg', 'RESP': 'resp'}
                            signal_type = signal_type_map[signal_name]
                            
                            category, notes = get_manual_validation_input(validation_log, subject, file_key, signal_type)
                            
                            validation_log["subjects"][subject][file_key][signal_type] = {
                                "category": category,
                                "notes": notes,
                                "validated_at": datetime.now().isoformat()
                            }
                        
                        save_validation_log(log_file_path, validation_log)
                        plt.close(fig)

                    except Exception as e:
                        print(f"  [ERROR] Ocurrió un error inesperado al procesar {file_key}: {e}")
                        import traceback
                        traceback.print_exc()
                        plt.close('all')
                        continue

print("-" * 70)
print("Proceso de validación completado. Log guardado en:", log_file_path)