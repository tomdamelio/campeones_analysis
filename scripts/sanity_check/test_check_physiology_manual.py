import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import neurokit2 as nk
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids


def process_physiology_data(bids_path):
    """Procesa los datos fisiológicos de un archivo BIDS."""
    print(f"\nProcesando archivo: {bids_path.fpath}")

    # Leer archivo raw usando mne-bids
    print("Leyendo archivo raw BIDS")
    raw = read_raw_bids(bids_path, verbose=True)

    print("Cargando datos raw")
    raw.load_data()
    
    # Mostrar información del archivo
    print(f"\n📊 INFORMACIÓN DEL ARCHIVO:")
    print(f"   Duración: {raw.times[-1]:.2f} segundos ({raw.times[-1]/60:.2f} minutos)")
    print(f"   Frecuencia de muestreo: {raw.info['sfreq']} Hz")
    print(f"   Número de canales: {len(raw.ch_names)}")
    print(f"   Canales: {raw.ch_names}")

    # Obtener datos y tiempos
    data, times = raw.get_data(return_times=True)
    df = pd.DataFrame(data.T, columns=raw.ch_names)
    df["Time"] = times
    
    print(f"   Forma de los datos: {df.shape} (filas=muestras, columnas=canales+tiempo)")

    # Verificar qué canales fisiológicos están disponibles
    physio_channels = ['GSR', 'ECG', 'RESP']
    available_channels = [ch for ch in physio_channels if ch in df.columns]
    
    if not available_channels:
        print("❌ No se encontraron canales fisiológicos (GSR, ECG, RESP)")
        print(f"Canales disponibles: {list(df.columns)}")
        return
    
    print(f"✅ Canales fisiológicos encontrados: {available_channels}")

    # Procesar EDA/GSR
    if 'GSR' in available_channels:
        try:
            print("\nProcesando señal EDA/GSR...")
            # Invertir la señal EDA (multiplicar por -1) debido a polaridad invertida
            gsr_signal = -df["GSR"]
            print("⚠️  Señal EDA invertida (multiplicada por -1) debido a polaridad invertida")
            signals_eda, info_eda = nk.eda_process(gsr_signal, sampling_rate=raw.info["sfreq"])
            plt.figure(figsize=(12, 6))
            plt.title("Análisis de EDA/GSR (Señal Invertida)")
            nk.eda_plot(signals_eda, info=info_eda)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"❌ Error procesando EDA/GSR: {e}")

    # Procesar ECG
    if 'ECG' in available_channels:
        try:
            print("\nProcesando señal ECG...")
            # Invertir la señal EDA (multiplicar por -1) debido a polaridad invertida
            ecg = -df["ECG"]
            #print("⚠️  Señal ECG invertida (multiplicada por -1) debido a polaridad invertida")
            signals_ecg, info_ecg = nk.ecg_process(ecg, sampling_rate=raw.info["sfreq"])
            plt.figure(figsize=(12, 8))
            plt.title("Análisis de ECG")
            nk.ecg_plot(signals_ecg, info=info_ecg)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"❌ Error procesando ECG: {e}")

    # Procesar RESP
    if 'RESP' in available_channels:
        try:
            print("\nProcesando señal RESP...")
            signals_resp, info_resp = nk.rsp_process(
                df["RESP"], sampling_rate=raw.info["sfreq"]
            )
            plt.figure(figsize=(12, 6))
            plt.title("Análisis de Respiración")
            nk.rsp_plot(signals_resp, info=info_resp)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"❌ Error procesando RESP: {e}")
    
    print(f"\n✅ Procesamiento completado para {len(available_channels)} señales fisiológicas")


def main():
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description="Procesar y visualizar datos fisiológicos de archivos BIDS"
    )
    parser.add_argument("--subject", type=str, default="18", 
                        help="ID del sujeto (default: 18)")
    parser.add_argument("--session", type=str, default="vr",
                        help="ID de la sesión (default: vr)")
    parser.add_argument("--task", type=str, default="02",
                        help="ID de la tarea (default: 02)")
    parser.add_argument("--run", type=str, default="007",
                        help="ID del run (default: 007)")
    parser.add_argument("--acq", type=str, default="b",
                        help="ID de adquisición (default: b)")
    
    args = parser.parse_args()
    
    # Configuración de paths - buscar en data/raw
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # Subir dos niveles para llegar a la raíz del proyecto
    bids_root = project_root / "data" / "raw"

    # Usar parámetros de línea de comandos o valores por defecto
    subject = args.subject
    session = args.session
    task = args.task
    run = args.run
    acq = args.acq

    print(f"Buscando archivo para:")
    print(f"  Subject: {subject}")
    print(f"  Session: {session}")
    print(f"  Task: {task}")
    print(f"  Run: {run}")
    print(f"  Acquisition: {acq}")
    print(f"  BIDS root: {bids_root}")

    # Crear BIDSPath para el archivo
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        acquisition=acq,
        datatype="eeg",
        extension=".vhdr",  # BrainVision header file
        root=bids_root,
        check=False  # No verificar validez, solo construir el path
    )

    print(f"\nPath esperado: {bids_path.fpath}")
    
    # Verificar que el archivo existe
    if not bids_path.fpath.exists():
        print(f"\n❌ ERROR: El archivo no existe: {bids_path.fpath}")
        
        # Buscar archivos similares en el directorio
        subject_dir = bids_root / f"sub-{subject}" / f"ses-{session}" / "eeg"
        if subject_dir.exists():
            print(f"\nArchivos disponibles en {subject_dir}:")
            for file in subject_dir.glob("*.vhdr"):
                print(f"  - {file.name}")
        else:
            print(f"\n❌ El directorio {subject_dir} no existe")
        return
    
    print(f"✅ Archivo encontrado: {bids_path.fpath}")
    
    # Verificar archivos asociados
    eeg_file = bids_path.fpath.with_suffix('.eeg')
    vmrk_file = bids_path.fpath.with_suffix('.vmrk')
    
    if not eeg_file.exists():
        print(f"❌ Archivo .eeg faltante: {eeg_file}")
        return
        
    if not vmrk_file.exists():
        print(f"❌ Archivo .vmrk faltante: {vmrk_file}")
        return
    
    print(f"✅ Todos los archivos BrainVision están presentes")

    # Procesar los datos
    process_physiology_data(bids_path)


if __name__ == "__main__":
    main()
