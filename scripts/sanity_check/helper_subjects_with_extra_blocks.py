import mne
from mne_bids import BIDSPath, read_raw_bids
import argparse
import matplotlib
from pathlib import Path
import os
from datetime import datetime

# Configurar el backend para que la ventana de MNE sea interactiva
try:
    matplotlib.use('Qt5Agg')
except ImportError:
    print("Qt5Agg backend not found, trying TkAgg...")
    matplotlib.use('TkAgg')

# --- VALORES POR DEFECTO ---
DEFAULT_SUBJECTS = ["36"]
DEFAULT_SESSIONS = ["vr"]
DEFAULT_TASKS = ["01", "02", "03", "04"]
DEFAULT_RUNS = ["002", "003", "004", "005", "006", "007", "008", "009"]
DEFAULT_ACQS = ["a", "b"]

def parse_arguments():
    """Configura la lectura de argumentos desde la terminal."""
    parser = argparse.ArgumentParser(description="Visualizador secuencial de EEG ordenado por fecha original XDF.")
    
    parser.add_argument('--subject', nargs='+', default=DEFAULT_SUBJECTS, help='Sujeto(s) a procesar o "all"')
    parser.add_argument('--task', nargs='+', default=DEFAULT_TASKS, help='Tarea(s) a procesar o "all"')
    parser.add_argument('--acq', nargs='+', default=DEFAULT_ACQS, help='Adquisición(es) a procesar o "all"')
    parser.add_argument('--session', nargs='+', default=DEFAULT_SESSIONS, help='Sesión(es) o "all"')
    parser.add_argument('--run', nargs='+', default=DEFAULT_RUNS, help='Run(s) o "all"')

    args = parser.parse_args()

    subjects = DEFAULT_SUBJECTS if "all" in args.subject else args.subject
    tasks = DEFAULT_TASKS if "all" in args.task else args.task
    acqs = DEFAULT_ACQS if "all" in args.acq else args.acq
    sessions = DEFAULT_SESSIONS if "all" in args.session else args.session
    runs = DEFAULT_RUNS if "all" in args.run else args.run

    return subjects, sessions, tasks, runs, acqs

def main():
    subjects, sessions, tasks, runs, acqs = parse_arguments()

    # --- CONFIGURACIÓN DE RUTAS ---
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent 
    bids_root = project_root / "data" / "raw"
    xdf_base_root = project_root / "data" / "sourcedata" / "xdf" # <--- Nueva ruta base original

    print(f"Buscando BIDS en: {bids_root}")
    print(f"Buscando fechas en XDF en: {xdf_base_root}")
    print(f"Filtros -> Subjects: {subjects} | Tasks: {tasks} | Acqs: {acqs}")

    # =========================================================================
    # FASE 1: BÚSQUEDA Y RECOLECCIÓN DE ARCHIVOS
    # =========================================================================
    archivos_encontrados = []

    for subject in subjects:
        for session in sessions:
            for task in tasks:
                for run in runs:
                    for acq in acqs:
                        file_key = f"ses-{session}_task-{task}_acq-{acq}_run-{run}"

                        # 1. Ruta del archivo BIDS que vamos a plotear
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

                        if bids_path.fpath.exists():
                            # 2. Reconstruir la ruta del XDF original según tus indicaciones
                            # Carpeta: sub-{SUBJECT}/ses-VR/physio
                            xdf_dir = xdf_base_root / f"sub-{subject}" / "ses-VR" / "physio"
                            # Archivo: sub-{SUBJECT}_ses-vr_day-{ACQ}_task-{TASK}_run-{RUN}_eeg.xdf
                            xdf_filename = f"sub-{subject}_ses-{session}_day-{acq}_task-{task}_run-{run}_eeg.xdf"
                            xdf_path = xdf_dir / xdf_filename

                            # 3. Obtener la fecha de modificación
                            if xdf_path.exists():
                                mtime = xdf_path.stat().st_mtime
                                origen_fecha = "XDF"
                            else:
                                print(f"  [AVISO] No se encontró el original {xdf_filename}. Usando fecha del BIDS como plan B.")
                                mtime = bids_path.fpath.stat().st_mtime
                                origen_fecha = "BIDS"
                            
                            # Guardar toda la info en un diccionario para usarla luego
                            archivos_encontrados.append({
                                'bids_path': bids_path,
                                'file_key': file_key,
                                'subject': subject,
                                'session': session,
                                'task': task,
                                'run': run,
                                'acq': acq,
                                'mtime': mtime,
                                'origen_fecha': origen_fecha # Solo para saber de dónde sacó la fecha
                            })

    if not archivos_encontrados:
        print("No se encontraron archivos BIDS que coincidan con los filtros.")
        return

    # =========================================================================
    # FASE 2: ORDENAR POR FECHA (ORIGINAL) Y VISUALIZAR
    # =========================================================================
    # Ordenar la lista basándose en el 'mtime' (de menor a mayor = de más viejo a más nuevo)
    archivos_encontrados.sort(key=lambda x: x['mtime'])

    print("-" * 70)
    print(f"Se encontraron {len(archivos_encontrados)} archivos. Abriendo por orden cronológico (XDF original)...")
    print("-" * 70)

    # Bucle simple sobre la lista ya ordenada
    for info_archivo in archivos_encontrados:
        bids_path = info_archivo['bids_path']
        file_key = info_archivo['file_key']
        subject = info_archivo['subject']
        origen_fecha = info_archivo['origen_fecha']
        
        # Convertir timestamp a fecha legible
        fecha_legible = datetime.fromtimestamp(info_archivo['mtime']).strftime('%d/%m/%Y %H:%M:%S')
        
        print(f"Abriendo BIDS: {file_key}")
        print(f"Fecha ({origen_fecha}): {fecha_legible}")

        try:
            # Leer archivo raw usando mne-bids
            raw = read_raw_bids(bids_path, verbose=False)
            raw.load_data(verbose=False)

            canales_deseados = ['AUDIO', 'PHOTO', 'joystick_x']
            
            # Validamos que los canales existan en el archivo para evitar errores
            canales_presentes = [ch for ch in canales_deseados if ch in raw.ch_names]
            
            if not canales_presentes:
                print("  [AVISO] No se encontraron los canales deseados en este archivo. Saltando.")
                print("-" * 70)
                continue
                
            # Dejar solo los canales deseados y descartar el resto
            raw.pick(canales_presentes)

            # PLOTEAR EN MNE
            fig = raw.plot(
                block=True, 
                title=f"Sujeto: {subject} | {file_key} | {fecha_legible} ({origen_fecha})",
                duration=10.0, 
                n_channels=len(canales_presentes)
            )

            print(f"Cerrado: {file_key}. Buscando siguiente archivo...\n" + "-" * 70)

        except Exception as e:
            print(f"  [ERROR] Ocurrió un error al procesar {file_key}: {e}")
            print("-" * 70)
            continue

    print("Ya no hay más archivos que cumplan con los filtros especificados.")

if __name__ == "__main__":
    main()