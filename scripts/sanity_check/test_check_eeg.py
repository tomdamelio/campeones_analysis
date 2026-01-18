import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne
from mne_bids import BIDSPath, read_raw_bids

def process_eeg_data(bids_path):
    """Procesa los datos EEG de un archivo BIDS."""
    print(f"\nProcesando archivo: {bids_path.fpath}")

    # Leer archivo raw usando mne-bids
    print("Leyendo archivo raw BIDS")
    # For EEG we might want to be careful with preloading if files are huge, 
    # but for sanity check it's usually fine.
    raw = read_raw_bids(bids_path, verbose=True)

    print("Cargando datos raw")
    raw.load_data()
    
    # Mostrar información del archivo
    print(f"\n📊 INFORMACIÓN DEL ARCHIVO:")
    print(f"   Duración: {raw.times[-1]:.2f} segundos ({raw.times[-1]/60:.2f} minutos)")
    print(f"   Frecuencia de muestreo: {raw.info['sfreq']} Hz")
    print(f"   Número de canales: {len(raw.ch_names)}")
    
    # Verificar canales EEG
    print("   Tipos de canales encontrados:", raw.get_channel_types(unique=True))
    
    # Plotear datos
    print("\nGenerando plot de señales EEG...")
    
    # Generate PSD plot
    print("Generando PSD de la señal...")
    raw.compute_psd(fmax=50).plot(show=True)

    # Scalings for EEG might need adjustment, but auto usually works
    # block=True is essential to keep the window open when running from script
    # We use scalings='auto' to ensure signals are visible
    raw.plot(duration=10, n_channels=20, scalings='auto', title=f"EEG - {bids_path.basename}", block=True)

def main():
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description="Procesar y visualizar datos EEG de archivos BIDS"
    )
    parser.add_argument("--subject", type=str, default="18", 
                        help="ID del sujeto (default: 18)")
    parser.add_argument("--session", type=str, default="vr",
                        help="ID de la sesión (default: vr)")
    parser.add_argument("--task", type=str, default="02",
                        help="ID de la tarea (default: 02)")
    parser.add_argument("--run", type=str, default=None,
                        help="ID del run (opcional si es único)")
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
    print(f"  Run: {run if run else 'Auto-detect'}")
    print(f"  Acquisition: {acq}")
    print(f"  BIDS root: {bids_root}")

    # Si no se especifica run, buscar el archivo automáticamente
    if run is None:
        subject_dir = bids_root / f"sub-{subject}" / f"ses-{session}" / "eeg"
        if not subject_dir.exists():
            print(f"\n❌ El directorio {subject_dir} no existe")
            return

        # Patrón de búsqueda: sub-XX_ses-YY_task-ZZ_acq-AA_run-*_eeg.vhdr
        pattern = f"sub-{subject}_ses-{session}_task-{task}_acq-{acq}_run-*_eeg.vhdr"
        found_files = list(subject_dir.glob(pattern))
        
        if not found_files:
            print(f"\n❌ No se encontraron archivos coincidiendo con: {pattern}")
            print(f"En {subject_dir}")
            return
        
        if len(found_files) > 1:
            print(f"\n⚠️ Múltiples archivos encontrados para task={task}, acq={acq}:")
            for f in found_files:
                print(f"  - {f.name}")
            print("Por favor especifica el run usando --run")
            return
        
        # Tomar el único archivo encontrado y extraer el run
        target_file = found_files[0]
        # Extraer el run del nombre del archivo (asumiendo formato estándar BIDS)
        # sub-18_ses-vr_task-02_acq-b_run-007_eeg.vhdr
        parts = target_file.stem.split('_')
        for part in parts:
            if part.startswith('run-'):
                run = part.split('-')[1]
                break
        
        print(f"✅ Archivo encontrado automáticamente: {target_file.name}")
        print(f"✅ Run detectado: {run}")
    
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
        
        # Buscar archivos similares en el directorio para ayudar al usuario
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
    process_eeg_data(bids_path)


if __name__ == "__main__":
    main()
