"""
Visualización de la señal joystick_x con z-score.

Uso:
    micromamba run -n campeones python scripts/diagnostics/visualize_joystick_zscore.py \
        --subject 39 --session vr --task 02 --acq b
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore

# Agregar repo root al path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from mne_bids import BIDSPath, read_raw_bids


def main():
    parser = argparse.ArgumentParser(description="Visualizar joystick_x z-scoreado")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session", default="vr")
    parser.add_argument("--task", required=True)
    parser.add_argument("--acq", default=None)
    parser.add_argument("--channel", default="joystick_x", help="Canal a visualizar")
    args = parser.parse_args()

    task = args.task.zfill(2) if args.task.isdigit() else args.task
    acq = args.acq.lower() if args.acq else None

    # Buscar archivo
    import glob, re
    bids_root = repo_root / "data" / "raw"
    pattern = f"sub-{args.subject}/ses-{args.session}/eeg/sub-{args.subject}_ses-{args.session}_task-{task}_acq-{acq}_run-*_eeg.vhdr"
    matches = glob.glob(str(bids_root / pattern))

    if not matches:
        print(f"No se encontró archivo para el patrón: {pattern}")
        sys.exit(1)

    run = re.search(r"_run-(\d+)_", Path(matches[0]).name).group(1)
    print(f"Archivo: {Path(matches[0]).name} (run={run})")

    bids_path = BIDSPath(
        subject=args.subject, session=args.session, task=task,
        run=run, acquisition=acq, datatype="eeg",
        root=bids_root, extension=".vhdr",
    )

    raw = read_raw_bids(bids_path, verbose=False)
    raw.load_data()

    if args.channel not in raw.ch_names:
        print(f"Canal '{args.channel}' no encontrado. Canales disponibles:")
        for ch in raw.ch_names:
            print(f"  - {ch}")
        sys.exit(1)

    # Extraer datos y aplicar z-score
    data = raw.get_data(picks=[args.channel])[0]
    times = raw.times
    data_z = zscore(data)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    axes[0].plot(times, data, linewidth=0.5, color="steelblue")
    axes[0].set_title(f"{args.channel} - señal original (sub-{args.subject} task-{task} acq-{acq} run-{run})")
    axes[0].set_ylabel("Amplitud (raw)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, data_z, linewidth=0.5, color="darkorange")
    axes[1].set_title(f"{args.channel} - z-score")
    axes[1].set_ylabel("Z-score")
    axes[1].set_xlabel("Tiempo (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Stats
    print(f"\n--- Estadísticas de {args.channel} ---")
    print(f"  Min: {data.min():.6f}")
    print(f"  Max: {data.max():.6f}")
    print(f"  Mean: {data.mean():.6f}")
    print(f"  Std: {data.std():.6f}")
    print(f"  Rango: {data.max() - data.min():.6f}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
