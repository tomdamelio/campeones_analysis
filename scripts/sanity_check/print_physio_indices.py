#!/usr/bin/env python
"""
Script para imprimir y comparar los índices de segmentos de fisiología de un sujeto específico (sub-14) en Campeones Analysis.

Abre e imprime la forma y todas las filas de:
- idx_data_sub-14_concatenated.npz

Además, compara el valor máximo de los índices de clean con la cantidad de filas del archivo de datos:
- sub-14_desc-physio_features_concatenated.npz

en data/derivatives/physio/sub-14.
"""
import numpy as np
from pathlib import Path

sampling_rate = 250.0  # Hz

# Definir ruta base
repo_root = Path(__file__).resolve()
while not (repo_root / ".git").exists() and repo_root != repo_root.parent:
    repo_root = repo_root.parent

sub = "14"
physio_dir = repo_root / "data" / "derivatives" / "physio" / f"sub-{sub}"

idx_file = physio_dir / f"idx_data_sub-{sub}_concatenated.npz"
data_file = physio_dir / f"sub-{sub}_desc-physio_features_concatenated.npz"
old_idx_file = physio_dir / f"idx_data_OLD_timepoints_sub-{sub}_concatenated.npz"

# Imprimir índices clean
print(f"\n{'='*60}")
print(f"Clean indices (physio)")
print(f"Archivo: {idx_file}")
if not idx_file.exists():
    print(f"❌ Archivo no encontrado")
    clean_indices = None
else:
    arr = np.load(idx_file)['arr_0']
    print(f"Forma del array: {arr.shape}")
    print(f"Todas las filas:")
    for i, row in enumerate(arr):
        print(f"{i:3d}: {row}")
    print(f"Min: {arr.min()}  Max: {arr.max()}")
    clean_indices = arr

# Imprimir índices originales (OLD timepoints)
print(f"\n{'='*60}")
print(f"Original indices (OLD timepoints)")
print(f"Archivo: {old_idx_file}")
if not old_idx_file.exists():
    print(f"❌ Archivo no encontrado")
    old_indices = None
else:
    arr_old = np.load(old_idx_file)['arr_0']
    print(f"Forma del array: {arr_old.shape}")
    print(f"Todas las filas (en segundos, fs={sampling_rate} Hz):")
    for i, row in enumerate(arr_old):
        start_sec = row[0] / sampling_rate
        end_sec = row[1] / sampling_rate
        print(f"{i:3d}: [{start_sec:.3f} s, {end_sec:.3f} s]")
    print(f"Min: {arr_old.min()/sampling_rate:.3f} s  Max: {arr_old.max()/sampling_rate:.3f} s")
    old_indices = arr_old

# Comparar el max index de clean_indices con la cantidad de filas del archivo de datos
print(f"\n{'='*60}")
print(f"Comparando max index de clean_indices con la cantidad de filas del archivo de datos:")
print(f"Archivo de datos: {data_file}")
if clean_indices is None or not data_file.exists():
    print(f"❌ No se puede comparar (faltan archivos)")
else:
    data_arr = np.load(data_file)['arr_0']
    n_rows = data_arr.shape[0]
    max_clean_idx = clean_indices.max()
    print(f"Cantidad de filas en datos: {n_rows}")
    print(f"Max index en clean_indices: {max_clean_idx}")
    if max_clean_idx <= n_rows:
        print(f"✅ Relación válida: max_clean_idx <= n_rows")
    else:
        print(f"❌ max_clean_idx > n_rows (posible error de índices)")

# Imprimir filelog_concatenate
filelog_path = physio_dir / f"filelog_concatenate_sub-{sub}.csv"
print(f"\n{'='*60}")
print(f"Filelog de archivos concatenados (physio)")
print(f"Archivo: {filelog_path}")
if not filelog_path.exists():
    print(f"❌ Archivo no encontrado")
else:
    with open(filelog_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            print(f"{i:3d}: {line.strip()}")

# Imprimir idx_data_OLD_timepoints_in_seconds_from_start
old_seconds_path = physio_dir / f"idx_data_OLD_timepoints_in_seconds_from_start_sub-{sub}_concatenated.npz"
print(f"\n{'='*60}")
print(f"OLD timepoints in seconds from start (physio)")
print(f"Archivo: {old_seconds_path}")
if not old_seconds_path.exists():
    print(f"❌ Archivo no encontrado")
else:
    arr_sec = np.load(old_seconds_path)['arr_0']
    print(f"Forma del array: {arr_sec.shape}")
    print(f"Todas las filas:")
    for i, row in enumerate(arr_sec):
        print(f"{i:3d}: {row}")
    print(f"Min: {arr_sec.min()}  Max: {arr_sec.max()}") 