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

# Definir ruta base
repo_root = Path(__file__).resolve()
while not (repo_root / ".git").exists() and repo_root != repo_root.parent:
    repo_root = repo_root.parent

sub = "all_subs"
physio_dir = repo_root / "data" / "derivatives" / "physio" 

idx_file = physio_dir / f"idx_data_{sub}.npz"
data_file = physio_dir / f"{sub}_desc-physio_features.npz"

# Imprimir índices
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