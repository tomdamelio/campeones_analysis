#!/usr/bin/env python
"""
Script para comparar los índices de todos los sujetos entre physiological features y TFR data.

Compara:
- data/derivatives/physio/idx_data_all_subs.npz
- data/derivatives/trf/idx_data_all_subs.npz

Los índices deberían ser idénticos ya que ambos procesan los mismos archivos concatenados.
"""
import numpy as np
from pathlib import Path

def inspect_npz_file(filepath):
    """Inspect the keys and structure of an .npz file"""
    try:
        data = np.load(filepath)
        print(f"📁 Archivo: {filepath}")
        print(f"🔑 Claves disponibles: {list(data.keys())}")
        
        for key in data.keys():
            arr = data[key]
            print(f"   - {key}: shape={arr.shape}, dtype={arr.dtype}")
            if len(arr.shape) <= 2 and arr.size < 100:  # Show small arrays
                print(f"     Contenido: {arr}")
        print()
        return data
    except Exception as e:
        print(f"❌ Error inspeccionando {filepath}: {e}")
        return None

def load_and_print_indices(filepath, label):
    """Load and print indices from .npz file"""
    try:
        data = np.load(filepath)
        print(f"============================================================")
        print(f"{label}")
        print(f"Archivo: {filepath}")
        print(f"Claves disponibles: {list(data.keys())}")
        
        # Try to find the indices array - common names might be:
        possible_keys = ['idx_data', 'indices', 'clean_indices', 'arr_0', 'data']
        indices_key = None
        
        for key in possible_keys:
            if key in data.keys():
                indices_key = key
                break
        
        if indices_key is None:
            # If no standard key found, use the first array that looks like indices
            for key in data.keys():
                arr = data[key]
                if len(arr.shape) == 2 and arr.shape[1] == 2:  # Looks like indices (n_segments, 2)
                    indices_key = key
                    break
        
        if indices_key is None:
            print(f"❌ No se pudo encontrar un array de índices en el archivo")
            return None
            
        indices = data[indices_key]
        print(f"Usando clave: '{indices_key}'")
        print(f"Forma del array: {indices.shape}")
        print(f"Todas las filas:")
        for i, row in enumerate(indices):
            print(f"  {i:2d}: [{row[0]:8d} {row[1]:8d}]")
        print(f"Min: {indices.min()}  Max: {indices.max()}")
        print(f"Total segments: {len(indices)}")
        print(f"Total timepoints span: {indices[-1, 1] - indices[0, 0]}")
        return indices
    except Exception as e:
        print(f"❌ Error cargando {filepath}: {e}")
        return None

def compare_indices(indices1, indices2, label1, label2):
    """Compare two index arrays"""
    print(f"============================================================")
    print(f"COMPARACIÓN ENTRE {label1.upper()} Y {label2.upper()}")
    print(f"============================================================")
    
    if indices1 is None or indices2 is None:
        print("❌ No se pueden comparar - uno o ambos archivos fallaron al cargar")
        return
    
    # Check shapes
    if indices1.shape != indices2.shape:
        print(f"❌ DIFERENTES FORMAS: {label1}: {indices1.shape}, {label2}: {indices2.shape}")
        return
    else:
        print(f"✅ Misma forma: {indices1.shape}")
    
    # Check if arrays are identical
    if np.array_equal(indices1, indices2):
        print(f"✅ ÍNDICES IDÉNTICOS: Los archivos de {label1} y {label2} tienen exactamente los mismos índices")
    else:
        print(f"❌ ÍNDICES DIFERENTES: Los archivos de {label1} y {label2} NO son idénticos")
        
        # Find differences
        diff_mask = (indices1 != indices2)
        if diff_mask.any():
            print(f"Diferencias encontradas en {diff_mask.sum()} posiciones:")
            diff_positions = np.where(diff_mask)
            for i in range(min(10, len(diff_positions[0]))):  # Show first 10 differences
                row, col = diff_positions[0][i], diff_positions[1][i]
                print(f"  Posición [{row}, {col}]: {label1}={indices1[row, col]}, {label2}={indices2[row, col]}")
            if len(diff_positions[0]) > 10:
                print(f"  ... y {len(diff_positions[0]) - 10} diferencias más")
    
    # Check total span
    span1 = indices1[-1, 1] - indices1[0, 0]
    span2 = indices2[-1, 1] - indices2[0, 0]
    if span1 == span2:
        print(f"✅ Mismo span total de timepoints: {span1}")
    else:
        print(f"❌ Diferentes spans: {label1}={span1}, {label2}={span2}")

def main():
    # Define paths
    repo_root = Path(__file__).resolve()
    while not (repo_root / ".git").exists() and repo_root != repo_root.parent:
        repo_root = repo_root.parent
    
    physio_path = repo_root / "data" / "derivatives" / "physio" / "idx_data_all_subs.npz"
    trf_path = repo_root / "data" / "derivatives" / "trf" / "idx_data_all_subs.npz"
    
    print("COMPARACIÓN DE ÍNDICES DE TODOS LOS SUJETOS")
    print("===========================================")
    print("Comparando índices entre physiological features y TFR data")
    print()
    
    # First, inspect both files to see their structure
    print("🔍 INSPECCIÓN INICIAL DE ARCHIVOS:")
    print("=" * 50)
    physio_data = inspect_npz_file(physio_path)
    trf_data = inspect_npz_file(trf_path)
    
    if physio_data is None or trf_data is None:
        print("❌ No se pueden inspeccionar uno o ambos archivos")
        return
    
    # Load and print both files
    physio_indices = load_and_print_indices(physio_path, "PHYSIOLOGICAL FEATURES INDICES")
    print()
    trf_indices = load_and_print_indices(trf_path, "TFR INDICES")
    print()
    
    # Compare them
    compare_indices(physio_indices, trf_indices, "PHYSIO", "TFR")

if __name__ == "__main__":
    main() 