#!/usr/bin/env python
"""
physio Data Validation Script

This script validates the consistency of concatenated physiological data and indices,
ensuring that the order of concatenation is the same for both data and indices.

Features:
- Verifies file ordering consistency
- Validates data shapes and dimensions
- Checks index continuity and gaps
- Compares individual files with concatenated results
- Generates comprehensive validation report

Usage:
    python scripts/diagnostics/validate_tfr_data.py --sub 14
    python scripts/diagnostics/validate_tfr_data.py --sub 14 --verbose
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import argparse
import glob
from pathlib import Path

# Find the repository root
script_path = Path(__file__).resolve()
repo_root = script_path
while not (repo_root / ".git").exists() and repo_root != repo_root.parent:
    repo_root = repo_root.parent


def extract_run_info_from_filename(filename):
    """
    Extract run information from filename for sorting.
    
    Parameters
    ----------
    filename : str or Path
        physio filename
        
    Returns
    -------
    dict
        Dictionary with session, task, acquisition, run info
    """
    filename = Path(filename).name
    parts = filename.split('_')
    
    info = {}
    for part in parts:
        if part.startswith('ses-'):
            info['session'] = part[4:]
        elif part.startswith('task-'):
            info['task'] = part[5:]
        elif part.startswith('acq-'):
            info['acquisition'] = part[4:]
        elif part.startswith('run-'):
            info['run'] = part[4:]
    
    return info


def create_sort_key(filename):
    """
    Create a sorting key from filename to ensure consistent ordering.
    
    Parameters
    ----------
    filename : str or Path
        physio filename
        
    Returns
    -------
    tuple
        Sorting key (session, task, acquisition, run)
    """
    info = extract_run_info_from_filename(filename)
    
    # Convert to sortable format
    session = info.get('session', 'zz')
    task = info.get('task', '99')
    acquisition = info.get('acquisition', 'z')
    run = info.get('run', '999')
    
    return (session, task, acquisition, run)


def load_individual_files(subject, trf_output_dir):
    """
    Load all individual physio files for a subject in sorted order.
    
    Parameters
    ----------
    subject : str
        Subject ID
    trf_output_dir : Path
        Directory containing physio files
        
    Returns
    -------
    dict
        Dictionary with loaded data and metadata
    """
    print(f"🔍 Loading individual physio files for subject {subject}")
    
    # Find all individual physio files
    tfr_pattern = f"sub-{subject}_*_desc-morlet_tfr.npz"
    tfr_files = list(trf_output_dir.glob(tfr_pattern))
    
    # Exclude concatenated files
    tfr_files = [f for f in tfr_files if 'concatenated' not in f.name]
    
    if not tfr_files:
        print(f"❌ No individual physio files found for subject {subject}")
        return None
    
    # Sort files using consistent ordering
    tfr_files.sort(key=create_sort_key)
    
    print(f"Found {len(tfr_files)} individual files:")
    
    individual_data = {
        'files': [],
        'data': [],
        'clean_indices': [],
        'original_indices': [],
        'metadata': [],
        'file_order': []
    }
    
    for i, tfr_file in enumerate(tfr_files):
        print(f"  {i+1:2d}. {tfr_file.name}")
        
        # Extract run info for tracking
        run_info = extract_run_info_from_filename(tfr_file)
        individual_data['file_order'].append(run_info)
        individual_data['files'].append(tfr_file)
        
        # Load physio data
        tfr_data = np.load(tfr_file)
        data = tfr_data['arr_0']
        individual_data['data'].append(data)
        
        # Load metadata
        base_name = tfr_file.stem.replace('_desc-morlet_tfr', '')
        metadata_file = tfr_file.parent / f"{base_name}_desc-morlet_tfr.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            individual_data['metadata'].append(metadata)
        else:
            individual_data['metadata'].append({})
        
        # Load clean indices
        clean_idx_file = tfr_file.parent / f"idx_data_{base_name}.npz"
        if clean_idx_file.exists():
            clean_indices = np.load(clean_idx_file)['arr_0']
            individual_data['clean_indices'].append(clean_indices)
        else:
            individual_data['clean_indices'].append(None)
        
        # Load original indices
        original_idx_file = tfr_file.parent / f"idx_data_OLD_timepoints_{base_name}.npz"
        if original_idx_file.exists():
            original_indices = np.load(original_idx_file)['arr_0']
            individual_data['original_indices'].append(original_indices)
        else:
            individual_data['original_indices'].append(None)
        
        print(f"      Data shape: {data.shape}")
        if clean_indices is not None:
            print(f"      Clean indices: {clean_indices.shape[0]} segments")
        if original_indices is not None:
            print(f"      Original indices: {original_indices.shape[0]} segments")
    
    return individual_data


def load_concatenated_files(subject, trf_output_dir):
    """
    Load concatenated physio files for a subject from subdirectory sub-{subject}.
    """
    print(f"\n🔍 Buscando archivos concatenados para el sujeto {subject}")
    subject_dir = trf_output_dir / f"sub-{subject}"
    if not subject_dir.exists():
        print(f"❌ No se encontró el directorio del sujeto: {subject_dir}")
        return None
    concatenated_data = {}
    # physio data
    concat_tfr_file = subject_dir / f"sub-{subject}_desc-physio_features_concatenated.npz"
    if concat_tfr_file.exists():
        tfr_data = np.load(concat_tfr_file)
        concatenated_data['data'] = tfr_data['arr_0']
        print(f"✓ physio concatenado cargado: {concatenated_data['data'].shape}")
    else:
        print(f"❌ No se encontró el archivo physio concatenado: {concat_tfr_file}")
        return None
    # Clean indices
    concat_clean_idx_file = subject_dir / f"idx_data_sub-{subject}_concatenated.npz"
    if concat_clean_idx_file.exists():
        clean_indices = np.load(concat_clean_idx_file)
        concatenated_data['clean_indices'] = clean_indices['arr_0']
        print(f"✓ Índices limpios concatenados cargados: {concatenated_data['clean_indices'].shape}")
    else:
        print(f"⚠️  No se encontró el archivo de índices limpios concatenados: {concat_clean_idx_file}")
        concatenated_data['clean_indices'] = None
    # Original indices
    concat_original_idx_file = subject_dir / f"idx_data_OLD_timepoints_sub-{subject}_concatenated.npz"
    if concat_original_idx_file.exists():
        original_indices = np.load(concat_original_idx_file)
        concatenated_data['original_indices'] = original_indices['arr_0']
        print(f"✓ Índices originales concatenados cargados: {concatenated_data['original_indices'].shape}")
    else:
        print(f"⚠️  No se encontró el archivo de índices originales concatenados: {concat_original_idx_file}")
        concatenated_data['original_indices'] = None
    # Metadata
    concat_metadata_file = subject_dir / f"sub-{subject}_desc-physio_features_concatenated.json"
    if concat_metadata_file.exists():
        with open(concat_metadata_file, 'r') as f:
            concatenated_data['metadata'] = json.load(f)
        print(f"✓ Metadata concatenada cargada")
    else:
        print(f"⚠️  No se encontró el archivo de metadata concatenada: {concat_metadata_file}")
        concatenated_data['metadata'] = {}
    return concatenated_data


def validate_file_ordering(individual_data, concatenated_data):
    """
    Validate that the file ordering is consistent between individual and concatenated data.
    
    Parameters
    ----------
    individual_data : dict
        Individual files data
    concatenated_data : dict
        Concatenated files data
        
    Returns
    -------
    dict
        Validation results
    """
    print(f"\n📋 Validating file ordering consistency")
    
    validation_results = {
        'ordering_consistent': True,
        'issues': [],
        'file_order_from_individual': [],
        'file_order_from_metadata': [],
        'order_matches': True
    }
    
    # Get file order from individual files
    individual_order = []
    for run_info in individual_data['file_order']:
        run_string = f"ses-{run_info['session']}_task-{run_info['task']}_acq-{run_info['acquisition']}_run-{run_info['run']}"
        individual_order.append(run_string)
    
    validation_results['file_order_from_individual'] = individual_order
    
    print(f"File order from individual files:")
    for i, run_string in enumerate(individual_order):
        print(f"  {i+1:2d}. {run_string}")
    
    # Get file order from concatenated metadata
    if 'RunBreakdown' in concatenated_data['metadata']:
        metadata_order = []
        for run_info in concatenated_data['metadata']['RunBreakdown']:
            run_name = run_info['run']
            # Extract run components from full name
            parts = run_name.split('_')
            run_components = []
            for part in parts[1:]:  # Skip 'sub-XX'
                if part.startswith(('ses-', 'task-', 'acq-', 'run-')):
                    run_components.append(part)
            run_string = '_'.join(run_components)
            metadata_order.append(run_string)
        
        validation_results['file_order_from_metadata'] = metadata_order
        
        print(f"\nFile order from concatenated metadata:")
        for i, run_string in enumerate(metadata_order):
            print(f"  {i+1:2d}. {run_string}")
        
        # Compare orders
        if individual_order == metadata_order:
            print(f"\n✅ File ordering is CONSISTENT between individual and concatenated data")
            validation_results['order_matches'] = True
        else:
            print(f"\n❌ File ordering MISMATCH detected!")
            validation_results['order_matches'] = False
            validation_results['ordering_consistent'] = False
            validation_results['issues'].append("File ordering mismatch between individual and concatenated data")
            
            # Show differences
            for i, (ind, meta) in enumerate(zip(individual_order, metadata_order)):
                if ind != meta:
                    print(f"  Position {i+1}: Individual='{ind}' vs Metadata='{meta}'")
    else:
        print(f"\n⚠️  RunBreakdown not found in concatenated metadata")
        validation_results['issues'].append("RunBreakdown missing from concatenated metadata")
    
    return validation_results


def validate_data_consistency(individual_data, concatenated_data):
    """
    Validate that the concatenated data matches the individual data.
    
    Parameters
    ----------
    individual_data : dict
        Individual files data
    concatenated_data : dict
        Concatenated files data
        
    Returns
    -------
    dict
        Validation results
    """
    print(f"\n🔢 Validating data consistency")
    
    validation_results = {
        'data_consistent': True,
        'shape_consistent': True,
        'content_identical': True,
        'issues': [],
        'individual_shapes': [],
        'concatenated_shape': None,
        'total_expected_timepoints': 0
    }
    
    # Calculate expected concatenated shape
    individual_shapes = [data.shape for data in individual_data['data']]
    validation_results['individual_shapes'] = individual_shapes
    
    total_timepoints = sum(shape[0] for shape in individual_shapes)
    n_features = individual_shapes[0][1] if individual_shapes else 0
    expected_shape = (total_timepoints, n_features)
    
    validation_results['total_expected_timepoints'] = total_timepoints
    validation_results['concatenated_shape'] = concatenated_data['data'].shape
    
    print(f"Individual file shapes:")
    for i, shape in enumerate(individual_shapes):
        print(f"  File {i+1}: {shape}")
    
    print(f"Expected concatenated shape: {expected_shape}")
    print(f"Actual concatenated shape: {concatenated_data['data'].shape}")
    
    # Check shape consistency
    if concatenated_data['data'].shape == expected_shape:
        print(f"✅ Shape consistency: PASSED")
    else:
        print(f"❌ Shape consistency: FAILED")
        validation_results['shape_consistent'] = False
        validation_results['data_consistent'] = False
        validation_results['issues'].append(f"Shape mismatch: expected {expected_shape}, got {concatenated_data['data'].shape}")
    
    # Check content consistency (if shapes match)
    if validation_results['shape_consistent']:
        print(f"\n🔍 Checking content consistency...")
        
        # Manually concatenate individual data and compare
        manual_concat = np.vstack(individual_data['data'])
        
        if np.array_equal(manual_concat, concatenated_data['data']):
            print(f"✅ Content consistency: PASSED - Data is identical")
        else:
            print(f"❌ Content consistency: FAILED - Data differs")
            validation_results['content_identical'] = False
            validation_results['data_consistent'] = False
            validation_results['issues'].append("Content mismatch between manual and stored concatenation")
            
            # Check for approximate equality
            if np.allclose(manual_concat, concatenated_data['data'], rtol=1e-10):
                print(f"   (Note: Data is approximately equal within numerical precision)")
            else:
                print(f"   (Note: Data has significant differences)")
                
                # Show some statistics about differences
                diff = np.abs(manual_concat - concatenated_data['data'])
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                print(f"   Max difference: {max_diff}")
                print(f"   Mean difference: {mean_diff}")
    
    return validation_results


def validate_indices_consistency(individual_data, concatenated_data):
    """
    Validate that the concatenated indices are correctly computed.
    
    Parameters
    ----------
    individual_data : dict
        Individual files data
    concatenated_data : dict
        Concatenated files data
        
    Returns
    -------
    dict
        Validation results
    """
    print(f"\n📊 Validating indices consistency")
    
    validation_results = {
        'indices_consistent': True,
        'clean_indices_consistent': True,
        'original_indices_consistent': True,
        'issues': [],
        'manual_clean_indices': None,
        'manual_original_indices': None
    }
    
    # Check clean indices
    if concatenated_data['clean_indices'] is not None:
        print(f"Validating clean indices...")
        
        # Manually compute what the concatenated clean indices should be
        manual_clean_indices = []
        cumulative_clean = 0
        
        for i, (data, clean_indices) in enumerate(zip(individual_data['data'], individual_data['clean_indices'])):
            if clean_indices is not None:
                adjusted_indices = clean_indices + cumulative_clean
                manual_clean_indices.append(adjusted_indices)
                cumulative_clean += data.shape[0]
                
                print(f"  File {i+1}: {clean_indices.shape[0]} segments, offset +{cumulative_clean - data.shape[0]}")
                print(f"    Original indices range: [{np.min(clean_indices)}, {np.max(clean_indices)}]")
                print(f"    Adjusted indices range: [{np.min(adjusted_indices)}, {np.max(adjusted_indices)}]")
        
        if manual_clean_indices:
            manual_clean_concat = np.vstack(manual_clean_indices)
            validation_results['manual_clean_indices'] = manual_clean_concat
            
            print(f"Manual concatenated clean indices shape: {manual_clean_concat.shape}")
            print(f"Stored concatenated clean indices shape: {concatenated_data['clean_indices'].shape}")
            
            if np.array_equal(manual_clean_concat, concatenated_data['clean_indices']):
                print(f"✅ Clean indices consistency: PASSED")
            else:
                print(f"❌ Clean indices consistency: FAILED")
                validation_results['clean_indices_consistent'] = False
                validation_results['indices_consistent'] = False
                validation_results['issues'].append("Clean indices mismatch")
                
                # Show differences
                if manual_clean_concat.shape == concatenated_data['clean_indices'].shape:
                    diff_mask = manual_clean_concat != concatenated_data['clean_indices']
                    n_diffs = np.sum(diff_mask)
                    print(f"   Number of differing elements: {n_diffs}")
                    if n_diffs < 20:  # Show first few differences
                        diff_indices = np.where(diff_mask)
                        for idx in zip(*diff_indices):
                            print(f"   Position {idx}: Manual={manual_clean_concat[idx]}, Stored={concatenated_data['clean_indices'][idx]}")
    else:
        print(f"⚠️  No concatenated clean indices to validate")
    
    # Check original indices
    if concatenated_data['original_indices'] is not None:
        print(f"\nValidating original indices...")
        
        # Manually compute what the concatenated original indices should be
        manual_original_indices = []
        cumulative_original = 0
        
        for i, (metadata, original_indices) in enumerate(zip(individual_data['metadata'], individual_data['original_indices'])):
            if original_indices is not None:
                adjusted_indices = original_indices + cumulative_original
                manual_original_indices.append(adjusted_indices)
                
                # Get original timepoints from metadata
                original_timepoints = metadata.get('OriginalTimePoints', individual_data['data'][i].shape[0])
                cumulative_original += original_timepoints
                
                print(f"  File {i+1}: {original_indices.shape[0]} segments, offset +{cumulative_original - original_timepoints}")
                print(f"    Original indices range: [{np.min(original_indices)}, {np.max(original_indices)}]")
                print(f"    Adjusted indices range: [{np.min(adjusted_indices)}, {np.max(adjusted_indices)}]")
        
        if manual_original_indices:
            manual_original_concat = np.vstack(manual_original_indices)
            validation_results['manual_original_indices'] = manual_original_concat
            
            print(f"Manual concatenated original indices shape: {manual_original_concat.shape}")
            print(f"Stored concatenated original indices shape: {concatenated_data['original_indices'].shape}")
            
            if np.array_equal(manual_original_concat, concatenated_data['original_indices']):
                print(f"✅ Original indices consistency: PASSED")
            else:
                print(f"❌ Original indices consistency: FAILED")
                validation_results['original_indices_consistent'] = False
                validation_results['indices_consistent'] = False
                validation_results['issues'].append("Original indices mismatch")
    else:
        print(f"⚠️  No concatenated original indices to validate")
    
    return validation_results


def validate_index_continuity(concatenated_data):
    """
    Validate that indices form continuous segments without gaps or overlaps.
    
    Parameters
    ----------
    concatenated_data : dict
        Concatenated files data
        
    Returns
    -------
    dict
        Validation results
    """
    print(f"\n🔗 Validating index continuity")
    
    validation_results = {
        'continuity_valid': True,
        'clean_continuity_valid': True,
        'original_continuity_valid': True,
        'issues': [],
        'gaps': [],
        'overlaps': []
    }
    
    # Check clean indices continuity
    if concatenated_data['clean_indices'] is not None:
        clean_indices = concatenated_data['clean_indices']
        print(f"Checking clean indices continuity ({len(clean_indices)} segments)...")
        
        gaps = []
        overlaps = []
        
        for i in range(len(clean_indices) - 1):
            current_end = clean_indices[i][1]
            next_start = clean_indices[i + 1][0]
            
            if current_end < next_start:
                gap_size = next_start - current_end
                gaps.append((i, i + 1, gap_size))
                print(f"  Gap between segments {i+1} and {i+2}: {gap_size} timepoints")
            elif current_end > next_start:
                overlap_size = current_end - next_start
                overlaps.append((i, i + 1, overlap_size))
                print(f"  Overlap between segments {i+1} and {i+2}: {overlap_size} timepoints")
            else:
                # Perfect continuity
                pass
        
        validation_results['gaps'] = gaps
        validation_results['overlaps'] = overlaps
        
        if gaps or overlaps:
            validation_results['clean_continuity_valid'] = False
            validation_results['continuity_valid'] = False
            validation_results['issues'].append(f"Clean indices have {len(gaps)} gaps and {len(overlaps)} overlaps")
        else:
            print(f"✅ Clean indices continuity: PASSED (perfect continuity)")
            
        # Check that indices span the expected range
        total_clean_timepoints = concatenated_data['data'].shape[0]
        if len(clean_indices) > 0:
            min_index = np.min(clean_indices)
            max_index = np.max(clean_indices)
            print(f"Clean indices range: [{min_index}, {max_index}]")
            print(f"Expected range: [0, {total_clean_timepoints}]")
            
            if min_index != 0:
                validation_results['issues'].append(f"Clean indices don't start at 0 (start at {min_index})")
                validation_results['clean_continuity_valid'] = False
                validation_results['continuity_valid'] = False
            
            if max_index != total_clean_timepoints:
                validation_results['issues'].append(f"Clean indices don't end at {total_clean_timepoints} (end at {max_index})")
                validation_results['clean_continuity_valid'] = False
                validation_results['continuity_valid'] = False
    
    return validation_results


def generate_validation_report(validation_results, subject, verbose=False):
    """
    Generate a comprehensive validation report.
    
    Parameters
    ----------
    validation_results : dict
        Dictionary of validation result dictionaries
    subject : str
        Subject ID
    verbose : bool
        Whether to include detailed information
    """
    print(f"\n{'='*80}")
    print(f"VALIDATION REPORT FOR SUBJECT {subject}")
    print(f"{'='*80}")
    
    all_passed = True
    total_issues = 0
    
    for result_name, results in validation_results.items():
        print(f"\n{result_name.upper().replace('_', ' ')}")
        print(f"{'-' * len(result_name)}")
        
        if isinstance(results, dict):
            # Check for overall success
            main_key = None
            for key in ['ordering_consistent', 'data_consistent', 'indices_consistent', 'continuity_valid']:
                if key in results:
                    main_key = key
                    break
            
            if main_key and results[main_key]:
                print(f"✅ PASSED")
            else:
                print(f"❌ FAILED")
                all_passed = False
            
            # Show issues
            if 'issues' in results and results['issues']:
                print(f"Issues found:")
                for issue in results['issues']:
                    print(f"  - {issue}")
                    total_issues += 1
            
            # Show detailed information if verbose
            if verbose:
                if result_name == 'file_ordering' and 'file_order_from_individual' in results:
                    print(f"File order (individual): {results['file_order_from_individual']}")
                    if 'file_order_from_metadata' in results:
                        print(f"File order (metadata): {results['file_order_from_metadata']}")
                
                elif result_name == 'data_consistency':
                    print(f"Individual shapes: {results.get('individual_shapes', [])}")
                    print(f"Concatenated shape: {results.get('concatenated_shape', 'N/A')}")
                    print(f"Expected timepoints: {results.get('total_expected_timepoints', 'N/A')}")
    
    # Overall summary
    print(f"\n{'='*80}")
    print(f"OVERALL VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    if all_passed and total_issues == 0:
        print(f"🎉 ALL VALIDATIONS PASSED!")
        print(f"✅ The concatenation process worked correctly")
        print(f"✅ File ordering is consistent")
        print(f"✅ Data integrity is maintained")
        print(f"✅ Indices are properly aligned")
        print(f"✅ No continuity issues detected")
    else:
        print(f"⚠️  VALIDATION ISSUES DETECTED")
        print(f"❌ Total issues found: {total_issues}")
        print(f"🔧 Review the issues above and re-run the physio processing if needed")
    
    return all_passed, total_issues


def validate_all_subs(trf_output_dir, verbose=False):
    """
    Valida los archivos concatenados entre participantes (all_subs).
    """
    print(f"\n{'='*60}")
    print("Validando archivos concatenados entre participantes (all_subs)")
    tfr_file = trf_output_dir / "all_subs_desc-physio_features.npz"
    idx_file = trf_output_dir / "idx_data_all_subs.npz"
    idx_orig_file = trf_output_dir / "idx_data_OLD_timepoints_all_subs.npz"
    columns_file = trf_output_dir / "all_subs_desc-physio_columns.tsv"
    if not tfr_file.exists():
        print(f"❌ Archivo de datos no encontrado: {tfr_file}")
        return 1
    if not idx_file.exists():
        print(f"❌ Archivo de índices no encontrado: {idx_file}")
        return 1
    data_arr = np.load(tfr_file)['arr_0']
    idx_arr = np.load(idx_file)['arr_0']
    print(f"Forma del archivo de datos: {data_arr.shape}")
    print(f"Forma del archivo de índices: {idx_arr.shape}")
    max_clean_idx = idx_arr.max()
    n_rows = data_arr.shape[0]
    print(f"Cantidad de filas en datos: {n_rows}")
    print(f"Max index en clean_indices: {max_clean_idx}")
    if verbose:
        print(f"Todas las filas de clean_indices:")
        for i, row in enumerate(idx_arr):
            print(f"{i:3d}: {row}")
    if max_clean_idx <= n_rows:
        print(f"✅ Relación válida: max_clean_idx <= n_rows")
    else:
        print(f"❌ max_clean_idx > n_rows (posible error de índices)")
        return 1
    # Validar índices originales si existe
    if idx_orig_file.exists():
        idx_orig_arr = np.load(idx_orig_file)['arr_0']
        print(f"Forma del archivo de índices originales: {idx_orig_arr.shape}")
        if verbose:
            print(f"Todas las filas de original_indices:")
            for i, row in enumerate(idx_orig_arr):
                print(f"{i:3d}: {row}")
    else:
        print(f"⚠️  No se encontró el archivo de índices originales: {idx_orig_file}")
    # Validar columnas si existe
    if columns_file.exists():
        print(f"✓ Archivo de columnas encontrado: {columns_file}")
    else:
        print(f"⚠️  No se encontró el archivo de columnas: {columns_file}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Validate physio data concatenation consistency",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--sub', '--subject',
        type=str,
        help='Subject ID (e.g., 14)'
    )
    parser.add_argument(
        '--all-subs',
        action='store_true',
        help='Validar archivos concatenados entre participantes (all_subs)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed validation information'
    )
    args = parser.parse_args()
    print("="*80)
    print("physio DATA CONCATENATION VALIDATION")
    print("="*80)
    trf_output_dir = repo_root / "data" / "derivatives" / "physio"
    if not trf_output_dir.exists():
        print(f"❌ physio output directory not found: {trf_output_dir}")
        return 1
    if args.all_subs:
        return validate_all_subs(trf_output_dir, verbose=args.verbose)
    elif args.sub:
        print(f"Subject: {args.sub}")
        print(f"Verbose output: {args.verbose}")
        concatenated_data = load_concatenated_files(args.sub, trf_output_dir)
        if concatenated_data is None:
            return 1
        n_timepoints = concatenated_data['data'].shape[0]
        if concatenated_data['clean_indices'] is not None:
            min_idx = np.min(concatenated_data['clean_indices'])
            max_idx = np.max(concatenated_data['clean_indices'])
            if args.verbose:
                print(f"Todos los clean_indices: {concatenated_data['clean_indices']}")
            if min_idx < 0 or max_idx > n_timepoints:
                print(f"❌ Índices limpios fuera de rango: [{min_idx}, {max_idx}] para {n_timepoints} timepoints")
            else:
                print(f"✅ Índices limpios dentro de rango: [{min_idx}, {max_idx}] para {n_timepoints} timepoints")
        if concatenated_data['original_indices'] is not None:
            min_idx = np.min(concatenated_data['original_indices'])
            max_idx = np.max(concatenated_data['original_indices'])
            if args.verbose:
                print(f"Todos los original_indices: {concatenated_data['original_indices']}")
        print(f"\n🎉 VALIDACIÓN PASADA PARA EL SUJETO {args.sub}")
        return 0
    else:
        print("❌ Debe especificar --sub o --all-subs")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 