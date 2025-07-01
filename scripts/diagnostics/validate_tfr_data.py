#!/usr/bin/env python
"""
TFR Data Validation and Diagnostic Script for Clean (Cropped) Data

This script validates clean TFR data saved in .npy files to ensure they are correct and suitable for analysis.
The TFR files now contain only clean data (bad segments excluded) with corresponding index mapping files.

Usage:
    python scripts/diagnostics/validate_tfr_data.py --subject 14 --session vr --task 01 --acquisition b --run 006
    python scripts/diagnostics/validate_tfr_data.py --subject 14 --all-files
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import argparse
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

# Find repository root
script_path = Path(__file__).resolve()
repo_root = script_path
while not (repo_root / ".git").exists() and repo_root != repo_root.parent:
    repo_root = repo_root.parent


def load_and_validate_tfr_file(subject, session, task, acquisition, run):
    """
    Load and validate a single clean TFR file with comprehensive diagnostics.
    """
    print(f"\n{'='*80}")
    print(f"VALIDATING CLEAN TFR DATA")
    print(f"Subject: {subject}, Session: {session}, Task: {task}, Acquisition: {acquisition}, Run: {run}")
    print(f"{'='*80}")
    
    # Define paths
    trf_dir = repo_root / "data" / "derivatives" / "trf"
    base_filename = f"sub-{subject}_ses-{session}_task-{task}_acq-{acquisition}_run-{run}"
    
    # File paths
    data_path = trf_dir / f"{base_filename}_desc-morlet_tfr.npy"
    columns_path = trf_dir / f"{base_filename}_desc-morlet_columns.tsv"
    metadata_path = trf_dir / f"{base_filename}_desc-morlet_tfr.json"
    clean_idx_path = trf_dir / f"idx_data_{base_filename}.npy"
    original_idx_path = trf_dir / f"idx_data_OLD_timepoints_{base_filename}.npy"
    
    validation_results = {
        'file_info': {
            'subject': subject, 'session': session, 'task': task,
            'acquisition': acquisition, 'run': run
        },
        'files_found': {},
        'data_integrity': {},
        'consistency_checks': {},
        'recommendations': []
    }
    
    # 1. Check file existence
    print(f"\n=== FILE EXISTENCE CHECK ===")
    files_to_check = {
        'data': data_path,
        'columns': columns_path,
        'metadata': metadata_path,
        'clean_idx': clean_idx_path,
        'original_idx': original_idx_path
    }
    
    for file_type, file_path in files_to_check.items():
        exists = file_path.exists()
        validation_results['files_found'][file_type] = exists
        status = "✓" if exists else "❌"
        print(f"{status} {file_type:12s}: {file_path.name}")
        
        if not exists and file_type == 'data':
            print(f"❌ CRITICAL: Clean TFR data file not found!")
            return validation_results
    
    # 2. Load clean data
    print(f"\n=== LOADING CLEAN DATA ===")
    try:
        clean_data = np.load(data_path)
        print(f"✓ Clean TFR data loaded: {clean_data.shape}")
        validation_results['data_shape'] = clean_data.shape
    except Exception as e:
        print(f"❌ Error loading clean TFR data: {e}")
        return validation_results
    
    # Load columns if available
    column_names = None
    if columns_path.exists():
        try:
            columns_df = pd.read_csv(columns_path, sep='\t')
            column_names = columns_df['column_name'].tolist()
            print(f"✓ Column names loaded: {len(column_names)} features")
        except Exception as e:
            print(f"⚠️  Error loading columns: {e}")
    
    # Load metadata if available
    metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"✓ Metadata loaded")
        except Exception as e:
            print(f"⚠️  Error loading metadata: {e}")
    
    # 3. Data integrity validation
    print(f"\n=== DATA INTEGRITY VALIDATION ===")
    
    # Check dimensions
    if len(clean_data.shape) != 2:
        print(f"❌ Invalid dimensions: {len(clean_data.shape)}D (expected 2D)")
        validation_results['data_integrity']['dimensions'] = 'FAIL'
        return validation_results
    else:
        print(f"✓ Data has correct 2D shape: (timepoints={clean_data.shape[0]}, features={clean_data.shape[1]})")
        validation_results['data_integrity']['dimensions'] = 'PASS'
    
    # Check for NaN/Inf values
    nan_count = np.sum(np.isnan(clean_data))
    inf_count = np.sum(np.isinf(clean_data))
    total_values = clean_data.size
    
    validation_results['data_integrity']['nan_count'] = nan_count
    validation_results['data_integrity']['inf_count'] = inf_count
    validation_results['data_integrity']['total_values'] = total_values
    
    print(f"NaN values: {nan_count:,} / {total_values:,} ({100*nan_count/total_values:.2f}%)")
    print(f"Inf values: {inf_count:,} / {total_values:,} ({100*inf_count/total_values:.2f}%)")
    
    if nan_count == total_values:
        print(f"❌ CRITICAL: ALL values are NaN!")
        validation_results['data_integrity']['nan_status'] = 'ALL_NAN'
        validation_results['recommendations'].append("Regenerate TFR data - all values are NaN")
        return validation_results
    elif nan_count > total_values * 0.5:
        print(f"❌ WARNING: >50% values are NaN")
        validation_results['data_integrity']['nan_status'] = 'HIGH_NAN'
        validation_results['recommendations'].append("High NaN percentage - check TFR parameters")
    elif nan_count > 0:
        print(f"⚠️  Some NaN values present ({100*nan_count/total_values:.2f}%)")
        validation_results['data_integrity']['nan_status'] = 'SOME_NAN'
    else:
        print(f"✓ No NaN values")
        validation_results['data_integrity']['nan_status'] = 'NO_NAN'
    
    # Calculate statistics on finite values
    finite_mask = np.isfinite(clean_data)
    if np.any(finite_mask):
        finite_data = clean_data[finite_mask]
        stats = {
            'mean': np.mean(finite_data),
            'std': np.std(finite_data),
            'min': np.min(finite_data),
            'max': np.max(finite_data),
            'median': np.median(finite_data)
        }
        
        validation_results['data_integrity']['statistics'] = stats
        
        print(f"\nData statistics (finite values only):")
        print(f"  Mean: {stats['mean']:.6e}")
        print(f"  Std:  {stats['std']:.6e}")
        print(f"  Min:  {stats['min']:.6e}")
        print(f"  Max:  {stats['max']:.6e}")
        print(f"  Median: {stats['median']:.6e}")
        
        # Data quality checks
        if stats['min'] < 0:
            print(f"⚠️  Negative values found in TFR power data")
            validation_results['recommendations'].append("TFR power should be non-negative")
        
        if stats['std'] == 0:
            print(f"❌ Zero standard deviation - data is constant")
            validation_results['data_integrity']['constant_data'] = True
        else:
            validation_results['data_integrity']['constant_data'] = False
        
        # Check for reasonable TFR power values
        if stats['max'] / stats['min'] < 2 and stats['std'] / stats['mean'] < 0.1:
            print(f"⚠️  Very low dynamic range - data may be over-normalized")
            validation_results['recommendations'].append("Low dynamic range - check preprocessing")
    
    # 4. Consistency checks
    print(f"\n=== CONSISTENCY CHECKS ===")
    
    # Column names consistency
    if column_names is not None:
        if len(column_names) == clean_data.shape[1]:
            print(f"✓ Column names match feature count: {len(column_names)}")
            validation_results['consistency_checks']['column_names'] = 'PASS'
        else:
            print(f"❌ Column count mismatch: {len(column_names)} names vs {clean_data.shape[1]} features")
            validation_results['consistency_checks']['column_names'] = 'FAIL'
            validation_results['recommendations'].append("Fix column names file")
    else:
        print(f"⚠️  No column names file found")
        validation_results['consistency_checks']['column_names'] = 'MISSING'
    
    # Metadata consistency
    if metadata:
        meta_shape = metadata.get('DataShape', [])
        if meta_shape and list(meta_shape) == list(clean_data.shape):
            print(f"✓ Metadata shape matches: {meta_shape}")
            validation_results['consistency_checks']['metadata_shape'] = 'PASS'
        elif meta_shape:
            print(f"❌ Metadata shape mismatch: {meta_shape} vs {clean_data.shape}")
            validation_results['consistency_checks']['metadata_shape'] = 'FAIL'
        
        # Check other metadata fields
        expected_timepoints = metadata.get('NumberOfTimePoints', None)
        if expected_timepoints == clean_data.shape[0]:
            print(f"✓ Metadata timepoints match: {expected_timepoints}")
        elif expected_timepoints is not None:
            print(f"⚠️  Metadata timepoints mismatch: {expected_timepoints} vs {clean_data.shape[0]}")
        
        sampling_freq = metadata.get('SamplingFrequency', None)
        if sampling_freq:
            print(f"Sampling frequency from metadata: {sampling_freq} Hz")
            calculated_duration = clean_data.shape[0] / sampling_freq
            print(f"Calculated clean data duration: {calculated_duration:.2f} seconds")
        
        # Check cleaning information
        if metadata.get('CleaningApplied'):
            original_timepoints = metadata.get('OriginalTimePoints', 0)
            clean_timepoints = metadata.get('CleanTimePoints', 0)
            data_retention = metadata.get('DataRetention', 0)
            n_segments = metadata.get('NumberOfCleanSegments', 0)
            
            print(f"Cleaning information from metadata:")
            print(f"  Original timepoints: {original_timepoints:,}")
            print(f"  Clean timepoints: {clean_timepoints:,}")
            print(f"  Data retention: {data_retention:.1%}")
            print(f"  Number of segments: {n_segments}")
    
    # 5. Parse channel and frequency information
    if column_names:
        print(f"\n=== CHANNEL AND FREQUENCY ANALYSIS ===")
        channels = []
        frequencies = []
        
        for col_name in column_names:
            if '_' in col_name:
                parts = col_name.split('_')
                channel = parts[0]
                freq_str = parts[1].replace('hz', '').replace('Hz', '')
                
                channels.append(channel)
                try:
                    freq = float(freq_str)
                    frequencies.append(freq)
                except ValueError:
                    print(f"⚠️  Could not parse frequency from: {freq_str}")
        
        unique_channels = sorted(list(set(channels)))
        unique_frequencies = sorted(list(set(frequencies)))
        
        print(f"Unique channels: {len(unique_channels)} ({unique_channels[:5]}{'...' if len(unique_channels) > 5 else ''})")
        print(f"Unique frequencies: {len(unique_frequencies)} ({unique_frequencies} Hz)")
        print(f"Expected features: {len(unique_channels)} × {len(unique_frequencies)} = {len(unique_channels) * len(unique_frequencies)}")
        
        validation_results['channel_freq_info'] = {
            'n_channels': len(unique_channels),
            'n_frequencies': len(unique_frequencies),
            'channels': unique_channels,
            'frequencies': unique_frequencies
        }
        
        if len(unique_channels) * len(unique_frequencies) == clean_data.shape[1]:
            print(f"✓ Feature count matches channel×frequency combinations")
        else:
            print(f"⚠️  Feature count mismatch")
    
    # 6. Load index files to analyze segmentation
    print(f"\n=== LOADING INDEX FILES ===")
    clean_segments_indices = None
    original_segments_indices = None
    
    if clean_idx_path.exists() and original_idx_path.exists():
        try:
            clean_segments_indices = np.load(clean_idx_path)
            original_segments_indices = np.load(original_idx_path)
            print(f"✓ Index files loaded successfully")
            print(f"  Clean indices shape: {clean_segments_indices.shape}")
            print(f"  Original indices shape: {original_segments_indices.shape}")
            
            # Validate index consistency
            if clean_segments_indices.shape[0] == original_segments_indices.shape[0]:
                n_segments = clean_segments_indices.shape[0]
                print(f"✓ Consistent number of segments: {n_segments}")
                
                # Show segment details
                total_clean_timepoints = 0
                for seg_idx in range(n_segments):
                    clean_start, clean_end = clean_segments_indices[seg_idx]
                    orig_start, orig_end = original_segments_indices[seg_idx]
                    
                    clean_duration = clean_end - clean_start
                    orig_duration = orig_end - orig_start
                    total_clean_timepoints += clean_duration
                    
                    print(f"  Segment {seg_idx+1:2d}: Clean [{clean_start:6d}-{clean_end:6d}] | Original [{orig_start:6d}-{orig_end:6d}] ({clean_duration:6d} pts)")
                
                print(f"Total timepoints from indices: {total_clean_timepoints:,}")
                if total_clean_timepoints == clean_data.shape[0]:
                    print(f"✓ Index total matches data shape")
                    validation_results['index_validation'] = 'PASS'
                else:
                    print(f"❌ Index total doesn't match data shape")
                    validation_results['index_validation'] = 'FAIL'
            else:
                print(f"❌ Inconsistent number of segments")
                validation_results['index_validation'] = 'FAIL'
                
        except Exception as e:
            print(f"⚠️  Error loading index files: {e}")
            validation_results['index_validation'] = 'ERROR'
    else:
        print(f"⚠️  Index files not found - skipping segmentation analysis")
        validation_results['index_validation'] = 'MISSING'
    
    # 7. Create diagnostic visualization
    print(f"\n=== CREATING DIAGNOSTIC PLOT ===")
    
    # Define plots directory
    plots_dir = repo_root / "data" / "derivatives" / "trf_diagnostics"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if np.any(finite_mask):
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Data distribution
            finite_data = clean_data[finite_mask]
            axes[0, 0].hist(finite_data.flatten(), bins=50, alpha=0.7, edgecolor='black')
            axes[0, 0].set_xlabel('TFR Power Values (Clean Data)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of Clean TFR Power Values')
            axes[0, 0].set_yscale('log')
            
            # Plot 2: Clean TFR overview
            print("  → Creating clean TFR overview plot...")
            
            n_time_clean = clean_data.shape[0]
            n_feat_clean = clean_data.shape[1]
            
            # Downsample for visualization if dataset is very large
            time_downsample_clean = max(1, n_time_clean // 5000)
            feat_downsample_clean = max(1, n_feat_clean // 100)
            
            time_indices_clean = np.arange(0, n_time_clean, time_downsample_clean)
            feat_indices_clean = np.arange(0, n_feat_clean, feat_downsample_clean)
            
            clean_data_subset = clean_data[np.ix_(time_indices_clean, feat_indices_clean)].copy()
            clean_data_subset[~np.isfinite(clean_data_subset)] = np.nan
            
            # Calculate robust range for clean data
            finite_clean_data = clean_data_subset[np.isfinite(clean_data_subset)]
            if len(finite_clean_data) > 0:
                vmin_clean = float(np.percentile(finite_clean_data, 1))
                vmax_clean = float(np.percentile(finite_clean_data, 99))
                vmin_clean = max(vmin_clean, float(np.min(finite_clean_data[finite_clean_data > 0]))) if np.any(finite_clean_data > 0) else 1e-15
                vmax_clean = max(vmax_clean, vmin_clean * 10)
                
                print(f"    Clean TFR Overview color scale: {vmin_clean:.2e} to {vmax_clean:.2e} (log scale)")
                norm_clean = LogNorm(vmin=vmin_clean, vmax=vmax_clean)
            else:
                norm_clean = None
            
            im = axes[0, 1].imshow(clean_data_subset.T, aspect='auto', cmap='viridis',
                                   extent=[0, n_time_clean, 0, len(feat_indices_clean)],
                                   norm=norm_clean)
            axes[0, 1].set_xlabel('Time Points (Clean Data)')
            axes[0, 1].set_ylabel('Features (downsampled)')
            axes[0, 1].set_title(f'Clean TFR Overview (Log Scale)\n({n_time_clean:,} timepoints, {n_feat_clean} features)')
            
            # Add segment boundaries if available
            if clean_segments_indices is not None and len(clean_segments_indices) > 0:
                for seg_idx, (start_clean, end_clean) in enumerate(clean_segments_indices):
                    if start_clean <= n_time_clean:
                        axes[0, 1].axvline(x=start_clean, color='white', alpha=0.8, linewidth=1, linestyle='--')
                        if end_clean <= n_time_clean:
                            axes[0, 1].axvline(x=end_clean, color='white', alpha=0.8, linewidth=1, linestyle='--')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[0, 1], shrink=0.8)
            cbar.set_label('Power (log scale)', rotation=270, labelpad=20)
            
            # Plot 3: Time series of mean power
            print("  → Creating mean power time series...")
            
            mean_power_clean = np.nanmean(clean_data, axis=1)
            time_points_clean = np.arange(len(mean_power_clean))
            
            axes[1, 0].plot(time_points_clean, mean_power_clean, alpha=0.7, color='blue', linewidth=0.5)
            axes[1, 0].set_xlabel('Time Points (Clean Data)')
            axes[1, 0].set_ylabel('Mean Power Across Features')
            axes[1, 0].set_title(f'Clean Temporal Evolution of Mean Power\n({len(mean_power_clean):,} timepoints)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add segment boundaries
            if clean_segments_indices is not None and len(clean_segments_indices) > 0:
                for seg_idx, (start_clean, end_clean) in enumerate(clean_segments_indices):
                    if start_clean <= len(mean_power_clean):
                        axes[1, 0].axvline(x=start_clean, color='red', alpha=0.6, linewidth=2, linestyle='--')
                        if end_clean <= len(mean_power_clean):
                            axes[1, 0].axvline(x=end_clean, color='red', alpha=0.6, linewidth=2, linestyle='--')
            
            # Plot 4: Mean power per feature
            print("  → Creating feature power analysis...")
            
            mean_power_feat_clean = np.nanmean(clean_data, axis=0)
            feature_indices_clean = np.arange(len(mean_power_feat_clean))
            
            axes[1, 1].plot(feature_indices_clean, mean_power_feat_clean, 'o-', alpha=0.7, markersize=2, linewidth=0.5)
            axes[1, 1].set_xlabel('Feature Index')
            axes[1, 1].set_ylabel('Mean Power Across Time (Clean)')
            axes[1, 1].set_title(f'Clean Feature Power Analysis\n({len(mean_power_feat_clean)} features)')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Highlight outliers in clean data
            if np.any(np.isfinite(mean_power_feat_clean)):
                mean_val_clean = np.nanmean(mean_power_feat_clean)
                std_val_clean = np.nanstd(mean_power_feat_clean)
                threshold_clean = mean_val_clean + 3 * std_val_clean
                outliers_clean = feature_indices_clean[mean_power_feat_clean > threshold_clean]
                
                if len(outliers_clean) > 0:
                    axes[1, 1].scatter(outliers_clean, mean_power_feat_clean[outliers_clean], 
                                      color='red', s=20, alpha=0.8, label=f'Outliers (>3σ): {len(outliers_clean)}')
                    axes[1, 1].legend(fontsize=8)
                    print(f"    Found {len(outliers_clean)} outlier features in clean data")
                
                # Add channel-frequency grouping if column names available
                if column_names:
                    n_channels = validation_results.get('channel_freq_info', {}).get('n_channels', 0)
                    n_frequencies = validation_results.get('channel_freq_info', {}).get('n_frequencies', 0)
                    
                    if n_channels > 0 and n_frequencies > 0:
                        # Add vertical lines to separate frequency bands
                        for freq_idx in range(1, n_frequencies):
                            x_pos = freq_idx * n_channels
                            if x_pos < len(mean_power_feat_clean):
                                axes[1, 1].axvline(x=x_pos, color='gray', alpha=0.5, linestyle='--', linewidth=1)
                        
                        # Add frequency labels at the top
                        frequencies = validation_results['channel_freq_info']['frequencies']
                        for freq_idx, freq in enumerate(frequencies):
                            x_center = freq_idx * n_channels + n_channels // 2
                            if x_center < len(mean_power_feat_clean):
                                axes[1, 1].text(x_center, axes[1, 1].get_ylim()[1] * 0.95, 
                                               f'{freq}Hz', ha='center', va='top', fontsize=8, 
                                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot
            plot_path = plots_dir / f"{base_filename}_clean_diagnostic.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Diagnostic plot saved: {plot_path}")
        else:
            print(f"⚠️  Cannot create plots - no finite data")
    
    except Exception as e:
        print(f"⚠️  Error creating plots: {e}")
        import traceback
        traceback.print_exc()
    
    # 8. Overall assessment
    print(f"\n=== OVERALL ASSESSMENT ===")
    
    # Determine overall status
    if validation_results['data_integrity'].get('nan_status') == 'ALL_NAN':
        overall_status = 'CRITICAL_FAIL'
        print(f"❌ CRITICAL FAILURE: Data is completely unusable")
    elif validation_results['data_integrity'].get('constant_data', False):
        overall_status = 'FAIL'
        print(f"❌ FAILURE: Data has critical issues")
    elif (validation_results['data_integrity'].get('nan_status') == 'HIGH_NAN' or 
          validation_results['consistency_checks'].get('column_names') == 'FAIL' or
          validation_results.get('index_validation') == 'FAIL'):
        overall_status = 'WARNING'
        print(f"⚠️  WARNING: Data has issues but may be usable with caution")
    elif validation_results.get('index_validation') == 'ERROR':
        overall_status = 'WARNING'
        print(f"⚠️  WARNING: Index validation failed")
    else:
        overall_status = 'PASS'
        print(f"✓ PASS: Clean data is valid for analysis")
    
    validation_results['overall_status'] = overall_status
    
    # Summary of available data
    print(f"\nData summary:")
    print(f"  📊 Clean TFR data: {clean_data.shape} ({clean_data.shape[0]:,} timepoints, {clean_data.shape[1]} features)")
    if metadata.get('CleaningApplied'):
        original_timepoints = metadata.get('OriginalTimePoints', 0)
        data_retention = metadata.get('DataRetention', 0)
        print(f"  📈 Data retention: {data_retention:.1%} (from {original_timepoints:,} original timepoints)")
        n_segments = metadata.get('NumberOfCleanSegments', 0)
        print(f"  🔗 Clean segments: {n_segments}")
    
    # Generated outputs
    print(f"\nGenerated outputs:")
    print(f"  📊 Diagnostic plot: {base_filename}_clean_diagnostic.png")
    
    # Recommendations
    if validation_results['recommendations']:
        print(f"\nRecommendations:")
        for i, rec in enumerate(validation_results['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    return validation_results


def main():
    parser = argparse.ArgumentParser(description="Validate clean TFR data files")
    parser.add_argument('--subject', type=str, required=True, help='Subject ID')
    parser.add_argument('--session', type=str, help='Session ID')
    parser.add_argument('--task', type=str, help='Task ID')
    parser.add_argument('--acquisition', type=str, help='Acquisition parameter')
    parser.add_argument('--run', type=str, help='Run ID')
    parser.add_argument('--all-files', action='store_true', 
                       help='Validate all TFR files for the subject')
    
    args = parser.parse_args()
    
    print("="*80)
    print("CLEAN TFR DATA VALIDATION TOOL")
    print("="*80)
    
    if args.all_files:
        # Find all TFR files for the subject
        trf_dir = repo_root / "data" / "derivatives" / "trf"
        pattern = f"sub-{args.subject}_*_desc-morlet_tfr.npy"
        files = glob.glob(str(trf_dir / pattern))
        
        if not files:
            print(f"❌ No TFR files found for subject {args.subject}")
            return 1
        
        print(f"Found {len(files)} TFR files for subject {args.subject}")
        
        # Validate each file
        all_results = []
        for filepath in sorted(files):
            filename = Path(filepath).name
            # Parse filename to get parameters
            parts = filename.replace('_desc-morlet_tfr.npy', '').split('_')
            
            params = {}
            for part in parts:
                if part.startswith('sub-'):
                    params['subject'] = part[4:]
                elif part.startswith('ses-'):
                    params['session'] = part[4:]
                elif part.startswith('task-'):
                    params['task'] = part[5:]
                elif part.startswith('acq-'):
                    params['acquisition'] = part[4:]
                elif part.startswith('run-'):
                    params['run'] = part[4:]
            
            if len(params) == 5:
                results = load_and_validate_tfr_file(**params)
                all_results.append(results)
        
        # Summary
        print(f"\n{'='*80}")
        print("VALIDATION SUMMARY")
        print(f"{'='*80}")
        
        status_counts = {}
        for result in all_results:
            status = result.get('overall_status', 'ERROR')
            status_counts[status] = status_counts.get(status, 0) + 1
            
            file_key = (f"sub-{result['file_info']['subject']}_"
                       f"ses-{result['file_info']['session']}_"
                       f"task-{result['file_info']['task']}_"
                       f"acq-{result['file_info']['acquisition']}_"
                       f"run-{result['file_info']['run']}")
            print(f"{status:12s}: {file_key}")
        
        print(f"\nStatus Summary:")
        for status, count in status_counts.items():
            print(f"  {status}: {count} files")
    
    else:
        # Validate single file
        if not all([args.session, args.task, args.acquisition, args.run]):
            print("❌ Please specify all parameters or use --all-files")
            return 1
        
        results = load_and_validate_tfr_file(
            args.subject, args.session, args.task, args.acquisition, args.run
        )
        
        # Return appropriate exit code
        status = results.get('overall_status', 'ERROR')
        if status in ['CRITICAL_FAIL', 'FAIL']:
            return 1
        elif status == 'WARNING':
            return 2
        else:
            return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
