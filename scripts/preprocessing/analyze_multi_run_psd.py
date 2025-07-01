#!/usr/bin/env python
# coding: utf-8

# # Multi-Run PSD Analysis for Campeones Analysis Project
# 
# This script loads all preprocessed EEG files for a single subject and creates
# a combined PSD analysis across all runs and trial types using merged_events.

#%%
# CONFIGURATION
subject = "14"  # Set subject number here

#%%
# IMPORTS AND SETUP
import os
import sys
from git import Repo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

import mne
from mne_bids import BIDSPath, read_raw_bids

# Add src directory to Python path
repo = Repo(os.getcwd(), search_parent_directories=True)
repo_root = repo.git.rev_parse("--show-toplevel")
src_path = os.path.join(repo_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

#%%
# FIND ALL PREPROCESSED FILES FOR SUBJECT
print(f"=== LOADING PREPROCESSED DATA FOR SUBJECT {subject} ===")

# Define paths
pipeline_name = "campeones_preproc"
derivatives_root = os.path.join(repo_root, "data", "derivatives")
derivatives_folder = os.path.join(derivatives_root, pipeline_name)
subject_folder = os.path.join(derivatives_folder, f"sub-{subject}")

# Also define merged_events path
merged_events_folder = os.path.join(derivatives_root, "merged_events", f"sub-{subject}", "ses-vr", "eeg")

# Find all preprocessed EEG files
preprocessed_files = []
if os.path.exists(subject_folder):
    for root, dirs, files in os.walk(subject_folder):
        for file in files:
            if file.endswith("_desc-preproc_eeg.vhdr"):
                preprocessed_files.append(os.path.join(root, file))

preprocessed_files.sort()  # Sort for consistent ordering
print(f"Found {len(preprocessed_files)} preprocessed files:")
for file in preprocessed_files:
    print(f"  - {os.path.basename(file)}")

if not preprocessed_files:
    raise FileNotFoundError(f"No preprocessed files found for subject {subject}")

#%%
# LOAD ALL RUNS AND COLLECT RAW DATA WITH MERGED EVENTS
print(f"\n=== LOADING {len(preprocessed_files)} RUNS WITH MERGED EVENTS ===")

all_runs_raw = []  # Store raw data from all runs
run_id_mapping = {}  # Map raw objects to run IDs
all_conditions = set()  # Collect all unique conditions
run_info = []  # Store metadata for each run

for i, file_path in enumerate(preprocessed_files):
    print(f"\nLoading run {i+1}/{len(preprocessed_files)}: {os.path.basename(file_path)}")
    
    # Parse BIDS filename to extract run info
    filename = os.path.basename(file_path)
    parts = filename.replace("_desc-preproc_eeg.vhdr", "").split("_")
    run_dict = {}
    for part in parts:
        if "-" in part:
            key, value = part.split("-", 1)
            run_dict[key] = value
    
    run_info.append({
        'run_id': f"task-{run_dict.get('task', 'unknown')}_run-{run_dict.get('run', 'unknown')}",
        'task': run_dict.get('task', 'unknown'),
        'run': run_dict.get('run', 'unknown'),
        'file': file_path
    })
    
    # Create BIDSPath and load using BIDS
    try:
        # Create BIDSPath from the file structure
        bids_path = BIDSPath(
            subject=subject,
            session='vr',
            task=run_dict.get('task', 'unknown'),
            acquisition='b',
            run=run_dict.get('run', 'unknown'),
            suffix='eeg',
            extension='.vhdr',
            datatype='eeg',
            root=derivatives_folder,
            description='preproc'
        )
        
        # Load the preprocessed raw data using BIDS
        raw = read_raw_bids(bids_path, verbose=False)
        raw.load_data()
        
        print(f"  ✓ Loaded: {raw.info['nchan']} channels, {raw.times[-1]:.1f}s duration")
        
        # Load corresponding merged_events file
        merged_events_file = os.path.join(
            merged_events_folder,
            f"sub-{subject}_ses-vr_task-{run_dict.get('task', 'unknown')}_acq-b_run-{run_dict.get('run', 'unknown')}_desc-merged_events.tsv"
        )
        
        if os.path.exists(merged_events_file):
            print(f"  Loading merged events: {os.path.basename(merged_events_file)}")
            
            # Read merged events TSV
            events_df = pd.read_csv(merged_events_file, sep='\t')
            print(f"  Found {len(events_df)} events")
            
            # Get unique conditions
            conditions = set(events_df['condition'].unique())
            all_conditions.update(conditions)
            print(f"  Conditions: {list(conditions)}")
            
            # Create MNE annotations from merged events using condition column
            onsets = events_df['onset'].values
            durations = events_df['duration'].values
            descriptions = events_df['condition'].values
            
            # Create annotations object
            annotations = mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions
            )
            
            # Set annotations to raw data
            raw.set_annotations(annotations)
            print(f"  ✓ Set {len(annotations)} annotations from merged events")
            
            # Add to runs list
            all_runs_raw.append(raw)
            run_id_mapping[id(raw)] = run_info[-1]['run_id']
            
        else:
            print(f"  ⚠ Merged events file not found: {merged_events_file}")
            
    except Exception as e:
        print(f"  ⚠ Error loading with BIDS: {str(e)}")
        print(f"  Falling back to direct BrainVision loading...")
        
        # Fallback to direct loading
        raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
        
        # Try to load merged_events file even for fallback
        merged_events_file = os.path.join(
            merged_events_folder,
            f"sub-{subject}_ses-vr_task-{run_dict.get('task', 'unknown')}_acq-b_run-{run_dict.get('run', 'unknown')}_desc-merged_events.tsv"
        )
        
        if os.path.exists(merged_events_file):
            print(f"  Loading merged events: {os.path.basename(merged_events_file)}")
            events_df = pd.read_csv(merged_events_file, sep='\t')
            
            conditions = set(events_df['condition'].unique())
            all_conditions.update(conditions)
            print(f"  Conditions: {list(conditions)}")
            
            # Create annotations
            onsets = events_df['onset'].values
            durations = events_df['duration'].values
            descriptions = events_df['condition'].values
            
            annotations = mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions
            )
            
            raw.set_annotations(annotations)
            all_runs_raw.append(raw)
            run_id_mapping[id(raw)] = run_info[-1]['run_id']

print(f"\n✓ Loaded {len(all_runs_raw)} runs successfully")
print(f"All conditions found: {sorted(all_conditions)}")

#%%
# COMPUTE COMBINED PSD FOR EACH CONDITION ACROSS RUNS
print(f"\n=== COMPUTING COMBINED PSD ACROSS RUNS ===")

# Prepare the plot
fig, ax = plt.subplots(figsize=(14, 10))

# Define colors for different conditions
conditions = sorted(all_conditions)
colors = cm.get_cmap('Set1')(np.linspace(0, 1, len(conditions)))
color_map = dict(zip(conditions, colors))

# Summary statistics for logging
summary_stats = {}

for condition in conditions:
    print(f"\nProcessing {condition} across all runs:")
    
    # Collect PSD data from all runs for this condition
    all_psd_data = []
    all_durations = []
    runs_with_data = 0
    
    for run_idx, raw in enumerate(all_runs_raw):
        run_id = run_id_mapping.get(id(raw), f'run_{run_idx}')
        
        # Check if this condition exists in this run
        run_conditions = set(raw.annotations.description)
        if condition not in run_conditions:
            print(f"  Run {run_id}: No {condition} annotations")
            continue
            
        try:
            # Create a copy of raw data for this condition
            raw_ann = raw.copy()
            
            # Strategy: Mark all OTHER conditions as 'bad' to isolate current condition
            ann_onsets = []
            ann_durations = []
            ann_descriptions = []
            
            for ann in raw.annotations:
                if ann['description'] == condition:
                    # Keep this condition as is
                    ann_onsets.append(ann['onset'])
                    ann_durations.append(ann['duration'])
                    ann_descriptions.append(ann['description'])
                else:
                    # Mark all other conditions as 'bad' to exclude them
                    ann_onsets.append(ann['onset'])
                    ann_durations.append(ann['duration'])
                    ann_descriptions.append('bad')
            
            if ann_onsets:
                # Set the filtered annotations where only current condition is preserved
                filtered_annotations = mne.Annotations(
                    onset=ann_onsets,
                    duration=ann_durations,
                    description=ann_descriptions
                )
                raw_ann.set_annotations(filtered_annotations)
                
                # Calculate duration for this condition in this run
                run_duration = sum([ann['duration'] for ann in raw.annotations 
                                  if ann['description'] == condition])
                
                print(f"  Run {run_id}: {run_duration:.1f}s of {condition}")
                
                # Compute PSD for this condition (will automatically exclude 'bad')
                psd_ann = raw_ann.compute_psd(
                    fmax=45,
                    picks='eeg',
                    reject_by_annotation=['bad']  # This will exclude all non-current conditions
                )
                
                # Get PSD data (channels x frequencies)
                psd_data_matrix = psd_ann.get_data()  # Shape: (n_channels, n_freqs)
                freqs = psd_ann.freqs
                
                # Convert to dB for better visualization
                psd_data_db = 10 * np.log10(psd_data_matrix)
                
                # Store PSD data from this run
                all_psd_data.append(psd_data_db)
                all_durations.append(run_duration)
                runs_with_data += 1
                
        except Exception as e:
            print(f"  Run {run_id}: Error processing {condition}: {str(e)}")
    
    # Combine PSD data across all runs if we have data
    if all_psd_data:
        # Concatenate all PSD data: shape becomes (total_channels_across_runs, n_freqs)
        combined_psd = np.vstack(all_psd_data)
        
        # Calculate statistics across all channels from all runs
        psd_mean = np.mean(combined_psd, axis=0)
        psd_std = np.std(combined_psd, axis=0)
        
        # Plot mean with solid line
        color = color_map[condition]
        total_duration = sum(all_durations)
        
        label = f'{condition} ({runs_with_data} runs, {total_duration:.1f}s)'
        
        ax.plot(freqs, psd_mean, 
               color=color, 
               linewidth=2.5, 
               label=label, 
               alpha=0.9)
        
        # Plot std as shaded area
        ax.fill_between(freqs, 
                       psd_mean - psd_std, 
                       psd_mean + psd_std, 
                       color=color, 
                       alpha=0.2)
        
        # Store summary statistics
        summary_stats[condition] = {
            'runs_with_data': runs_with_data,
            'total_duration': total_duration,
            'mean_duration_per_run': total_duration / runs_with_data if runs_with_data > 0 else 0,
            'total_channels': combined_psd.shape[0],
            'freq_range': f'{freqs[0]:.1f}-{freqs[-1]:.1f} Hz'
        }
        
        print(f"  ✓ Combined: {runs_with_data} runs, {total_duration:.1f}s total, {combined_psd.shape[0]} channels")
    
    else:
        print(f"  ⚠ No data found for {condition} across any runs")

# Customize the plot
ax.set_xlabel('Frequency (Hz)', fontsize=12)
ax.set_ylabel('Power Spectral Density (dB)', fontsize=12)
ax.set_title(f'Multi-Run PSD Analysis: Subject {subject}\n'
             f'Mean ± SD across all EEG channels and runs by condition', 
             fontsize=14, fontweight='bold')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 45)  # Focus on relevant EEG frequencies

# Add summary information
info_text = f"Subject {subject} - {len(all_runs_raw)} runs analyzed\n"
info_text += f"Conditions: {len(conditions)} types\n"
info_text += "Solid line: Mean, Shaded: ± 1 SD"

ax.text(0.02, 0.98, info_text, 
        transform=ax.transAxes, 
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontsize=10)

plt.tight_layout()
plt.show()

#%%
# SUMMARY STATISTICS
print(f"\n=== SUMMARY STATISTICS FOR SUBJECT {subject} ===")
print(f"Total runs analyzed: {len(all_runs_raw)}")
print(f"Conditions found: {len(conditions)}")
print("\nDetailed breakdown by condition:")

for condition, stats in summary_stats.items():
    print(f"\n{condition}:")
    print(f"  - Runs with data: {stats['runs_with_data']}")
    print(f"  - Total duration: {stats['total_duration']:.1f}s")
    print(f"  - Mean duration per run: {stats['mean_duration_per_run']:.1f}s")
    print(f"  - Total channels analyzed: {stats['total_channels']}")
    print(f"  - Frequency range: {stats['freq_range']}")

# Calculate overall statistics
total_duration_all = sum([stats['total_duration'] for stats in summary_stats.values()])
total_channels_all = sum([stats['total_channels'] for stats in summary_stats.values()])

print(f"\nOverall statistics:")
print(f"  - Total recording time analyzed: {total_duration_all:.1f}s ({total_duration_all/60:.1f} min)")
print(f"  - Total channel-time analyzed: {total_channels_all} channel instances")
print(f"  - Runs loaded: {len(all_runs_raw)}")

# Show run details
print(f"\nRun details:")
for run in run_info:
    print(f"  - {run['run_id']}: {os.path.basename(run['file'])}")

print("=== ANALYSIS COMPLETED ===") 

#%%