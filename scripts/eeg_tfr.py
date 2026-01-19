#!/usr/bin/env python
"""

EEG Time-Frequency Representation (TFR) Analysis Script with Event-Based Cleaning and Multi-Subject Support

This script computes time-frequency decomposition using Morlet wavelets for preprocessed EEG data
from the Campeones Analysis project. It automatically processes all available sessions, tasks, and runs
for specified subject(s), and saves the resulting TFR data in a clean format suitable for analysis.
Supports processing multiple subjects simultaneously and combining all subjects into unified files.

Features:
- Loads preprocessed EEG data from derivatives/campeones_preproc
- Automatically discovers all available sessions/tasks/runs for subject(s)
- Computes TFR using Morlet wavelets at specific frequency bands
- Loads event annotations from preprocessing to identify 'bad' segments
- Creates clean TFR data by excluding 'bad' segments and concatenating good segments
- Generates index mapping files to track timepoint correspondence
- Reshapes data to (n_times, n_channels*n_freqs) format
- Supports processing multiple subjects simultaneously
- Optional cross-subject concatenation into unified all_subs_* files
- Maintains consistent ordering between data and indices across all levels of concatenation
- Saves clean results as .npz files with comprehensive metadata and column descriptions

Output Files (per subject):
- sub-{subject}_desc-morlet_tfr_concatenated.npz : Clean TFR data (bad segments excluded)
- sub-{subject}_desc-morlet_columns_concatenated.tsv : Column names and order
- sub-{subject}_desc-morlet_tfr_concatenated.json : Comprehensive metadata including segment mapping
- idx_data_sub-{subject}_concatenated.npz: Clean data indices (segment boundaries in clean timepoints)
- idx_data_OLD_timepoints_sub-{subject}_concatenated.npz : Original data indices (segment boundaries in original timepoints)

Output Files (when --combine-all-subs is used):
- all_subs_desc-morlet_tfr.npz : Clean TFR data from all subjects (bad segments excluded)
- all_subs_desc-morlet_columns.tsv : Column names and order
- all_subs_desc-morlet_tfr.json : Comprehensive metadata including segment mapping
- idx_data_all_subs.npz: Clean data indices (segment boundaries in clean timepoints)
- idx_data_OLD_timepoints_all_subs.npz : Original data indices (segment boundaries in original timepoints)

Usage:
    # Single subject
    python scripts/eeg_tfr.py --sub 14
    python scripts/eeg_tfr.py --sub 14 --freqs 2 4 8 16 32 --cycles 4
    
    # Multiple subjects
    python scripts/eeg_tfr.py --sub 14 16 17 18
    python scripts/eeg_tfr.py --sub 14 16 17 18 --freqs 2 4 8 16 32 --cycles 4
    
    # Multiple subjects with cross-subject concatenation
    python scripts/eeg_tfr.py --sub 14 16 17 18 --combine-all-subs

"""

import os
import sys
import numpy as np
import pandas as pd
import json
import argparse
import glob
from pathlib import Path
from git import Repo
import mne
from mne_bids import BIDSPath, read_raw_bids
import csv

# Find the repository root
script_path = Path(__file__).resolve()
repo_root = script_path
while not (repo_root / ".git").exists() and repo_root != repo_root.parent:
    repo_root = repo_root.parent


def load_events_from_eeg(subject, session, task, acquisition, run):
    """
    Load events from the preprocessed EEG events file.
    
    Parameters
    ----------
    subject : str
        Subject ID
    session : str
        Session ID
    task : str
        Task ID
    acquisition : str
        Acquisition parameter
    run : str
        Run ID
        
    Returns
    -------
    events_df : pd.DataFrame
        DataFrame with events information
    """
    # Define path to events file
    preproc_dir = repo_root / "data" / "derivatives" / "campeones_preproc"
    events_path = (preproc_dir / f"sub-{subject}" / f"ses-{session}" / "eeg" / 
                  f"sub-{subject}_ses-{session}_task-{task}_acq-{acquisition}_run-{run}_desc-preproc_events.tsv")
    
    if not events_path.exists():
        print(f"⚠️  Events file not found: {events_path}")
        return None
    
    try:
        events_df = pd.read_csv(events_path, sep='\t')
        print(f"✓ Events loaded: {len(events_df)} events from {events_path.name}")
        
        # Show event types summary
        event_types = events_df['trial_type'].value_counts()
        print(f"  Event types: {dict(event_types)}")
        
        # Show timing summary
        total_duration = events_df['duration'].sum()
        bad_duration = events_df[events_df['trial_type'] == 'bad']['duration'].sum() if 'bad' in event_types else 0
        print(f"  Total annotated duration: {total_duration:.2f} seconds")
        if bad_duration > 0:
            print(f"  Bad segments duration: {bad_duration:.2f} seconds ({100*bad_duration/total_duration:.1f}%)")
        
        return events_df
        
    except Exception as e:
        print(f"⚠️  Error loading events: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_clean_tfr_data_and_indices(reshaped_data, events_df, sfreq, subject, session, task, acquisition, run):
    """
    Create clean TFR data by removing 'bad' segments and generate index mappings.
    
    Parameters
    ----------
    reshaped_data : np.ndarray
        Original TFR data (n_timepoints, n_features)
    events_df : pd.DataFrame
        Events dataframe with onset, duration, trial_type
    sfreq : float
        Sampling frequency in Hz
    subject : str
        Subject ID
    session : str
        Session ID
    task : str
        Task ID
    acquisition : str
        Acquisition parameter
    run : str
        Run ID
        
    Returns
    -------
    clean_data : np.ndarray
        Concatenated clean data without 'bad' segments
    clean_segments_indices : np.ndarray
        [N_clean_segments, 2] - start/end indices in clean data
    original_segments_indices : np.ndarray
        [N_clean_segments, 2] - start/end indices in original data
    """
    print(f"\n=== CREATING CLEAN TFR DATA (EXCLUDING 'BAD' SEGMENTS) ===")
    
    n_timepoints = reshaped_data.shape[0]
    
    # Define minimum segment length (2 seconds at current sampling rate)
    min_segment_length = int(2.0 * sfreq)  # 2 seconds minimum
    print(f"Minimum segment length: {min_segment_length} timepoints ({min_segment_length/sfreq:.1f} seconds)")
    
    # Create mask for bad segments
    bad_mask = np.zeros(n_timepoints, dtype=bool)
    
    # Mark bad segments
    bad_events = events_df[events_df['trial_type'] == 'bad']
    print(f"Found {len(bad_events)} 'bad' events to exclude")
    
    for idx, event in bad_events.iterrows():
        start_tp = int(event['onset'] * sfreq)
        end_tp = int((event['onset'] + event['duration']) * sfreq)
        
        # Ensure indices are within bounds
        start_tp = max(0, start_tp)
        end_tp = min(n_timepoints, end_tp)
        
        bad_mask[start_tp:end_tp] = True
        print(f"  Excluding bad segment: {start_tp}-{end_tp} ({end_tp-start_tp} timepoints)")
    
    # Find clean segments (continuous ranges of good data)
    good_mask = ~bad_mask
    
    # Find transitions in the good mask
    diff_mask = np.diff(np.concatenate(([False], good_mask, [False])).astype(int))
    starts = np.where(diff_mask == 1)[0]  # Start of good segments
    ends = np.where(diff_mask == -1)[0]   # End of good segments
    
    print(f"Found {len(starts)} potential clean segments (before filtering)")
    
    # Filter segments that are too short
    valid_segments = []
    filtered_segments = []
    
    for start_orig, end_orig in zip(starts, ends):
        segment_length = end_orig - start_orig
        if segment_length >= min_segment_length:
            valid_segments.append((start_orig, end_orig))
        else:
            filtered_segments.append((start_orig, end_orig, segment_length))
    
    # Report filtering results
    if filtered_segments:
        print(f"Filtered out {len(filtered_segments)} segments shorter than {min_segment_length} timepoints:")
        for start_orig, end_orig, length in filtered_segments:
            print(f"  Filtered segment: [{start_orig:6d}-{end_orig:6d}] ({length:6d} pts = {length/sfreq:.2f}s)")
    
    print(f"Keeping {len(valid_segments)} segments with sufficient length")
    
    # Calculate total clean timepoints (only from valid segments)
    total_clean_timepoints = sum(end - start for start, end in valid_segments)
    print(f"Total clean timepoints: {total_clean_timepoints:,} / {n_timepoints:,} ({100*total_clean_timepoints/n_timepoints:.1f}%)")
    
    # Extract clean data segments and build indices
    clean_segments = []
    clean_segments_indices = []
    original_segments_indices = []
    
    current_clean_idx = 0
    
    for i, (start_orig, end_orig) in enumerate(valid_segments):
        # Extract segment from original data
        segment = reshaped_data[start_orig:end_orig]
        clean_segments.append(segment)
        
        # Record indices in clean data
        segment_length = end_orig - start_orig
        clean_start = current_clean_idx
        clean_end = current_clean_idx + segment_length
        
        clean_segments_indices.append([clean_start, clean_end])
        original_segments_indices.append([start_orig, end_orig])
        
        print(f"  Segment {i+1}: Original [{start_orig:6d}-{end_orig:6d}] -> Clean [{clean_start:6d}-{clean_end:6d}] ({segment_length:6d} pts = {segment_length/sfreq:.1f}s)")
        
        current_clean_idx += segment_length
    
    # Concatenate all clean segments
    if clean_segments:
        clean_data = np.vstack(clean_segments)
    else:
        print("⚠️  No clean segments found!")
        clean_data = np.empty((0, reshaped_data.shape[1]))
    
    # Convert to numpy arrays
    clean_segments_indices = np.array(clean_segments_indices)
    original_segments_indices = np.array(original_segments_indices)
    
    print(f"✓ Clean TFR data created: {clean_data.shape}")
    print(f"✓ Index arrays shape: {clean_segments_indices.shape}")
    
    return clean_data, clean_segments_indices, original_segments_indices


def find_available_eeg_files(subject):
    """
    Find all available preprocessed EEG files for a given subject.
    
    Parameters
    ----------
    subject : str
        Subject ID (e.g., "14")
        
    Returns
    -------
    list of dict
        List of dictionaries containing session, task, acquisition, run information
    """
    print(f"\n=== DISCOVERING AVAILABLE EEG FILES FOR SUBJECT {subject} ===")
    
    # Define the preprocessed data directory
    preproc_dir = repo_root / "data" / "derivatives" / "campeones_preproc"
    
    # Search for all preprocessed EEG files for this subject
    pattern = f"sub-{subject}/*/eeg/sub-{subject}_*_desc-preproc_eeg.vhdr"
    search_path = preproc_dir / pattern
    
    # Use glob to find all matching files
    files = glob.glob(str(search_path))
    
    if not files:
        raise FileNotFoundError(f"No preprocessed EEG files found for subject {subject}")
    
    # Extract session, task, acquisition, run information from filenames
    file_info = []
    for filepath in sorted(files):
        filename = Path(filepath).name
        # Parse filename: sub-XX_ses-YY_task-ZZ_acq-AA_run-BBB_desc-preproc_eeg.vhdr
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
        
        # Verify we have all required information
        if all(key in info for key in ['session', 'task', 'acquisition', 'run']):
            info['subject'] = subject
            info['filepath'] = filepath
            file_info.append(info)
            print(f"  Found: ses-{info['session']} task-{info['task']} acq-{info['acquisition']} run-{info['run']}")
        else:
            print(f"  Warning: Could not parse filename {filename}")
    
    print(f"✓ Found {len(file_info)} EEG files for subject {subject}")
    
    # Group by session and task for summary
    sessions = set(info['session'] for info in file_info)
    tasks = set(info['task'] for info in file_info)
    acquisitions = set(info['acquisition'] for info in file_info)
    runs = set(info['run'] for info in file_info)
    
    print(f"  Sessions: {sorted(sessions)}")
    print(f"  Tasks: {sorted(tasks)}")
    print(f"  Acquisitions: {sorted(acquisitions)}")
    print(f"  Runs: {sorted(runs)}")
    
    # Sort file_info by acquisition, then task (as int), then run (as int)
    def sort_key(info):
        # Acquisition: sort alphabetically (a, b, ...)
        acq = info.get('acquisition', '')
        # Task: try to convert to int, fallback to string
        try:
            task = int(info.get('task', ''))
        except Exception:
            task = info.get('task', '')
        # Run: try to convert to int, fallback to string
        try:
            run = int(info.get('run', ''))
        except Exception:
            run = info.get('run', '')
        return (acq, task, run)
    file_info = sorted(file_info, key=sort_key)
    
    return file_info


def load_eeg_data(subject, session, task, acquisition, run, data="eeg"):
    """
    Load preprocessed EEG data using BIDS format.
    
    Parameters
    ----------
    subject : str
        Subject ID (e.g., "14")
    session : str
        Session ID (e.g., "vr")
    task : str
        Task ID (e.g., "01")
    acquisition : str
        Acquisition parameter (e.g., "a")
    run : str
        Run ID (e.g., "006")
    data : str, optional
        Data type (default: "eeg")
        
    Returns
    -------
    mne.io.Raw
        Loaded raw EEG data
    """
    print(f"\n=== LOADING EEG DATA ===")
    print(f"Subject: {subject}, Session: {session}, Task: {task}, Acquisition: {acquisition}, Run: {run}")
    
    # Define the file path components for Campeones Analysis project
    raw_data_folder = "data/derivatives/campeones_preproc"
    
    # Create a BIDSPath object pointing to preprocessed data
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        acquisition=acquisition,
        run=run,
        datatype=data,
        suffix=data,
        extension=".vhdr",  # Using .vhdr as the main file for BrainVision format
        root=os.path.join(repo_root, raw_data_folder),
        description="preproc"  # Adding description for preprocessed data
    )
    
    print(f"Loading data from: {bids_path.fpath}")
    
    # Check if file exists
    if not bids_path.fpath.exists():
        raise FileNotFoundError(f"Preprocessed EEG file not found: {bids_path.fpath}")
    
    # Load the data
    raw = read_raw_bids(bids_path, verbose=False)
    raw.load_data()
    
    print(f"✓ Datos cargados inicialmente: {raw.info['nchan']} canales, {raw.n_times} puntos de muestreo.")
    
    # Get channel types before filtering
    ch_types = raw.get_channel_types()
    unique_types = set(ch_types)
    print(f"✓ Tipos de canales encontrados: {unique_types}")
    
    # Count channels by type
    for ch_type in unique_types:
        count = ch_types.count(ch_type)
        print(f"  - {ch_type}: {count} canales")
    
    # Pick only EEG channels
    print("Filtrando solo canales EEG...")
    raw.pick_types(eeg=True)
    
    print(f"✓ Datos EEG filtrados: {raw.info['nchan']} canales EEG, {raw.n_times} puntos de muestreo.")
    print(f"✓ Frecuencia de muestreo: {raw.info['sfreq']} Hz")
    print(f"✓ Duración: {raw.times[-1]:.2f} segundos")
    print(f"✓ Canales EEG: {raw.ch_names}")
    
    return raw


def compute_tfr_morlet(raw, freqs=None, n_cycles=6):
    """
    Compute time-frequency representation using Morlet wavelets.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data (assumed to be filtered between 1-45 Hz from preprocessing)
    freqs : array-like, optional
        Frequencies for wavelet analysis (default: [2, 4, 8, 16, 32] Hz)
    n_cycles : int or array-like, optional
        Number of cycles per wavelet (default: 6)
        
    Returns
    -------
    power_tfr : mne.time_frequency.RawTFR
        Time-frequency representation object
    power_data : np.ndarray
        Power data array with shape (n_channels * n_freqs, n_times)
    """
    print(f"\n=== COMPUTING TIME-FREQUENCY REPRESENTATION ===")
    
    # Define default frequencies if not provided
    if freqs is None:
        freqs = np.array([2.0, 4.0, 8.0, 16.0, 32.0])  # Hz, logarithmically spaced by octaves
    
    print(f"Frequencies for analysis: {freqs} Hz")
    print(f"Number of cycles per wavelet: {n_cycles}")
    print("Note: Assuming data is already filtered between 1-45 Hz from preprocessing")
    
    # Compute time-frequency representation with Morlet wavelets
    print("Computing TFR using Morlet wavelets...")
    power_tfr = raw.compute_tfr(
        method='morlet', 
        freqs=freqs, 
        n_cycles=n_cycles,
        picks='eeg',
        reject_by_annotation=False,
        output='power',
        verbose=False
    )
    
    print(f"✓ TFR computed successfully")
    print(f"TFR object info: {power_tfr}")
    
    # Extract power data as NumPy array
    power_data = power_tfr.get_data()  # Shape: (n_channels, n_freqs, n_times)
    print(f"✓ Power data shape: {power_data.shape} (n_channels, n_freqs, n_times)")
    
    return power_tfr, power_data


def reshape_and_create_column_names(power_data, power_tfr):
    """
    Reshape power data and create corresponding column names.
    
    Parameters
    ----------
    power_data : np.ndarray
        Power data with shape (n_channels, n_freqs, n_times)
    power_tfr : mne.time_frequency.RawTFR
        TFR object containing channel and frequency information
        
    Returns
    -------
    reshaped_data : np.ndarray
        Reshaped data with shape (n_times, n_channels * n_freqs)
    column_names : list
        List of column names in format "channel_frequency" (e.g., "Fp1_2hz")
    """
    print(f"\n=== RESHAPING DATA AND CREATING COLUMN NAMES ===")
    
    # Get channel names and frequencies
    ch_names = power_tfr.ch_names
    freqs = power_tfr.freqs
    
    print(f"Channels: {len(ch_names)} ({ch_names[:5]}{'...' if len(ch_names) > 5 else ''})")
    print(f"Frequencies: {len(freqs)} ({freqs})")
    
    # Create column names by combining channel names with frequencies
    column_names = []
    for ch_name in ch_names:
        for freq in freqs:
            # Format frequency as integer if it's a whole number, otherwise with 1 decimal
            freq_str = f"{freq:.0f}hz" if freq == int(freq) else f"{freq:.1f}hz"
            column_names.append(f"{ch_name}_{freq_str}")
    
    print(f"✓ Created {len(column_names)} column names")
    print(f"Example column names: {column_names[:5]}{'...' if len(column_names) > 5 else ''}")
    
    # Reshape data from (n_channels, n_freqs, n_times) to (n_times, n_channels * n_freqs)
    # First, combine channel and frequency dimensions: (n_channels, n_freqs, n_times) -> (n_channels*n_freqs, n_times)
    n_channels, n_freqs, n_times = power_data.shape
    flattened_data = power_data.reshape(n_channels * n_freqs, n_times)
    
    # Then transpose to get (n_times, n_channels*n_freqs)
    reshaped_data = flattened_data.T
    
    print(f"✓ Data reshaped from {power_data.shape} to {reshaped_data.shape}")
    print(f"Final shape: (n_times={reshaped_data.shape[0]}, n_features={reshaped_data.shape[1]})")
    print(f"Expected features: {n_channels} channels × {n_freqs} frequencies = {n_channels * n_freqs}")
    
    # Verify that the number of columns matches the number of features
    assert len(column_names) == reshaped_data.shape[1], \
        f"Mismatch: {len(column_names)} column names vs {reshaped_data.shape[1]} features"
    
    return reshaped_data, column_names


def save_tfr_data(reshaped_data, column_names, subject, session, task, acquisition, run, sfreq, 
                  clean_data=None, clean_segments_indices=None, original_segments_indices=None, 
                  events_summary=None):
    """
    Save TFR data and column information to files.
    
    Parameters
    ----------
    reshaped_data : np.ndarray
        Original TFR data with shape (n_times, n_channels*n_freqs)
    column_names : list
        List of column names corresponding to the features
    subject : str
        Subject ID
    session : str
        Session ID  
    task : str
        Task ID
    acquisition : str
        Acquisition parameter
    run : str
        Run ID
    sfreq : float
        Sampling frequency of the original EEG data in Hz
    clean_data : np.ndarray, optional
        Clean TFR data with 'bad' segments removed
    clean_segments_indices : np.ndarray, optional
        [N_segments, 2] - start/end indices in clean data
    original_segments_indices : np.ndarray, optional
        [N_segments, 2] - start/end indices in original data
    events_summary : dict, optional
        Summary of events used for cleaning
        
    Returns
    -------
    data_path : Path
        Path to saved clean .npy file
    columns_path : Path
        Path to saved .tsv file with column names
    """
    print(f"\n=== SAVING TFR DATA ===")
    
    # Create output directory
    output_dir = repo_root / "data" / "derivatives" / "trf"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create BIDS-compliant filenames
    base_filename = f"sub-{subject}_ses-{session}_task-{task}_acq-{acquisition}_run-{run}"
    
    # Determine which data to save as main output (prefer clean data if available)
    if clean_data is not None and len(clean_data) > 0:
        main_data = clean_data
        data_description = "Clean time-frequency representation (bad segments excluded)"
        print(f"Saving clean TFR data as main output (excluding bad segments)")
    else:
        main_data = reshaped_data
        data_description = "Time-frequency representation (complete data)"
        print(f"Saving original TFR data as main output (no events available for cleaning)")
    
    # Save main TFR data as .npz file
    data_filename = f"{base_filename}_desc-morlet_tfr.npz"
    data_path = output_dir / data_filename
    np.savez(data_path, main_data)
    print(f"✓ Main TFR data saved: {data_path}")
    print(f"  Data shape: {main_data.shape}")
    print(f"  File size: {data_path.stat().st_size / (1024**2):.2f} MB")
    
    # Save column names as .tsv file
    columns_filename = f"{base_filename}_desc-morlet_columns.tsv"
    columns_path = output_dir / columns_filename
    
    # Create DataFrame with column names
    columns_df = pd.DataFrame({'column_name': column_names})
    columns_df.to_csv(columns_path, sep='\t', index=False)
    print(f"✓ Column names saved: {columns_path}")
    print(f"  Number of columns: {len(column_names)}")
    
    # Save index files if clean data was generated
    if clean_segments_indices is not None and original_segments_indices is not None:
        # Save clean data indices
        clean_idx_filename = f"idx_data_{base_filename}.npz"
        clean_idx_path = output_dir / clean_idx_filename
        np.savez(clean_idx_path, clean_segments_indices)
        print(f"✓ Clean data indices saved: {clean_idx_path}")
        print(f"  Index arrays shape: {clean_segments_indices.shape}")
        
        # Save original timepoints indices
        original_idx_filename = f"idx_data_OLD_timepoints_{base_filename}.npz"
        original_idx_path = output_dir / original_idx_filename
        np.savez(original_idx_path, original_segments_indices)
        print(f"✓ Original timepoints indices saved: {original_idx_path}")
    
    # Save metadata as JSON file for BIDS compliance
    metadata_filename = f"{base_filename}_desc-morlet_tfr.json"
    metadata_path = output_dir / metadata_filename
    
    # Extract frequency information from column names
    unique_freqs = sorted(list(set([name.split('_')[1] for name in column_names])))
    unique_channels = sorted(list(set([name.split('_')[0] for name in column_names])))
    
    metadata = {
        "Description": data_description,
        "Method": "Morlet wavelets",
        "Frequencies": unique_freqs,
        "Channels": unique_channels,
        "NumberOfChannels": len(unique_channels),
        "NumberOfFrequencies": len(unique_freqs),
        "NumberOfTimePoints": main_data.shape[0],
        "DataShape": list(main_data.shape),
        "Units": "Power (arbitrary units)",
        "SamplingFrequency": float(sfreq),
        "SamplingFrequencyUnit": "Hz",
        "ColumnOrder": column_names,  # Explicit column order
        "GeneratedBy": {
            "Name": "eeg_tfr.py",
            "Description": "Time-frequency analysis using MNE-Python Morlet wavelets",
            "Version": "2.0"
        }
    }
    
    # Add cleaning information if available
    if clean_data is not None:
        metadata["CleaningApplied"] = True
        metadata["OriginalTimePoints"] = reshaped_data.shape[0]
        metadata["CleanTimePoints"] = clean_data.shape[0]
        metadata["DataRetention"] = float(clean_data.shape[0] / reshaped_data.shape[0])
        metadata["NumberOfCleanSegments"] = len(clean_segments_indices) if clean_segments_indices is not None else 0
        
        if events_summary:
            metadata["EventsSummary"] = events_summary
            
        if clean_segments_indices is not None and original_segments_indices is not None:
            metadata["IndexFiles"] = {
                "CleanIndices": clean_idx_filename,
                "OriginalIndices": original_idx_filename,
                "Description": "Index files map between clean (cropped) and original timepoints"
            }
            
            # Add detailed segment information
            segments_info = []
            for i, (clean_seg, orig_seg) in enumerate(zip(clean_segments_indices, original_segments_indices)):
                segments_info.append({
                    "segment": i + 1,
                    "clean_start": int(clean_seg[0]),
                    "clean_end": int(clean_seg[1]),
                    "original_start": int(orig_seg[0]),
                    "original_end": int(orig_seg[1]),
                    "length": int(clean_seg[1] - clean_seg[0])
                })
            metadata["SegmentMapping"] = segments_info
    else:
        metadata["CleaningApplied"] = False
        metadata["Note"] = "No event-based cleaning applied - events file not available"
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"✓ Metadata saved: {metadata_path}")
    
    return data_path, columns_path


def process_single_file(file_info, freqs, n_cycles):
    """
    Process a single EEG file for TFR analysis with event-based cleaning.
    Returns data in memory without saving to disk.
    
    Parameters
    ----------
    file_info : dict
        Dictionary containing subject, session, task, acquisition, run information
    freqs : np.ndarray
        Frequencies for TFR analysis
    n_cycles : int
        Number of cycles per wavelet
        
    Returns
    -------
    dict or None
        Dictionary containing processed data and metadata, or None if processing failed
    """
    try:
        print(f"\n{'='*80}")
        print(f"PROCESSING: sub-{file_info['subject']} ses-{file_info['session']} "
              f"task-{file_info['task']} acq-{file_info['acquisition']} run-{file_info['run']}")
        print(f"{'='*80}")
        
        # Step 1: Load EEG data
        raw = load_eeg_data(
            file_info['subject'], file_info['session'], file_info['task'], 
            file_info['acquisition'], file_info['run']
        )
        
        # Step 2: Compute TFR using Morlet wavelets
        power_tfr, power_data = compute_tfr_morlet(raw, freqs, n_cycles)
        
        # Step 3: Reshape data and create column names
        reshaped_data, column_names = reshape_and_create_column_names(power_data, power_tfr)
        
        # Step 4: Load events for cleaning
        events_df = load_events_from_eeg(
            file_info['subject'], file_info['session'], file_info['task'], 
            file_info['acquisition'], file_info['run']
        )
        
        # Step 5: Create clean data and indices (if events are available)
        clean_data = None
        clean_segments_indices = None
        original_segments_indices = None
        events_summary = None
        
        if events_df is not None:
            # Create summary of events
            event_counts = events_df['trial_type'].value_counts().to_dict()
            events_summary = {
                "total_events": len(events_df),
                "event_types": event_counts,
                "bad_events_excluded": event_counts.get('bad', 0)
            }
            
            # Create clean data
            clean_data, clean_segments_indices, original_segments_indices = create_clean_tfr_data_and_indices(
                reshaped_data, events_df, raw.info['sfreq'], 
                file_info['subject'], file_info['session'], file_info['task'], 
                file_info['acquisition'], file_info['run']
            )
        else:
            print(f"⚠️  No events available - using original TFR data without cleaning")
            clean_data = reshaped_data
        
        print(f"\n✓ PROCESSING COMPLETED SUCCESSFULLY")
        
        if clean_data is not None and events_df is not None:
            print(f"  Clean data shape: {clean_data.shape}")
            print(f"  Original data shape: {reshaped_data.shape}")
            print(f"  Data retention: {100*clean_data.shape[0]/reshaped_data.shape[0]:.1f}%")
            print(f"  Clean segments: {len(clean_segments_indices) if clean_segments_indices is not None else 0}")
        else:
            print(f"  Final data shape: {clean_data.shape}")
        
        print(f"  Features (channels × frequencies): {clean_data.shape[1]}")
        print(f"  Column order maintained: {len(column_names)} features")
        
        # Return processed data in memory
        return {
            'file_info': file_info,
            'clean_data': clean_data,
            'original_data': reshaped_data,
            'column_names': column_names,
            'clean_segments_indices': clean_segments_indices,
            'original_segments_indices': original_segments_indices,
            'events_summary': events_summary,
            'sfreq': raw.info['sfreq'],
            'run_name': f"sub-{file_info['subject']}_ses-{file_info['session']}_task-{file_info['task']}_acq-{file_info['acquisition']}_run-{file_info['run']}"
        }
        
    except Exception as e:
        print(f"\n❌ ERROR processing sub-{file_info['subject']} ses-{file_info['session']} "
              f"task-{file_info['task']} acq-{file_info['acquisition']} run-{file_info['run']}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_and_concatenate_subject_data(subject, available_files, freqs, n_cycles):
    """
    Process all EEG files for a subject and directly save concatenated TFR data.
    
    This function processes all files in memory and creates concatenated outputs
    without saving individual files.
    
    Parameters
    ----------
    subject : str
        Subject ID (e.g., "14")
    available_files : list of dict
        List of file information dictionaries
    freqs : np.ndarray
        Frequencies for TFR analysis
    n_cycles : int
        Number of cycles per wavelet
        
    Returns
    -------
    bool
        True if processing was successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING AND CONCATENATING ALL DATA FOR SUBJECT {subject}")
    print(f"{'='*80}")
    
    try:
        # Lists to store processed data from all files
        all_clean_data = []
        all_clean_indices = []
        all_original_indices = []
        all_run_info = []
        filelog = []
        old_timepoints_seconds_from_start = []
        
        # Counters for index adjustment
        cumulative_clean_timepoints = 0
        cumulative_original_timepoints = 0
        
        # Column names (should be consistent across all files)
        column_names = None
        sfreq = None
        
        # Process each file
        successful_files = []
        failed_files = []
        
        for i, file_info in enumerate(available_files, 1):
            print(f"\n{'='*20} PROCESSING FILE {i}/{len(available_files)} {'='*20}")
            
            # Process single file in memory
            processed_data = process_single_file(file_info, freqs, n_cycles)
            
            if processed_data is None:
                failed_files.append(file_info)
                print(f"❌ Failed to process file {i}, skipping...")
                continue
            
            successful_files.append(file_info)
            
            # Extract data from processed result
            clean_data = processed_data['clean_data']
            original_data = processed_data['original_data']
            clean_segments_indices = processed_data['clean_segments_indices']
            original_segments_indices = processed_data['original_segments_indices']
            events_summary = processed_data['events_summary']
            
            # Store column names from first successful file
            if column_names is None:
                column_names = processed_data['column_names']
                sfreq = processed_data['sfreq']
                print(f"✓ Column names and sampling frequency stored from first file")
            
            # Verify column consistency
            if processed_data['column_names'] != column_names:
                print(f"⚠️  Warning: Column names differ for {processed_data['run_name']}")
            
            # Add clean data to collection
            all_clean_data.append(clean_data)
            print(f"✓ Added clean data: {clean_data.shape}")
            
            # Adjust indices for concatenation
            if clean_segments_indices is not None:
                # Adjust clean indices for concatenation
                adjusted_clean_indices = clean_segments_indices + cumulative_clean_timepoints
                all_clean_indices.append(adjusted_clean_indices)
                print(f"  Clean indices adjustment: +{cumulative_clean_timepoints}")
                print(f"  Clean indices shape: {clean_segments_indices.shape}")
            
            if original_segments_indices is not None:
                # Adjust original indices for concatenation
                adjusted_original_indices = original_segments_indices + cumulative_original_timepoints
                all_original_indices.append(adjusted_original_indices)
                print(f"  Original indices adjustment: +{cumulative_original_timepoints}")
                print(f"  Original indices shape: {original_segments_indices.shape}")
            
            # Create run information
            clean_points = clean_data.shape[0]
            original_points = original_data.shape[0]
            
            run_info = {
                "run": processed_data['run_name'],
                "clean_timepoints": clean_points,
                "original_timepoints": original_points,
                "clean_start_index": cumulative_clean_timepoints,
                "clean_end_index": cumulative_clean_timepoints + clean_points,
                "original_start_index": cumulative_original_timepoints,
                "original_end_index": cumulative_original_timepoints + original_points,
                "events_summary": events_summary
            }
            all_run_info.append(run_info)
            
            # Update cumulative counters
            cumulative_clean_timepoints += clean_points
            cumulative_original_timepoints += original_points
            
            print(f"  Cumulative clean timepoints: {cumulative_clean_timepoints:,}")
            print(f"  Cumulative original timepoints: {cumulative_original_timepoints:,}")
            
            filelog.append(file_info['filepath'])
            # Guardar los intervalos en segundos desde el inicio de cada archivo
            if original_segments_indices is not None:
                for seg in original_segments_indices:
                    start_tp, end_tp = seg
                    start_sec = start_tp / sfreq
                    end_sec = end_tp / sfreq
                    old_timepoints_seconds_from_start.append([start_sec, end_sec])
        
        # Check if we have any successful processing
        if not all_clean_data:
            print(f"❌ No files were successfully processed for subject {subject}")
            return False
        
        # Concatenate all data
        print(f"\n=== CONCATENATING DATA ===")
        concatenated_tfr = np.vstack(all_clean_data)
        print(f"Concatenated TFR data shape: {concatenated_tfr.shape}")
        print(f"Total files concatenated: {len(all_clean_data)}")
        
        # Concatenate indices if available
        concatenated_clean_indices = None
        concatenated_original_indices = None
        
        if all_clean_indices:
            concatenated_clean_indices = np.vstack(all_clean_indices)
            print(f"Concatenated clean indices shape: {concatenated_clean_indices.shape}")
        
        if all_original_indices:
            concatenated_original_indices = np.vstack(all_original_indices)
            print(f"Concatenated original indices shape: {concatenated_original_indices.shape}")
        
        # Create subject-specific output directory
        output_dir = repo_root / "data" / "derivatives" / "trf" / f"sub-{subject}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Subject output directory: {output_dir}")
        
        # Save concatenated TFR data
        concat_tfr_filename = f"sub-{subject}_desc-morlet_tfr_concatenated.npz"
        concat_tfr_path = output_dir / concat_tfr_filename
        np.savez(concat_tfr_path, concatenated_tfr)
        print(f"✓ Concatenated TFR data saved: {concat_tfr_path}")
        print(f"  File size: {concat_tfr_path.stat().st_size / (1024**2):.2f} MB")
        
        # Save concatenated clean indices
        if concatenated_clean_indices is not None:
            concat_clean_idx_filename = f"idx_data_sub-{subject}_concatenated.npz"
            concat_clean_idx_path = output_dir / concat_clean_idx_filename
            np.savez(concat_clean_idx_path, concatenated_clean_indices)
            print(f"✓ Concatenated clean indices saved: {concat_clean_idx_path}")
        
        # Save concatenated original indices
        if concatenated_original_indices is not None:
            concat_original_idx_filename = f"idx_data_OLD_timepoints_sub-{subject}_concatenated.npz"
            concat_original_idx_path = output_dir / concat_original_idx_filename
            np.savez(concat_original_idx_path, concatenated_original_indices)
            print(f"✓ Concatenated original indices saved: {concat_original_idx_path}")
        
        # Save column names
        if column_names is not None:
            concat_columns_filename = f"sub-{subject}_desc-morlet_columns_concatenated.tsv"
            concat_columns_path = output_dir / concat_columns_filename
            columns_df = pd.DataFrame({'column_name': column_names})
            columns_df.to_csv(concat_columns_path, sep='\t', index=False)
            print(f"✓ Column names saved: {concat_columns_path}")
        
        # Extract frequency information from column names
        unique_freqs = []
        unique_channels = []
        if column_names is not None:
            unique_freqs = sorted(list(set([name.split('_')[1] for name in column_names])))
            unique_channels = sorted(list(set([name.split('_')[0] for name in column_names])))
        
        # Create comprehensive metadata for concatenated data
        concat_metadata = {
            "Description": "Concatenated time-frequency representation for entire subject",
            "Subject": subject,
            "Method": "Morlet wavelets",
            "Frequencies": unique_freqs,
            "Channels": unique_channels,
            "NumberOfChannels": len(unique_channels),
            "NumberOfFrequencies": len(unique_freqs),
            "TotalTimePoints": concatenated_tfr.shape[0],
            "TotalOriginalTimePoints": cumulative_original_timepoints,
            "DataShape": list(concatenated_tfr.shape),
            "NumberOfRuns": len(all_run_info),
            "DataRetention": float(concatenated_tfr.shape[0] / cumulative_original_timepoints) if cumulative_original_timepoints > 0 else 1.0,
            "Units": "Power (arbitrary units)",
            "SamplingFrequency": float(sfreq) if sfreq else None,
            "SamplingFrequencyUnit": "Hz",
            "ColumnOrder": column_names,
            "RunBreakdown": all_run_info,
            "CleaningApplied": True,
            "GeneratedBy": {
                "Name": "eeg_tfr.py",
                "Description": "Direct concatenated TFR analysis using MNE-Python Morlet wavelets",
                "Version": "2.1"
            }
        }
        
        # Add indices information if available
        if concatenated_clean_indices is not None:
            concat_metadata["CleanSegments"] = concatenated_clean_indices.tolist()
            concat_metadata["NumberOfCleanSegments"] = len(concatenated_clean_indices)
            concat_metadata["IndexFiles"] = {
                "CleanIndices": concat_clean_idx_filename,
                "OriginalIndices": concat_original_idx_filename if concatenated_original_indices is not None else None,
                "Description": "Index files map between clean (cropped) and original timepoints"
            }
        
        # Save concatenated metadata
        concat_metadata_filename = f"sub-{subject}_desc-morlet_tfr_concatenated.json"
        concat_metadata_path = output_dir / concat_metadata_filename
        with open(concat_metadata_path, 'w') as f:
            json.dump(concat_metadata, f, indent=4)
        print(f"✓ Concatenated metadata saved: {concat_metadata_path}")
        
        # Guardar archivo de segundos desde inicio de cada archivo
        np.savez(output_dir / f"idx_data_OLD_timepoints_in_seconds_from_start_sub-{subject}_concatenated.npz",
                 np.array(old_timepoints_seconds_from_start))
        # Guardar filelog
        with open(output_dir / f"filelog_concatenate_sub-{subject}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename'])
            for fname in filelog:
                writer.writerow([fname])
        
        # Summary
        print(f"\n=== PROCESSING SUMMARY ===")
        print(f"Subject: {subject}")
        print(f"Files available: {len(available_files)}")
        print(f"Files successfully processed: {len(successful_files)}")
        print(f"Files failed: {len(failed_files)}")
        print(f"Total clean timepoints: {concatenated_tfr.shape[0]:,}")
        print(f"Total original timepoints: {cumulative_original_timepoints:,}")
        print(f"Overall data retention: {100*concatenated_tfr.shape[0]/cumulative_original_timepoints:.1f}%")
        print(f"Features per timepoint: {concatenated_tfr.shape[1]}")
        if concatenated_clean_indices is not None:
            print(f"Total clean segments: {len(concatenated_clean_indices)}")
        
        print(f"\nConcatenated files saved in: {output_dir}")
        print(f"  - {concat_tfr_filename}")
        if concatenated_clean_indices is not None:
            print(f"  - {concat_clean_idx_filename}")
        if concatenated_original_indices is not None:
            print(f"  - {concat_original_idx_filename}")
        print(f"  - {concat_columns_filename}")
        print(f"  - {concat_metadata_filename}")
        
        if failed_files:
            print(f"\n⚠️  Failed files:")
            for file_info in failed_files:
                print(f"  - sub-{file_info['subject']} ses-{file_info['session']} "
                      f"task-{file_info['task']} acq-{file_info['acquisition']} run-{file_info['run']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing and concatenating data for subject {subject}: {e}")
        import traceback
        traceback.print_exc()
        return False


def combine_all_subjects_data(successful_subjects, freqs, n_cycles):
    """
    Combine concatenated TFR data from all successful subjects into single files.
    
    This function loads the concatenated data from each subject and combines them
    into final all_subs_* files, maintaining consistent ordering between data and indices.
    
    Parameters
    ----------
    successful_subjects : list
        List of subject IDs that were successfully processed
    freqs : np.ndarray
        Frequencies used for TFR analysis (for metadata)
    n_cycles : int
        Number of cycles used (for metadata)
        
    Returns
    -------
    bool
        True if combination was successful, False otherwise
    """
    print(f"\n{'='*100}")
    print(f"COMBINING ALL SUBJECTS INTO FINAL CONCATENATED FILES")
    print(f"{'='*100}")
    
    if len(successful_subjects) < 2:
        print(f"❌ Need at least 2 subjects to combine, but only {len(successful_subjects)} successful")
        return False
    
    print(f"Subjects to combine (in order): {successful_subjects}")
    print(f"Total subjects: {len(successful_subjects)}")
    
    try:
        # Lists to store data from all subjects
        all_subjects_data = []
        all_subjects_clean_indices = []
        all_subjects_original_indices = []
        all_subjects_metadata = []
        all_filelogs = []
        all_old_timepoints_seconds_from_start = []
        
        # Counters for index adjustment across subjects
        cumulative_clean_timepoints = 0
        cumulative_original_timepoints = 0
        
        # Column names (should be consistent across subjects)
        column_names = None
        
        # Process each subject in order
        for subject_idx, subject in enumerate(successful_subjects):
            print(f"\n--- Loading data for subject {subject} ({subject_idx + 1}/{len(successful_subjects)}) ---")
            
            # Define paths for this subject's concatenated files
            subject_dir = repo_root / "data" / "derivatives" / "trf" / f"sub-{subject}"
            
            # Load TFR data
            tfr_file = subject_dir / f"sub-{subject}_desc-morlet_tfr_concatenated.npz"
            if not tfr_file.exists():
                print(f"❌ TFR file not found for subject {subject}: {tfr_file}")
                return False
            
            subject_tfr_data = np.load(tfr_file)['arr_0']
            all_subjects_data.append(subject_tfr_data)
            print(f"✓ Loaded TFR data: {subject_tfr_data.shape}")
            
            # Load metadata
            metadata_file = subject_dir / f"sub-{subject}_desc-morlet_tfr_concatenated.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    subject_metadata = json.load(f)
                all_subjects_metadata.append(subject_metadata)
                
                # Get original timepoints for this subject
                subject_original_timepoints = subject_metadata.get('TotalOriginalTimePoints', subject_tfr_data.shape[0])
                print(f"  Original timepoints: {subject_original_timepoints:,}")
                print(f"  Clean timepoints: {subject_tfr_data.shape[0]:,}")
                print(f"  Data retention: {subject_metadata.get('DataRetention', 'N/A')}")
            else:
                print(f"⚠️  Metadata file not found: {metadata_file}")
                subject_original_timepoints = subject_tfr_data.shape[0]  # Fallback
                all_subjects_metadata.append({})
            
            # Store column names from first subject
            if column_names is None and metadata_file.exists():
                column_names = subject_metadata.get('ColumnOrder', [])
                if column_names:
                    print(f"✓ Column names stored from subject {subject}: {len(column_names)} features")
                else:
                    # Fallback: load from .tsv file
                    columns_file = subject_dir / f"sub-{subject}_desc-morlet_columns_concatenated.tsv"
                    if columns_file.exists():
                        columns_df = pd.read_csv(columns_file, sep='\t')
                        column_names = columns_df['column_name'].tolist()
                        print(f"✓ Column names loaded from TSV: {len(column_names)} features")
            
            # Load clean data indices
            clean_idx_file = subject_dir / f"idx_data_sub-{subject}_concatenated.npz"
            if clean_idx_file.exists():
                subject_clean_indices = np.load(clean_idx_file)['arr_0']
                
                # Adjust clean indices for across-subjects concatenation
                adjusted_clean_indices = subject_clean_indices + cumulative_clean_timepoints
                all_subjects_clean_indices.append(adjusted_clean_indices)
                
                print(f"  Clean indices shape: {subject_clean_indices.shape}")
                print(f"  Clean indices adjustment: +{cumulative_clean_timepoints}")
                print(f"  Adjusted clean indices range: [{adjusted_clean_indices.min()}-{adjusted_clean_indices.max()}]")
            else:
                print(f"⚠️  Clean indices file not found: {clean_idx_file}")
            
            # Load original timepoints indices
            original_idx_file = subject_dir / f"idx_data_OLD_timepoints_sub-{subject}_concatenated.npz"
            if original_idx_file.exists():
                subject_original_indices = np.load(original_idx_file)['arr_0']
                
                # Adjust original indices for across-subjects concatenation
                adjusted_original_indices = subject_original_indices + cumulative_original_timepoints
                all_subjects_original_indices.append(adjusted_original_indices)
                
                print(f"  Original indices shape: {subject_original_indices.shape}")
                print(f"  Original indices adjustment: +{cumulative_original_timepoints}")
                print(f"  Adjusted original indices range: [{adjusted_original_indices.min()}-{adjusted_original_indices.max()}]")
            else:
                print(f"⚠️  Original indices file not found: {original_idx_file}")
            
            # Update cumulative counters
            cumulative_clean_timepoints += subject_tfr_data.shape[0]
            cumulative_original_timepoints += subject_original_timepoints
            
            print(f"  Cumulative clean timepoints: {cumulative_clean_timepoints:,}")
            print(f"  Cumulative original timepoints: {cumulative_original_timepoints:,}")
            
            # Cargar filelog y old_timepoints_seconds_from_start de cada sujeto
            filelog_path = subject_dir / f"filelog_concatenate_sub-{subject}.csv"
            old_seconds_path = subject_dir / f"idx_data_OLD_timepoints_in_seconds_from_start_sub-{subject}_concatenated.npz"
            if filelog_path.exists():
                with open(filelog_path, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)  # skip header
                    all_filelogs.extend([row[0] for row in reader])
            if old_seconds_path.exists():
                arr = np.load(old_seconds_path)['arr_0']
                all_old_timepoints_seconds_from_start.extend(arr)
        
        # Concatenate all subjects' data
        print(f"\n=== CONCATENATING ALL SUBJECTS ===")
        combined_tfr = np.vstack(all_subjects_data)
        print(f"Combined TFR data shape: {combined_tfr.shape}")
        print(f"Total subjects combined: {len(all_subjects_data)}")
        
        # Concatenate indices if available
        combined_clean_indices = None
        combined_original_indices = None
        
        if all_subjects_clean_indices:
            combined_clean_indices = np.vstack(all_subjects_clean_indices)
            print(f"Combined clean indices shape: {combined_clean_indices.shape}")
            print(f"Clean indices range: [{combined_clean_indices.min()}-{combined_clean_indices.max()}]")
        
        if all_subjects_original_indices:
            combined_original_indices = np.vstack(all_subjects_original_indices)
            print(f"Combined original indices shape: {combined_original_indices.shape}")
            print(f"Original indices range: [{combined_original_indices.min()}-{combined_original_indices.max()}]")
        
        # Create output directory for combined files
        output_dir = repo_root / "data" / "derivatives" / "trf"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Combined files output directory: {output_dir}")
        
        # Save combined TFR data
        combined_tfr_filename = "all_subs_desc-morlet_tfr.npz"
        combined_tfr_path = output_dir / combined_tfr_filename
        np.savez(combined_tfr_path, combined_tfr)
        print(f"✓ Combined TFR data saved: {combined_tfr_path}")
        print(f"  File size: {combined_tfr_path.stat().st_size / (1024**2):.2f} MB")
        
        # Save combined clean indices
        if combined_clean_indices is not None:
            combined_clean_idx_filename = "idx_data_all_subs.npz"
            combined_clean_idx_path = output_dir / combined_clean_idx_filename
            np.savez(combined_clean_idx_path, combined_clean_indices)
            print(f"✓ Combined clean indices saved: {combined_clean_idx_path}")
        
        # Save combined original indices
        if combined_original_indices is not None:
            combined_original_idx_filename = "idx_data_OLD_timepoints_all_subs.npz"
            combined_original_idx_path = output_dir / combined_original_idx_filename
            np.savez(combined_original_idx_path, combined_original_indices)
            print(f"✓ Combined original indices saved: {combined_original_idx_path}")
        
        # Save column names
        if column_names:
            combined_columns_filename = "all_subs_desc-morlet_columns.tsv"
            combined_columns_path = output_dir / combined_columns_filename
            columns_df = pd.DataFrame({'column_name': column_names})
            columns_df.to_csv(combined_columns_path, sep='\t', index=False)
            print(f"✓ Combined column names saved: {combined_columns_path}")
        
        # Extract frequency information from column names
        unique_freqs = []
        unique_channels = []
        if column_names:
            unique_freqs = sorted(list(set([name.split('_')[1] for name in column_names])))
            unique_channels = sorted(list(set([name.split('_')[0] for name in column_names])))
        
        # Create comprehensive metadata for combined data
        combined_metadata = {
            "Description": "Combined time-frequency representation from all subjects",
            "Subjects": successful_subjects,
            "SubjectOrder": successful_subjects,
            "Method": "Morlet wavelets",
            "Frequencies": unique_freqs,
            "Channels": unique_channels,
            "NumberOfChannels": len(unique_channels),
            "NumberOfFrequencies": len(unique_freqs),
            "TotalTimePoints": combined_tfr.shape[0],
            "TotalOriginalTimePoints": cumulative_original_timepoints,
            "DataShape": list(combined_tfr.shape),
            "NumberOfSubjects": len(successful_subjects),
            "DataRetention": float(combined_tfr.shape[0] / cumulative_original_timepoints) if cumulative_original_timepoints > 0 else 1.0,
            "Units": "Power (arbitrary units)",
            "ColumnOrder": column_names,
            "CleaningApplied": True,
            "GeneratedBy": {
                "Name": "eeg_tfr.py",
                "Description": "Combined multi-subject TFR analysis using MNE-Python Morlet wavelets",
                "Version": "2.2",
                "Parameters": {
                    "frequencies": freqs.tolist(),
                    "n_cycles": n_cycles
                }
            }
        }
        
        # Add subject-specific information
        subject_breakdown = []
        cumulative_clean = 0
        cumulative_original = 0
        
        for i, (subject, tfr_data, metadata) in enumerate(zip(successful_subjects, all_subjects_data, all_subjects_metadata)):
            clean_points = tfr_data.shape[0]
            original_points = metadata.get('TotalOriginalTimePoints', clean_points)
            num_runs = metadata.get('NumberOfRuns', 0)
            
            subject_breakdown.append({
                "subject": subject,
                "order": i + 1,
                "clean_timepoints": clean_points,
                "original_timepoints": original_points,
                "number_of_runs": num_runs,
                "data_retention": metadata.get('DataRetention', 1.0),
                "clean_start_index": cumulative_clean,
                "clean_end_index": cumulative_clean + clean_points,
                "original_start_index": cumulative_original,
                "original_end_index": cumulative_original + original_points
            })
            
            cumulative_clean += clean_points
            cumulative_original += original_points
        
        combined_metadata["SubjectBreakdown"] = subject_breakdown
        
        # Add indices information if available
        if combined_clean_indices is not None:
            combined_metadata["CleanSegments"] = combined_clean_indices.tolist()
            combined_metadata["NumberOfCleanSegments"] = len(combined_clean_indices)
            combined_metadata["IndexFiles"] = {
                "CleanIndices": combined_clean_idx_filename,
                "OriginalIndices": combined_original_idx_filename if combined_original_indices is not None else None,
                "Description": "Index files map between clean (cropped) and original timepoints across all subjects"
            }
        
        # Save combined metadata
        combined_metadata_filename = "all_subs_desc-morlet_tfr.json"
        combined_metadata_path = output_dir / combined_metadata_filename
        with open(combined_metadata_path, 'w') as f:
            json.dump(combined_metadata, f, indent=4)
        print(f"✓ Combined metadata saved: {combined_metadata_path}")
        
        # Validation: Verify data and indices consistency
        print(f"\n=== VALIDATION ===")
        data_length = combined_tfr.shape[0]
        if combined_clean_indices is not None:
            max_clean_idx = combined_clean_indices.max()
            if max_clean_idx > data_length:
                print(f"❌ Validation failed: max clean index ({max_clean_idx}) > data length ({data_length})")
                return False
            else:
                print(f"✓ Clean indices validation passed: max index {max_clean_idx} <= data length {data_length}")
        
        # Guardar archivos combinados
        output_dir = repo_root / "data" / "derivatives" / "trf"
        np.savez(output_dir / "idx_data_OLD_timepoints_in_seconds_from_start_all_subs.npz",
                 np.array(all_old_timepoints_seconds_from_start))
        with open(output_dir / "filelog_concatenate_all_subs.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename'])
            for fname in all_filelogs:
                writer.writerow([fname])
        
        # Summary
        print(f"\n=== COMBINATION SUMMARY ===")
        print(f"Subjects combined: {successful_subjects}")
        print(f"Subject order maintained consistently in data and indices")
        print(f"Total clean timepoints: {combined_tfr.shape[0]:,}")
        print(f"Total original timepoints: {cumulative_original_timepoints:,}")
        print(f"Overall data retention: {100*combined_tfr.shape[0]/cumulative_original_timepoints:.1f}%")
        print(f"Features per timepoint: {combined_tfr.shape[1]}")
        if combined_clean_indices is not None:
            print(f"Total clean segments: {len(combined_clean_indices)}")
        
        print(f"\nCombined files saved in: {output_dir}")
        print(f"  - {combined_tfr_filename}")
        if combined_clean_indices is not None:
            print(f"  - {combined_clean_idx_filename}")
        if combined_original_indices is not None:
            print(f"  - {combined_original_idx_filename}")
        print(f"  - {combined_columns_filename}")
        print(f"  - {combined_metadata_filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error combining all subjects data: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main function to execute the TFR analysis pipeline for all available files of a subject.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Compute time-frequency representation (TFR) for EEG data using Morlet wavelets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single subject
    python scripts/eeg_tfr.py --sub 14
    python scripts/eeg_tfr.py --sub 14 --freqs 2 4 8 16 32
    python scripts/eeg_tfr.py --sub 14 --cycles 4
    
    # Multiple subjects
    python scripts/eeg_tfr.py --sub 14 16 17 18
    python scripts/eeg_tfr.py --sub 14 16 17 18 --freqs 2 4 8 16 32 --cycles 4
    python scripts/eeg_tfr.py --sub 14 16 17 18 --skip-existing
    
    # Multiple subjects with cross-subject concatenation
    python scripts/eeg_tfr.py --sub 14 16 17 18 --combine-all-subs
    python scripts/eeg_tfr.py --sub 14 16 17 18 --combine-all-subs --freqs 2 4 8 16 32
        """
    )
    
    parser.add_argument(
        '--sub', '--subject', 
        type=str, 
        nargs='+',
        required=True,
        help='Subject ID(s) (e.g., 14 or 14 16 17 18 for multiple subjects)'
    )
    
    parser.add_argument(
        '--freqs', '--frequencies',
        type=float,
        nargs='+',
        default=[2.0, 4.0, 8.0, 16.0, 32.0],
        help='Frequencies for TFR analysis in Hz (default: 2 4 8 16 32)'
    )
    
    parser.add_argument(
        '--cycles', '--n-cycles',
        type=int,
        default=6,
        help='Number of cycles per wavelet (default: 6)'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip files that already have TFR output'
    )
    
    parser.add_argument(
        '--combine-all-subs',
        action='store_true',
        help='Combine all subjects into single concatenated files (all_subs_*). Requires multiple subjects.'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    print("="*80)
    print("EEG TIME-FREQUENCY REPRESENTATION ANALYSIS")
    print("="*80)
    
    # Handle multiple subjects
    subjects = args.sub
    print(f"Subjects to process: {subjects}")
    print(f"Total subjects: {len(subjects)}")
    print(f"Frequencies: {args.freqs} Hz")
    print(f"Number of cycles: {args.cycles}")
    print(f"Skip existing files: {args.skip_existing}")
    print(f"Combine all subjects: {args.combine_all_subs}")
    
    # Validate combine-all-subs flag
    if args.combine_all_subs and len(subjects) < 2:
        print(f"❌ Error: --combine-all-subs requires at least 2 subjects, but only {len(subjects)} provided")
        return 1
    
    # TFR parameters
    freqs = np.array(args.freqs)
    n_cycles = args.cycles
    
    # Track processing results across all subjects
    subject_results = {}
    successful_subjects = []
    failed_subjects = []
    skipped_subjects = []
    
    try:
        # Process each subject
        for subject_idx, subject in enumerate(subjects, 1):
            print(f"\n{'='*100}")
            print(f"PROCESSING SUBJECT {subject_idx}/{len(subjects)}: {subject}")
            print(f"{'='*100}")
            
            try:
                # Step 1: Find all available EEG files for the subject
                available_files = find_available_eeg_files(subject)
                
                if not available_files:
                    print(f"❌ No EEG files found for subject {subject}")
                    failed_subjects.append(subject)
                    subject_results[subject] = {
                        'status': 'failed',
                        'reason': 'No EEG files found',
                        'files_found': 0
                    }
                    continue
                
                # Step 2: Check if output already exists (for skip-existing option)
                if args.skip_existing:
                    output_dir = repo_root / "data" / "derivatives" / "trf" / f"sub-{subject}"
                    concat_tfr_filename = f"sub-{subject}_desc-morlet_tfr_concatenated.npz"
                    concat_tfr_path = output_dir / concat_tfr_filename
                    
                    if concat_tfr_path.exists():
                        print(f"⏭️  SKIPPING: Concatenated output already exists for subject {subject}")
                        print(f"Output file: {concat_tfr_path}")
                        skipped_subjects.append(subject)
                        subject_results[subject] = {
                            'status': 'skipped',
                            'reason': 'Output already exists',
                            'files_found': len(available_files),
                            'output_file': str(concat_tfr_path)
                        }
                        continue
                
                # Step 3: Process all files and generate concatenated data directly
                processing_success = process_and_concatenate_subject_data(subject, available_files, freqs, n_cycles)
                
                if processing_success:
                    successful_subjects.append(subject)
                    output_dir = repo_root / "data" / "derivatives" / "trf" / f"sub-{subject}"
                    subject_results[subject] = {
                        'status': 'success',
                        'files_found': len(available_files),
                        'output_dir': str(output_dir)
                    }
                    print(f"✓ Subject {subject} processed successfully")
                else:
                    failed_subjects.append(subject)
                    subject_results[subject] = {
                        'status': 'failed',
                        'reason': 'Processing failed',
                        'files_found': len(available_files)
                    }
                    print(f"❌ Subject {subject} processing failed")
                    
            except Exception as e:
                print(f"❌ Critical error processing subject {subject}: {e}")
                import traceback
                traceback.print_exc()
                failed_subjects.append(subject)
                subject_results[subject] = {
                    'status': 'failed',
                    'reason': f'Critical error: {str(e)}',
                    'files_found': 0
                }
        
        # Step 4: Combine all subjects if requested
        combination_success = True
        if args.combine_all_subs:
            if successful_subjects:
                print(f"\n{'='*100}")
                print(f"COMBINING ALL SUBJECTS (--combine-all-subs FLAG ACTIVE)")
                print(f"{'='*100}")
                combination_success = combine_all_subjects_data(successful_subjects, freqs, n_cycles)
                
                if combination_success:
                    print(f"✓ All subjects successfully combined into final files")
                else:
                    print(f"❌ Failed to combine all subjects data")
            else:
                print(f"\n⚠️  Cannot combine subjects: no subjects were successfully processed")
                combination_success = False
        
        # Final Summary across all subjects
        print(f"\n{'='*100}")
        print("FINAL PROCESSING SUMMARY - ALL SUBJECTS")
        print(f"{'='*100}")
        
        print(f"Total subjects requested: {len(subjects)}")
        print(f"Successfully processed: {len(successful_subjects)}")
        print(f"Failed: {len(failed_subjects)}")
        print(f"Skipped: {len(skipped_subjects)}")
        
        print(f"\nTFR Analysis Configuration:")
        print(f"  Frequencies analyzed: {freqs} Hz")
        print(f"  Number of cycles used: {n_cycles}")
        print(f"  Method: Morlet wavelets")
        print(f"  Event-based cleaning: Applied (excludes 'bad' segments)")
        print(f"  Output mode: Direct concatenation only")
        if args.combine_all_subs:
            print(f"  Multi-subject combination: {'✓ Applied' if combination_success else '❌ Failed'}")
        
        # Detailed results by subject
        if successful_subjects:
            print(f"\n✓ SUCCESSFULLY PROCESSED SUBJECTS ({len(successful_subjects)}):")
            for subject in successful_subjects:
                result = subject_results[subject]
                print(f"  - Subject {subject}: {result['files_found']} files processed")
                print(f"    Output: {result['output_dir']}")
        
        if skipped_subjects:
            print(f"\n⏭️  SKIPPED SUBJECTS ({len(skipped_subjects)}):")
            for subject in skipped_subjects:
                result = subject_results[subject]
                print(f"  - Subject {subject}: {result['reason']}")
                print(f"    Found {result['files_found']} files, output exists at: {result.get('output_file', 'N/A')}")
        
        if failed_subjects:
            print(f"\n❌ FAILED SUBJECTS ({len(failed_subjects)}):")
            for subject in failed_subjects:
                result = subject_results[subject]
                print(f"  - Subject {subject}: {result['reason']}")
                if result['files_found'] > 0:
                    print(f"    Files found: {result['files_found']}")
        
        print(f"\nOutput locations:")
        base_output_dir = repo_root / "data" / "derivatives" / "trf"
        print(f"  Base directory: {base_output_dir}")
        for subject in successful_subjects + skipped_subjects:
            print(f"  - sub-{subject}/")
            print(f"    └── sub-{subject}_desc-morlet_tfr_concatenated.npz")
            print(f"    └── idx_data_sub-{subject}_concatenated.npz")
            print(f"    └── idx_data_OLD_timepoints_sub-{subject}_concatenated.npz")
            print(f"    └── sub-{subject}_desc-morlet_columns_concatenated.tsv")
            print(f"    └── sub-{subject}_desc-morlet_tfr_concatenated.json")
        
        if args.combine_all_subs and combination_success:
            output_dir = repo_root / "data" / "derivatives" / "trf"
            print(f"\nCombined files for ALL SUBJECTS:")
            print(f"  Location: {output_dir}")
            print(f"  - all_subs_desc-morlet_tfr.npz")
            print(f"  - idx_data_all_subs.npz")
            print(f"  - idx_data_OLD_timepoints_all_subs.npz")
            print(f"  - all_subs_desc-morlet_columns.tsv")
            print(f"  - all_subs_desc-morlet_tfr.json")
            print(f"  Subject order: {successful_subjects}")
        
        print(f"\nNote: Individual run files are NOT saved in this version.")
        print(f"      All data is processed and concatenated directly for each subject.")
        if args.combine_all_subs:
            print(f"      When --combine-all-subs is used, additional all_subs_* files are created.")
        
        # Return appropriate exit code
        if failed_subjects and not successful_subjects and not skipped_subjects:
            return 1  # All subjects failed
        elif failed_subjects:
            return 2  # Some subjects failed
        elif args.combine_all_subs and not combination_success:
            return 3  # Subject processing succeeded but combination failed
        else:
            return 0  # All successful or skipped
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 