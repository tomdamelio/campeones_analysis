#!/usr/bin/env python
"""
EEG Time-Frequency Representation (TFR) Analysis Script with Event-Based Cleaning

This script computes time-frequency decomposition using Morlet wavelets for preprocessed EEG data
from the Campeones Analysis project. It automatically processes all available sessions, tasks, and runs
for a specified subject, and saves the resulting TFR data in a clean format suitable for analysis.

Features:
- Loads preprocessed EEG data from derivatives/campeones_preproc
- Automatically discovers all available sessions/tasks/runs for a subject
- Computes TFR using Morlet wavelets at specific frequency bands
- Loads event annotations from preprocessing to identify 'bad' segments
- Creates clean TFR data by excluding 'bad' segments and concatenating good segments
- Generates index mapping files to track timepoint correspondence
- Reshapes data to (n_times, n_channels*n_freqs) format
- Saves clean results as .npy files with comprehensive metadata and column descriptions

Output Files (per run):
- *_desc-morlet_tfr.npy : Clean TFR data (bad segments excluded)
- *_desc-morlet_columns.tsv : Column names and order
- *_desc-morlet_tfr.json : Comprehensive metadata including segment mapping
- idx_data_*.npy : Clean data indices (segment boundaries in clean timepoints)
- idx_data_OLD_timepoints_*.npy : Original data indices (segment boundaries in original timepoints)

Usage:
    python scripts/eeg_tfr.py --sub 14
    python scripts/eeg_tfr.py --sub 14 --freqs 2 4 8 16 32 --cycles 4

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
    bool
        True if processing was successful, False otherwise
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
            print(f"⚠️  No events available - saving original TFR data without cleaning")
        
        # Step 6: Save results
        data_path, columns_path = save_tfr_data(
            reshaped_data, column_names, 
            file_info['subject'], file_info['session'], file_info['task'], 
            file_info['acquisition'], file_info['run'], raw.info['sfreq'],
            clean_data, clean_segments_indices, original_segments_indices, events_summary
        )
        
        print(f"\n✓ PROCESSING COMPLETED SUCCESSFULLY")
        print(f"  Data file: {data_path}")
        print(f"  Columns file: {columns_path}")
        
        if clean_data is not None:
            print(f"  Clean data shape: {clean_data.shape}")
            print(f"  Original data shape: {reshaped_data.shape}")
            print(f"  Data retention: {100*clean_data.shape[0]/reshaped_data.shape[0]:.1f}%")
            print(f"  Clean segments: {len(clean_segments_indices) if clean_segments_indices is not None else 0}")
        else:
            print(f"  Final data shape: {reshaped_data.shape}")
        
        print(f"  Features (channels × frequencies): {reshaped_data.shape[1]}")
        print(f"  Column order maintained: {len(column_names)} features")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR processing sub-{file_info['subject']} ses-{file_info['session']} "
              f"task-{file_info['task']} acq-{file_info['acquisition']} run-{file_info['run']}: {e}")
        import traceback
        traceback.print_exc()
        return False


def concatenate_subject_data(subject, trf_output_dir):
    """
    Concatenate all TFR data files for a subject into single combined files.
    
    This function finds all processed TFR files for a subject, concatenates the data,
    and adjusts the indices appropriately to account for the concatenation.
    
    Parameters
    ----------
    subject : str
        Subject ID (e.g., "14")
    trf_output_dir : Path
        Directory containing the individual TFR files
        
    Returns
    -------
    bool
        True if concatenation was successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"CONCATENATING ALL DATA FOR SUBJECT {subject}")
    print(f"{'='*80}")
    
    try:
        # Find all TFR files for this subject
        tfr_pattern = f"sub-{subject}_*_desc-morlet_tfr.npz"
        tfr_files = sorted(list(trf_output_dir.glob(tfr_pattern)))
        
        if not tfr_files:
            print(f"❌ No TFR files found for subject {subject}")
            return False
        
        print(f"Found {len(tfr_files)} TFR files for subject {subject}:")
        for tfr_file in tfr_files:
            print(f"  - {tfr_file.name}")
        
        # Lists to store data and metadata
        all_tfr_data = []
        all_clean_indices = []
        all_original_indices = []
        all_metadata = []
        
        # Counters for index adjustment
        cumulative_clean_timepoints = 0
        cumulative_original_timepoints = 0
        
        # Process each file
        for i, tfr_file in enumerate(tfr_files):
            print(f"\nProcessing file {i+1}/{len(tfr_files)}: {tfr_file.name}")
            
            # Extract run information from filename
            base_name = tfr_file.stem.replace('_desc-morlet_tfr', '')
            
            # Load TFR data
            tfr_data = np.load(tfr_file)
            all_tfr_data.append(tfr_data['arr_0'])
            print(f"  TFR data shape: {tfr_data['arr_0'].shape}")
            
            # Load metadata
            metadata_file = tfr_file.parent / f"{base_name}_desc-morlet_tfr.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                all_metadata.append(metadata)
                
                original_timepoints = metadata.get('OriginalTimePoints', tfr_data['arr_0'].shape[0])
                print(f"  Original timepoints: {original_timepoints:,}")
                print(f"  Clean timepoints: {tfr_data['arr_0'].shape[0]:,}")
            else:
                print(f"  ⚠️  Metadata file not found: {metadata_file.name}")
                original_timepoints = tfr_data['arr_0'].shape[0]  # Fallback
                all_metadata.append({})
            
            # Load clean data indices
            clean_idx_file = tfr_file.parent / f"idx_data_{base_name}.npz"
            if clean_idx_file.exists():
                clean_indices = np.load(clean_idx_file)
                
                # Adjust clean indices for concatenation
                adjusted_clean_indices = clean_indices['arr_0'] + cumulative_clean_timepoints
                all_clean_indices.append(adjusted_clean_indices)
                
                print(f"  Clean indices shape: {clean_indices['arr_0'].shape}")
                print(f"  Clean indices adjustment: +{cumulative_clean_timepoints}")
                print(f"  Clean indices (original): {clean_indices['arr_0']}")
                print(f"  Clean indices (adjusted): {adjusted_clean_indices}")
            else:
                print(f"  ⚠️  Clean indices file not found: {clean_idx_file.name}")
            
            # Load original timepoints indices
            original_idx_file = tfr_file.parent / f"idx_data_OLD_timepoints_{base_name}.npz"
            if original_idx_file.exists():
                original_indices = np.load(original_idx_file)
                
                # Adjust original indices for concatenation
                adjusted_original_indices = original_indices['arr_0'] + cumulative_original_timepoints
                all_original_indices.append(adjusted_original_indices)
                
                print(f"  Original indices shape: {original_indices['arr_0'].shape}")
                print(f"  Original indices adjustment: +{cumulative_original_timepoints}")
                print(f"  Original indices (original): {original_indices['arr_0']}")
                print(f"  Original indices (adjusted): {adjusted_original_indices}")
            else:
                print(f"  ⚠️  Original indices file not found: {original_idx_file.name}")
            
            # Update cumulative counters
            cumulative_clean_timepoints += tfr_data['arr_0'].shape[0]
            cumulative_original_timepoints += original_timepoints
            
            print(f"  Cumulative clean timepoints: {cumulative_clean_timepoints:,}")
            print(f"  Cumulative original timepoints: {cumulative_original_timepoints:,}")
        
        # Concatenate all data
        print(f"\n=== CONCATENATING DATA ===")
        concatenated_tfr = np.vstack(all_tfr_data)
        print(f"Concatenated TFR data shape: {concatenated_tfr.shape}")
        
        # Concatenate indices if available
        concatenated_clean_indices = None
        concatenated_original_indices = None
        
        if all_clean_indices:
            concatenated_clean_indices = np.vstack(all_clean_indices)
            print(f"Concatenated clean indices shape: {concatenated_clean_indices.shape}")
            print(f"Final clean indices:\n{concatenated_clean_indices}")
        
        if all_original_indices:
            concatenated_original_indices = np.vstack(all_original_indices)
            print(f"Concatenated original indices shape: {concatenated_original_indices.shape}")
            print(f"Final original indices:\n{concatenated_original_indices}")
        
        # Create subject-specific output directory
        subject_output_dir = trf_output_dir / f"sub-{subject}"
        subject_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Subject output directory: {subject_output_dir}")
        
        # Save concatenated TFR data
        concat_tfr_filename = f"sub-{subject}_desc-morlet_tfr_concatenated.npz"
        concat_tfr_path = subject_output_dir / concat_tfr_filename
        np.savez(concat_tfr_path, concatenated_tfr)
        print(f"✓ Concatenated TFR data saved: {concat_tfr_path}")
        print(f"  File size: {concat_tfr_path.stat().st_size / (1024**2):.2f} MB")
        
        # Save concatenated clean indices
        if concatenated_clean_indices is not None:
            concat_clean_idx_filename = f"idx_data_sub-{subject}_concatenated.npz"
            concat_clean_idx_path = subject_output_dir / concat_clean_idx_filename
            np.savez(concat_clean_idx_path, concatenated_clean_indices)
            print(f"✓ Concatenated clean indices saved: {concat_clean_idx_path}")
        
        # Save concatenated original indices
        if concatenated_original_indices is not None:
            concat_original_idx_filename = f"idx_data_OLD_timepoints_sub-{subject}_concatenated.npz"
            concat_original_idx_path = subject_output_dir / concat_original_idx_filename
            np.savez(concat_original_idx_path, concatenated_original_indices)
            print(f"✓ Concatenated original indices saved: {concat_original_idx_path}")
        
        # Load column names from first file (should be same for all)
        first_base_name = tfr_files[0].stem.replace('_desc-morlet_tfr', '')
        columns_file = tfr_files[0].parent / f"{first_base_name}_desc-morlet_columns.tsv"
        if columns_file.exists():
            columns_df = pd.read_csv(columns_file, sep='\t')
            
            # Save concatenated column names
            concat_columns_filename = f"sub-{subject}_desc-morlet_columns_concatenated.tsv"
            concat_columns_path = subject_output_dir / concat_columns_filename
            columns_df.to_csv(concat_columns_path, sep='\t', index=False)
            print(f"✓ Column names copied: {concat_columns_path}")
        
        # Create comprehensive metadata for concatenated data
        concat_metadata = {
            "Description": "Concatenated time-frequency representation for entire subject",
            "Subject": subject,
            "Method": "Morlet wavelets",
            "ConcatenatedFiles": [f.name for f in tfr_files],
            "TotalTimePoints": concatenated_tfr.shape[0],
            "TotalOriginalTimePoints": cumulative_original_timepoints,
            "DataShape": list(concatenated_tfr.shape),
            "NumberOfRuns": len(tfr_files),
            "DataRetention": float(concatenated_tfr.shape[0] / cumulative_original_timepoints) if cumulative_original_timepoints > 0 else 1.0,
            "Units": "Power (arbitrary units)",
            "GeneratedBy": {
                "Name": "eeg_tfr.py",
                "Description": "Concatenated TFR analysis using MNE-Python Morlet wavelets",
                "Version": "2.0"
            }
        }
        
        # Add run-specific information
        run_info = []
        cumulative_clean = 0
        cumulative_original = 0
        
        for i, (tfr_data, metadata) in enumerate(zip(all_tfr_data, all_metadata)):
            run_name = tfr_files[i].stem.replace('_desc-morlet_tfr', '')
            clean_points = tfr_data.shape[0]
            original_points = metadata.get('OriginalTimePoints', clean_points)
            
            run_info.append({
                "run": run_name,
                "clean_timepoints": clean_points,
                "original_timepoints": original_points,
                "clean_start_index": cumulative_clean,
                "clean_end_index": cumulative_clean + clean_points,
                "original_start_index": cumulative_original,
                "original_end_index": cumulative_original + original_points
            })
            
            cumulative_clean += clean_points
            cumulative_original += original_points
        
        concat_metadata["RunBreakdown"] = run_info
        
        # Add indices information if available
        if concatenated_clean_indices is not None:
            concat_metadata["CleanSegments"] = concatenated_clean_indices.tolist()
            concat_metadata["NumberOfCleanSegments"] = len(concatenated_clean_indices)
        
        if concatenated_original_indices is not None:
            concat_metadata["OriginalSegments"] = concatenated_original_indices.tolist()
        
        # Save concatenated metadata
        concat_metadata_filename = f"sub-{subject}_desc-morlet_tfr_concatenated.json"
        concat_metadata_path = subject_output_dir / concat_metadata_filename
        with open(concat_metadata_path, 'w') as f:
            json.dump(concat_metadata, f, indent=4)
        print(f"✓ Concatenated metadata saved: {concat_metadata_path}")
        
        # Summary
        print(f"\n=== CONCATENATION SUMMARY ===")
        print(f"Subject: {subject}")
        print(f"Files concatenated: {len(tfr_files)}")
        print(f"Total clean timepoints: {concatenated_tfr.shape[0]:,}")
        print(f"Total original timepoints: {cumulative_original_timepoints:,}")
        print(f"Overall data retention: {100*concatenated_tfr.shape[0]/cumulative_original_timepoints:.1f}%")
        print(f"Features per timepoint: {concatenated_tfr.shape[1]}")
        if concatenated_clean_indices is not None:
            print(f"Total clean segments: {len(concatenated_clean_indices)}")
        
        print(f"\nConcatenated files saved in: {subject_output_dir}")
        print(f"  - {concat_tfr_filename}")
        if concatenated_clean_indices is not None:
            print(f"  - {concat_clean_idx_filename}")
        if concatenated_original_indices is not None:
            print(f"  - {concat_original_idx_filename}")
        print(f"  - {concat_columns_filename}")
        print(f"  - {concat_metadata_filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error concatenating data for subject {subject}: {e}")
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
    python scripts/eeg_tfr.py --sub 14
    python scripts/eeg_tfr.py --sub 14 --freqs 2 4 8 16 32
    python scripts/eeg_tfr.py --sub 14 --cycles 4
        """
    )
    
    parser.add_argument(
        '--sub', '--subject', 
        type=str, 
        required=True,
        help='Subject ID (e.g., 14)'
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
    
    # Parse arguments
    args = parser.parse_args()
    
    print("="*80)
    print("EEG TIME-FREQUENCY REPRESENTATION ANALYSIS")
    print("="*80)
    print(f"Subject: {args.sub}")
    print(f"Frequencies: {args.freqs} Hz")
    print(f"Number of cycles: {args.cycles}")
    print(f"Skip existing files: {args.skip_existing}")
    
    # TFR parameters
    freqs = np.array(args.freqs)
    n_cycles = args.cycles
    
    try:
        # Step 1: Find all available EEG files for the subject
        available_files = find_available_eeg_files(args.sub)
        
        if not available_files:
            print(f"❌ No EEG files found for subject {args.sub}")
            return 1
        
        # Step 2: Process each file
        successful_files = []
        failed_files = []
        skipped_files = []
        
        for i, file_info in enumerate(available_files, 1):
            print(f"\n{'='*20} FILE {i}/{len(available_files)} {'='*20}")
            
            # Check if output already exists
            if args.skip_existing:
                output_dir = repo_root / "data" / "derivatives" / "trf"
                base_filename = (f"sub-{file_info['subject']}_ses-{file_info['session']}_"
                               f"task-{file_info['task']}_acq-{file_info['acquisition']}_"
                               f"run-{file_info['run']}")
                data_filename = f"{base_filename}_desc-morlet_tfr.npz"
                data_path = output_dir / data_filename
                
                if data_path.exists():
                    print(f"⏭️  SKIPPING: Output already exists for {base_filename}")
                    skipped_files.append(file_info)
                    continue
            
            # Process the file
            success = process_single_file(file_info, freqs, n_cycles)
            
            if success:
                successful_files.append(file_info)
            else:
                failed_files.append(file_info)
        
        # Step 3: Generate concatenated files for the subject
        trf_output_dir = repo_root / "data" / "derivatives" / "trf"
        
        if successful_files or skipped_files:
            print(f"\n{'='*20} CONCATENATING SUBJECT DATA {'='*20}")
            concatenation_success = concatenate_subject_data(args.sub, trf_output_dir)
            
            if not concatenation_success:
                print(f"⚠️  Warning: Failed to concatenate data for subject {args.sub}")
        else:
            print(f"\n⚠️  No files were processed or found - skipping concatenation")
        
        # Step 4: Summary
        print(f"\n{'='*80}")
        print("PROCESSING SUMMARY")
        print(f"{'='*80}")
        print(f"Total files found: {len(available_files)}")
        print(f"Successfully processed: {len(successful_files)}")
        print(f"Failed: {len(failed_files)}")
        print(f"Skipped: {len(skipped_files)}")
        
        if successful_files:
            print(f"\n✓ Successfully processed files:")
            for file_info in successful_files:
                print(f"  - sub-{file_info['subject']} ses-{file_info['session']} "
                      f"task-{file_info['task']} acq-{file_info['acquisition']} run-{file_info['run']}")
        
        if failed_files:
            print(f"\n❌ Failed files:")
            for file_info in failed_files:
                print(f"  - sub-{file_info['subject']} ses-{file_info['session']} "
                      f"task-{file_info['task']} acq-{file_info['acquisition']} run-{file_info['run']}")
        
        if skipped_files:
            print(f"\n⏭️  Skipped files:")
            for file_info in skipped_files:
                print(f"  - sub-{file_info['subject']} ses-{file_info['session']} "
                      f"task-{file_info['task']} acq-{file_info['acquisition']} run-{file_info['run']}")
        
        print(f"\nTFR Analysis Configuration:")
        print(f"  Frequencies analyzed: {freqs} Hz")
        print(f"  Number of cycles used: {n_cycles}")
        print(f"  Method: Morlet wavelets")
        print(f"  Event-based cleaning: Applied (excludes 'bad' segments)")
        
        print(f"\nOutput directory: {repo_root / 'data' / 'derivatives' / 'trf'}")
        print(f"Individual files per run:")
        print(f"  - *_desc-morlet_tfr.npz (Clean TFR data)")
        print(f"  - *_desc-morlet_columns.tsv (Column names and order)")  
        print(f"  - *_desc-morlet_tfr.json (Comprehensive metadata)")
        print(f"  - idx_data_*.npz (Clean data segment indices)")
        print(f"  - idx_data_OLD_timepoints_*.npz (Original data indices)")
        
        if successful_files:
            print(f"\nConcatenated files for subject {args.sub}:")
            print(f"  Location: {trf_output_dir / f'sub-{args.sub}'}")
            print(f"  - sub-{args.sub}_desc-morlet_tfr_concatenated.npz")
            print(f"  - idx_data_sub-{args.sub}_concatenated.npz")
            print(f"  - idx_data_OLD_timepoints_sub-{args.sub}_concatenated.npz")
            print(f"  - sub-{args.sub}_desc-morlet_columns_concatenated.tsv")
            print(f"  - sub-{args.sub}_desc-morlet_tfr_concatenated.json")
        
        # Return appropriate exit code
        if failed_files and not successful_files:
            return 1  # All files failed
        elif failed_files:
            return 2  # Some files failed
        else:
            return 0  # All successful
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 