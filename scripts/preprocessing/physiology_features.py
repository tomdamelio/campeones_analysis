#!/usr/bin/env python
"""
Physiological Features Extraction Script with Event-Based Cleaning and Multi-Subject Support

This script extracts physiological features from preprocessed physiological data (ECG, RESP, GSR)
from the Campeones Analysis project. It automatically processes all available sessions, tasks, and runs
for specified subject(s), and saves the resulting physiological features in a clean format suitable for analysis.
Supports processing multiple subjects simultaneously and combining all subjects into unified files.

Features:
- Loads preprocessed physiological data from derivatives/campeones_preproc
- Automatically discovers all available sessions/tasks/runs for subject(s)
- Extracts physiological features from ECG, RESP (respiration), and GSR channels
- Loads event annotations from preprocessing to identify 'bad' segments
- Creates clean feature data by excluding 'bad' segments and concatenating good segments
- Generates index mapping files to track timepoint correspondence
- Reshapes data to (n_times, physiological_features) format
- Supports processing multiple subjects simultaneously
- Optional cross-subject concatenation into unified all_subs_* files
- Maintains consistent ordering between data and indices across all levels of concatenation
- Saves clean results as .npz files with comprehensive metadata and feature descriptions

Output Files (per subject):
- sub-{subject}_desc-physio_features_concatenated.npz : Clean physiological features (bad segments excluded)
- sub-{subject}_desc-physio_columns_concatenated.tsv : Feature names and order
- sub-{subject}_desc-physio_features_concatenated.json : Comprehensive metadata including segment mapping
- idx_data_sub-{subject}_concatenated.npz: Clean data indices (segment boundaries in clean timepoints)
- idx_data_OLD_timepoints_sub-{subject}_concatenated.npz : Original data indices (segment boundaries in original timepoints)

Output Files (when --combine-all-subs is used):
- all_subs_desc-physio_features.npz : Clean physiological features from all subjects (bad segments excluded)
- all_subs_desc-physio_columns.tsv : Feature names and order
- all_subs_desc-physio_features.json : Comprehensive metadata including segment mapping
- idx_data_all_subs.npz: Clean data indices (segment boundaries in clean timepoints)
- idx_data_OLD_timepoints_all_subs.npz : Original data indices (segment boundaries in original timepoints)

Usage:
    # Single subject
    python scripts/preprocessing/physiology_features.py --sub 14
    
    # Multiple subjects
    python scripts/preprocessing/physiology_features.py --sub 14 16 17 18
    
    # Multiple subjects with cross-subject concatenation
    python scripts/preprocessing/physiology_features.py --sub 14 16 17 18 --combine-all-subs

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
import neurokit2 as nk

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


def load_physio_data(subject, session, task, acquisition, run, data="eeg"):
    """
    Load preprocessed physiological data using BIDS format.
    
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
        Data type (default: "eeg" - same file contains physiological channels)
        
    Returns
    -------
    mne.io.Raw
        Loaded raw physiological data with GSR, ECG, RESP, and joystick_y channels
    """
    print(f"\n=== LOADING PHYSIOLOGICAL DATA ===")
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
        raise FileNotFoundError(f"Preprocessed data file not found: {bids_path.fpath}")
    
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
    
    # Define physiological channels we want to extract
    target_channels = ["GSR", "ECG", "RESP", "joystick_y"]
    
    # Check which target channels are available
    available_channels = []
    missing_channels = []
    
    for ch in target_channels:
        if ch in raw.ch_names:
            available_channels.append(ch)
        else:
            missing_channels.append(ch)
    
    if not available_channels:
        raise ValueError(f"None of the target physiological channels {target_channels} found in data. Available channels: {raw.ch_names}")
    
    if missing_channels:
        print(f"⚠️  Missing channels: {missing_channels}")
    
    print(f"Filtrando canales fisiológicos: {available_channels}")
    
    # Pick only the available physiological channels
    raw.pick_channels(available_channels)
    
    print(f"✓ Datos fisiológicos filtrados: {raw.info['nchan']} canales, {raw.n_times} puntos de muestreo.")
    print(f"✓ Frecuencia de muestreo: {raw.info['sfreq']} Hz")
    print(f"✓ Duración: {raw.times[-1]:.2f} segundos")
    print(f"✓ Canales fisiológicos: {raw.ch_names}")
    
    return raw


def extract_physio_features(raw):
    """
    Extract continuous physiological features from preprocessed physiological data.
    
    This function extracts:
    - GSR tonic and phasic components using NeuroKit2 decomposition
    - Raw signals from ECG, RESP, and joystick_y channels
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw physiological data containing GSR, ECG, RESP, and joystick_y channels
        
    Returns
    -------
    features_array : np.ndarray
        Physiological features array with shape (n_times, n_features)
        Features include: GSR_tonic, GSR_phasic, ECG, RESP, joystick_y
    column_names : list
        List of feature names corresponding to the features
        ['GSR_tonic', 'GSR_phasic', 'ECG', 'RESP', 'joystick_y']
    """
    print(f"\n=== EXTRACTING PHYSIOLOGICAL FEATURES ===")
    
    # Get available channels
    available_channels = raw.ch_names
    print(f"Available channels: {available_channels}")
    
    # Get sampling frequency and number of timepoints
    sfreq = raw.info['sfreq']
    n_times = raw.n_times
    print(f"Sampling frequency: {sfreq} Hz")
    print(f"Number of timepoints: {n_times:,}")
    
    # Initialize features list and column names
    features_list = []
    column_names = []
    
    # Process GSR signal if available
    if 'GSR' in available_channels:
        print("Processing GSR signal...")
        
        # Get GSR data
        gsr_data = raw.get_data(picks=['GSR'])[0]  # Shape: (n_times,)
        print(f"GSR data shape: {gsr_data.shape}")
        
        try:
            # Use NeuroKit2 to decompose GSR signal into tonic and phasic components
            print("Decomposing GSR into tonic and phasic components using NeuroKit2...")
            
            # Process EDA (GSR) signal
            eda_decomposed = nk.eda_process(gsr_data, sampling_rate=sfreq)[0]
            
            # Extract tonic and phasic components
            tonic_component = eda_decomposed['EDA_Tonic'].values
            phasic_component = eda_decomposed['EDA_Phasic'].values
            
            print(f"✓ GSR tonic component shape: {tonic_component.shape}")
            print(f"✓ GSR phasic component shape: {phasic_component.shape}")
            
            # Add features to list
            features_list.append(tonic_component)
            features_list.append(phasic_component)
            
            # Generate column names programmatically
            gsr_feature_names = ['GSR_tonic', 'GSR_phasic']
            column_names.extend(gsr_feature_names)
            
            print(f"✓ Added GSR features: {gsr_feature_names}")
            
        except Exception as e:
            print(f"⚠️  Error processing GSR signal: {e}")
            print("Adding zero arrays as fallback...")
            
            # Fallback: add zero arrays
            features_list.append(np.zeros(n_times))
            features_list.append(np.zeros(n_times))
            column_names.extend(['GSR_tonic', 'GSR_phasic'])
    
    else:
        print("⚠️  GSR channel not found. Adding zero features as placeholders.")
        features_list.append(np.zeros(n_times))
        features_list.append(np.zeros(n_times))
        column_names.extend(['GSR_tonic', 'GSR_phasic'])
    
    # Process raw signals for ECG, RESP, and joystick_y if available
    raw_signal_channels = ['ECG', 'RESP', 'joystick_y']
    
    for channel in raw_signal_channels:
        if channel in available_channels:
            print(f"Adding raw {channel} signal...")
            
            # Get raw channel data
            channel_data = raw.get_data(picks=[channel])[0]  # Shape: (n_times,)
            print(f"✓ {channel} data shape: {channel_data.shape}")
            
            # Add raw signal to features
            features_list.append(channel_data)
            column_names.append(channel)
            
            print(f"✓ Added raw {channel} signal")
            
        else:
            print(f"⚠️  {channel} channel not found. Adding zero signal as placeholder.")
            features_list.append(np.zeros(n_times))
            column_names.append(channel)
    
    # Stack features into final array
    if features_list:
        # Convert to numpy array: each feature is a column
        features_array = np.column_stack(features_list)
        print(f"✓ Features array created with shape: {features_array.shape} (n_times, n_features)")
    else:
        print("⚠️  No features extracted, creating empty array")
        features_array = np.empty((n_times, 0))
    
    # Validate shapes
    expected_shape = (n_times, len(column_names))
    if features_array.shape != expected_shape:
        raise ValueError(f"Shape mismatch: features_array {features_array.shape} != expected {expected_shape}")
    
    print(f"✓ Final features shape: {features_array.shape}")
    print(f"✓ Feature names: {column_names}")
    
    # Summary of what was actually extracted vs placeholders
    extracted_channels = [ch for ch in ['GSR', 'ECG', 'RESP', 'joystick_y'] if ch in available_channels]
    missing_channels = [ch for ch in ['GSR', 'ECG', 'RESP', 'joystick_y'] if ch not in available_channels]
    
    print(f"✓ Successfully extracted features from: {extracted_channels}")
    if missing_channels:
        print(f"⚠️  Zero placeholders added for missing channels: {missing_channels}")
    
    print(f"✓ Total features extracted: {len(column_names)} (GSR_tonic, GSR_phasic + {len(raw_signal_channels)} raw signals)")
    
    return features_array, column_names


def save_physiological_data(reshaped_data, column_names, subject, session, task, acquisition, run, sfreq, 
                        clean_data=None, clean_segments_indices=None, original_segments_indices=None, 
                        events_summary=None):
    """
    Save physiological features and column information to files.
    
    Parameters
    ----------
    reshaped_data : np.ndarray
        Original physiological features with shape (n_times, n_features)
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
        Sampling frequency of the original data in Hz
    clean_data : np.ndarray, optional
        Clean physiological features with 'bad' segments removed
    clean_segments_indices : np.ndarray, optional
        [N_segments, 2] - start/end indices in clean data
    original_segments_indices : np.ndarray, optional
        [N_segments, 2] - start/end indices in original data
    events_summary : dict, optional
        Summary of events used for cleaning
        
    Returns
    -------
    data_path : Path
        Path to saved clean .npz file
    columns_path : Path
        Path to saved .tsv file with column names
    """
    print(f"\n=== SAVING PHYSIOLOGICAL FEATURES DATA ===")
    
    # Create output directory
    output_dir = repo_root / "data" / "derivatives" / "physio"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create BIDS-compliant filenames
    base_filename = f"sub-{subject}_ses-{session}_task-{task}_acq-{acquisition}_run-{run}"
    
    # Determine which data to save as main output (prefer clean data if available)
    if clean_data is not None and len(clean_data) > 0:
        main_data = clean_data
        data_description = "Physiological features (continuous, cleaned)"
        print(f"Saving clean physiological features as main output (excluding bad segments)")
    else:
        main_data = reshaped_data
        data_description = "Physiological features (continuous, uncleaned)"
        print(f"Saving original physiological features as main output (no events available for cleaning)")
    
    # Save main physiological features as .npz file
    data_filename = f"{base_filename}_desc-physio_features.npz"
    data_path = output_dir / data_filename
    np.savez(data_path, main_data)
    print(f"✓ Main physiological features saved: {data_path}")
    print(f"  Data shape: {main_data.shape}")
    print(f"  File size: {data_path.stat().st_size / (1024**2):.2f} MB")
    
    # Save column names as .tsv file
    columns_filename = f"{base_filename}_desc-physio_columns.tsv"
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
    metadata_filename = f"{base_filename}_desc-physio_features.json"
    metadata_path = output_dir / metadata_filename
    
    metadata = {
        "Description": data_description,
        "Method": "Continuous physiological feature extraction (raw and processed)",
        "Channels": column_names,
        "NumberOfChannels": len(column_names),
        "NumberOfTimePoints": main_data.shape[0],
        "DataShape": list(main_data.shape),
        "Units": "arbitrary units (raw/processed)",
        "SamplingFrequency": float(sfreq),
        "SamplingFrequencyUnit": "Hz",
        "ColumnOrder": column_names,  # Explicit column order
        "GeneratedBy": {
            "Name": "physiology_features.py",
            "Description": "Continuous physiological feature extraction using NeuroKit2 and raw signals",
            "Version": "1.0"
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


def process_single_file(file_info):
    """
    Process a single physiological file for feature extraction with event-based cleaning.
    Returns data in memory without saving to disk.
    
    Parameters
    ----------
    file_info : dict
        Dictionary containing subject, session, task, acquisition, run information

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
        
        # Step 1: Load physiological data
        raw = load_physio_data(
            file_info['subject'], file_info['session'], file_info['task'], 
            file_info['acquisition'], file_info['run']
        )
        
        # Step 2: Extract physiological features
        features_data, column_names = extract_physio_features(raw)
        
        # Step 3: Load events for cleaning
        events_df = load_events_from_eeg(
            file_info['subject'], file_info['session'], file_info['task'], 
            file_info['acquisition'], file_info['run']
        )
        
        # Step 4: Create clean data and indices (if events are available)
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
                features_data, events_df, raw.info['sfreq'], 
                file_info['subject'], file_info['session'], file_info['task'], 
                file_info['acquisition'], file_info['run']
            )
        else:
            print(f"⚠️  No events available - using original physiological features without cleaning")
            clean_data = features_data
        
        print(f"\n✓ PROCESSING COMPLETED SUCCESSFULLY")
        
        if clean_data is not None and events_df is not None:
            print(f"  Clean data shape: {clean_data.shape}")
            print(f"  Original data shape: {features_data.shape}")
            print(f"  Data retention: {100*clean_data.shape[0]/features_data.shape[0]:.1f}%")
            print(f"  Clean segments: {len(clean_segments_indices) if clean_segments_indices is not None else 0}")
        else:
            print(f"  Final data shape: {clean_data.shape}")
        
        print(f"  Physiological features: {clean_data.shape[1]}")
        print(f"  Column order maintained: {len(column_names)} features")
        
        # Return processed data in memory
        return {
            'file_info': file_info,
            'clean_data': clean_data,
            'original_data': features_data,
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


def process_and_concatenate_subject_data(subject, available_files):
    """
    Process all physiological files for a subject and directly save concatenated feature data.
    
    This function processes all files in memory and creates concatenated outputs
    without saving individual files.
    
    Parameters
    ----------
    subject : str
        Subject ID (e.g., "14")
    available_files : list of dict
        List of file information dictionaries
        
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
            processed_data = process_single_file(file_info)
            
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
        output_dir = repo_root / "data" / "derivatives" / "physio" / f"sub-{subject}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Subject output directory: {output_dir}")
        
        # Save concatenated TFR data
        concat_tfr_filename = f"sub-{subject}_desc-physio_features_concatenated.npz"
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
            concat_columns_filename = f"sub-{subject}_desc-physio_columns_concatenated.tsv"
            concat_columns_path = output_dir / concat_columns_filename
            columns_df = pd.DataFrame({'column_name': column_names})
            columns_df.to_csv(concat_columns_path, sep='\t', index=False)
            print(f"✓ Column names saved: {concat_columns_path}")
        
        # Create comprehensive metadata for concatenated data
        concat_metadata = {
            "Description": "Concatenated time-frequency representation for entire subject",
            "Subject": subject,
            "Method": "Continuous physiological feature extraction (raw and processed)",
            "Channels": column_names,
            "NumberOfChannels": len(column_names),
            "NumberOfTimePoints": concatenated_tfr.shape[0],
            "DataShape": list(concatenated_tfr.shape),
            "Units": "arbitrary units (raw/processed)",
            "SamplingFrequency": float(sfreq) if sfreq else None,
            "SamplingFrequencyUnit": "Hz",
            "ColumnOrder": column_names,
            "RunBreakdown": all_run_info,
            "CleaningApplied": True,
            "GeneratedBy": {
                "Name": "physiology_features.py",
                "Description": "Continuous physiological feature extraction using NeuroKit2 and raw signals",
                "Version": "1.0"
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
        concat_metadata_filename = f"sub-{subject}_desc-physio_features_concatenated.json"
        concat_metadata_path = output_dir / concat_metadata_filename
        with open(concat_metadata_path, 'w') as f:
            json.dump(concat_metadata, f, indent=4)
        print(f"✓ Concatenated metadata saved: {concat_metadata_path}")
        
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


def combine_all_subjects_data(successful_subjects):
    """
    Combine concatenated physiological feature data from all successful subjects into single files.
    
    This function loads the concatenated data from each subject and combines them
    into final all_subs_* files, maintaining consistent ordering between data and indices.
    
    Parameters
    ----------
    successful_subjects : list
        List of subject IDs that were successfully processed
        
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
        
        # Counters for index adjustment across subjects
        cumulative_clean_timepoints = 0
        cumulative_original_timepoints = 0
        
        # Column names (should be consistent across subjects)
        column_names = None
        
        # Process each subject in order
        for subject_idx, subject in enumerate(successful_subjects):
            print(f"\n--- Loading data for subject {subject} ({subject_idx + 1}/{len(successful_subjects)}) ---")
            
            # Define paths for this subject's concatenated files
            subject_dir = repo_root / "data" / "derivatives" / "physio" / f"sub-{subject}"
            
            # Load TFR data
            tfr_file = subject_dir / f"sub-{subject}_desc-physio_features_concatenated.npz"
            if not tfr_file.exists():
                print(f"❌ TFR file not found for subject {subject}: {tfr_file}")
                return False
            
            subject_tfr_data = np.load(tfr_file)['arr_0']
            all_subjects_data.append(subject_tfr_data)
            print(f"✓ Loaded TFR data: {subject_tfr_data.shape}")
            
            # Load metadata
            metadata_file = subject_dir / f"sub-{subject}_desc-physio_features_concatenated.json"
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
                    columns_file = subject_dir / f"sub-{subject}_desc-physio_columns_concatenated.tsv"
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
        output_dir = repo_root / "data" / "derivatives" / "physio"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Combined files output directory: {output_dir}")
        
        # Save combined TFR data
        combined_tfr_filename = "all_subs_desc-physio_features.npz"
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
            combined_columns_filename = "all_subs_desc-physio_columns.tsv"
            combined_columns_path = output_dir / combined_columns_filename
            columns_df = pd.DataFrame({'column_name': column_names})
            columns_df.to_csv(combined_columns_path, sep='\t', index=False)
            print(f"✓ Combined column names saved: {combined_columns_path}")
        
        # Create comprehensive metadata for combined data
        combined_metadata = {
            "Description": "Combined time-frequency representation from all subjects",
            "Subjects": successful_subjects,
            "SubjectOrder": successful_subjects,
            "Method": "Continuous physiological feature extraction (raw and processed)",
            "Channels": column_names,
            "NumberOfChannels": len(column_names),
            "NumberOfTimePoints": combined_tfr.shape[0],
            "DataShape": list(combined_tfr.shape),
            "Units": "arbitrary units (raw/processed)",
            "SamplingFrequency": float(sfreq) if sfreq else None,
            "SamplingFrequencyUnit": "Hz",
            "ColumnOrder": column_names,
            "CleaningApplied": True,
            "GeneratedBy": {
                "Name": "physiology_features.py",
                "Description": "Continuous physiological feature extraction using NeuroKit2 and raw signals",
                "Version": "1.0"
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
        combined_metadata_filename = "all_subs_desc-physio_features.json"
        combined_metadata_path = output_dir / combined_metadata_filename
        with open(combined_metadata_path, 'w') as f:
            json.dump(combined_metadata, f, indent=4)
        print(f"✓ Combined metadata saved: {combined_metadata_path}")
        
        # Validation: Verify data and indices consistency
        print(f"\n=== VALIDATION ===")
        data_length = combined_tfr.shape[0]
        if combined_clean_indices is not None:
            max_clean_idx = combined_clean_indices.max()
            if max_clean_idx >= data_length:
                print(f"❌ Validation failed: max clean index ({max_clean_idx}) >= data length ({data_length})")
                return False
            else:
                print(f"✓ Clean indices validation passed: max index {max_clean_idx} < data length {data_length}")
        
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
    Main function to execute the physiological feature extraction pipeline for all available files of a subject.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Extract continuous physiological features (GSR, ECG, RESP, joystick_y) for all available files of a subject.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single subject
    python scripts/preprocessing/physiology_features.py --sub 14
    
    # Multiple subjects
    python scripts/preprocessing/physiology_features.py --sub 14 16 17 18
    
    # Multiple subjects with cross-subject concatenation
    python scripts/preprocessing/physiology_features.py --sub 14 16 17 18 --combine-all-subs
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
        '--combine-all-subs',
        action='store_true',
        help='Combine all subjects into single concatenated files (all_subs_*). Requires multiple subjects.'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    print("="*80)
    print("PHYSIOLOGICAL FEATURE EXTRACTION ANALYSIS")
    print("="*80)
    
    # Handle multiple subjects
    subjects = args.sub
    print(f"Subjects to process: {subjects}")
    print(f"Total subjects: {len(subjects)}")
    print(f"Combine all subjects: {args.combine_all_subs}")
    
    # Validate combine-all-subs flag
    if args.combine_all_subs and len(subjects) < 2:
        print(f"❌ Error: --combine-all-subs requires at least 2 subjects, but only {len(subjects)} provided")
        return 1
    
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
                # Step 1: Find all available physiological files for the subject
                available_files = find_available_eeg_files(subject)
                
                if not available_files:
                    print(f"❌ No physiological files found for subject {subject}")
                    failed_subjects.append(subject)
                    subject_results[subject] = {
                        'status': 'failed',
                        'reason': 'No physiological files found',
                        'files_found': 0
                    }
                    continue
                
                # Step 2: Check if output already exists (for skip-existing option)
                if args.combine_all_subs:
                    output_dir = repo_root / "data" / "derivatives" / "physio" / f"sub-{subject}"
                    concat_tfr_filename = f"sub-{subject}_desc-physio_features_concatenated.npz"
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
                processing_success = process_and_concatenate_subject_data(subject, available_files)
                
                if processing_success:
                    successful_subjects.append(subject)
                    output_dir = repo_root / "data" / "derivatives" / "physio" / f"sub-{subject}"
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
                combination_success = combine_all_subjects_data(successful_subjects)
                
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
        
        print(f"\nPhysiological Feature Extraction Configuration:")
        print(f"  Method: Continuous physiological feature extraction (raw and processed)")
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
        base_output_dir = repo_root / "data" / "derivatives" / "physio"
        print(f"  Base directory: {base_output_dir}")
        for subject in successful_subjects + skipped_subjects:
            print(f"  - sub-{subject}/")
            print(f"    └── sub-{subject}_desc-physio_features_concatenated.npz")
            print(f"    └── idx_data_sub-{subject}_concatenated.npz")
            print(f"    └── idx_data_OLD_timepoints_sub-{subject}_concatenated.npz")
            print(f"    └── sub-{subject}_desc-physio_columns_concatenated.tsv")
            print(f"    └── sub-{subject}_desc-physio_features_concatenated.json")
        
        if args.combine_all_subs and combination_success:
            output_dir = repo_root / "data" / "derivatives" / "physio"
            print(f"\nCombined files for ALL SUBJECTS:")
            print(f"  Location: {output_dir}")
            print(f"  - all_subs_desc-physio_features.npz")
            print(f"  - idx_data_all_subs.npz")
            print(f"  - idx_data_OLD_timepoints_all_subs.npz")
            print(f"  - all_subs_desc-physio_columns.tsv")
            print(f"  - all_subs_desc-physio_features.json")
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