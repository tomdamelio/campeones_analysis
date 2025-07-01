#!/usr/bin/env python
"""
GLHMM (Gaussian Linear Hidden Markov Model) Analysis Script for  TFR Data

This script implements GLHMM analysis for  time-frequency representation (TFR) data from EEG recordings.
It loads preprocessed and cleaned TFR data (with 'bad' segments excluded), loads pre-generated segment indices,
validates data consistency, and trains GLHMM models.

Features:
- Loads clean TFR data from .npy files and column information from .tsv files
- Loads pre-generated segment indices (idx_data) from eeg_tfr.py processing
- Validates data consistency between TFR data and segment indices
- Preprocesses data (standardization) and applies PCA reduction for GLHMM requirements
- Initializes and trains GLHMM with specified number of states
- Provides model inspection including state means, covariances, and transition probabilities
- Supports analysis of concatenated clean segments from a single recording session

Data Structure:
- TFR data: Clean (cropped) time-frequency power values, excluding 'bad' segments
- Index data: Pre-computed segment boundaries mapping clean timepoints
- Validation: Ensures data integrity between TFR and index files

Usage:
    python scripts/glhmm/test_glhmm.py --subject 14 --session vr --task 01 --acquisition b --run 006
    python scripts/glhmm/test_glhmm.py --subject 14 --session vr --task 01 --acquisition b --run 006 --states 6
    python scripts/glhmm/test_glhmm.py --subject 14 --session vr --task 01 --acquisition b --run 006 --pca-components 30

"""

import os
import sys
import json
import pickle
import traceback
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add the repository root to Python path
script_path = Path(__file__).resolve()
repo_root = script_path
while not (repo_root / ".git").exists() and repo_root != repo_root.parent:
    repo_root = repo_root.parent

sys.path.append(str(repo_root))

# Import GLHMM - assuming it's installed
try:
    import glhmm
    from glhmm import glhmm, preproc, statistics, graphics
    print("✓ GLHMM library imported successfully")
except ImportError as e:
    print(f"❌ Error importing GLHMM: {e}")
    print("Please install GLHMM: pip install glhmm")
    sys.exit(1)


def load_tfr_data(subject, session, task, acquisition, run):
    """
    Load clean TFR data and column information for a specific recording session.
    
    This function loads clean TFR data that has already been processed by eeg_tfr.py
    to exclude 'bad' segments and concatenate clean segments.
    
    Parameters
    ----------
    subject : str
        Subject ID (e.g., "14")
    session : str
        Session ID (e.g., "vr")
    task : str
        Task ID (e.g., "01")
    acquisition : str
        Acquisition parameter (e.g., "b")
    run : str
        Run ID (e.g., "006")
        
    Returns
    -------
    data : np.ndarray
        Clean TFR data with shape (n_timepoints, n_features)
        Note: n_timepoints represents clean timepoints (bad segments excluded)
    column_names : list
        List of feature names (channel_frequency combinations)
    metadata : dict
        Metadata information from JSON file including cleaning details
    """
    print(f"\n=== LOADING CLEAN TFR DATA ===")
    print(f"Subject: {subject}, Session: {session}, Task: {task}, Acquisition: {acquisition}, Run: {run}")
    print(f"Note: Loading clean data with 'bad' segments already excluded")
    
    # Define TFR data directory
    trf_dir = repo_root / "data" / "derivatives" / "trf"
    
    # Create filenames based on BIDS convention
    base_filename = f"sub-{subject}_ses-{session}_task-{task}_acq-{acquisition}_run-{run}"
    
    # Load TFR data (.npy file)
    data_filename = f"{base_filename}_desc-morlet_tfr.npy"
    data_path = trf_dir / data_filename
    
    if not data_path.exists():
        raise FileNotFoundError(f"TFR data file not found: {data_path}")
    
    print(f"Loading TFR data from: {data_path}")
    data = np.load(data_path)
    print(f"✓ Data loaded with shape: {data.shape}")
    print(f"  Timepoints: {data.shape[0]}")
    print(f"  Features: {data.shape[1]}")
    
    # Load column names (.tsv file)
    columns_filename = f"{base_filename}_desc-morlet_columns.tsv"
    columns_path = trf_dir / columns_filename
    
    if not columns_path.exists():
        raise FileNotFoundError(f"Column names file not found: {columns_path}")
    
    print(f"Loading column names from: {columns_path}")
    columns_df = pd.read_csv(columns_path, sep='\t')
    column_names = columns_df['column_name'].tolist()
    print(f"✓ Column names loaded: {len(column_names)} features")
    print(f"  Example features: {column_names[:5]}{'...' if len(column_names) > 5 else ''}")
    
    # Load metadata (.json file)
    metadata_filename = f"{base_filename}_desc-morlet_tfr.json"
    metadata_path = trf_dir / metadata_filename
    
    metadata = {}
    if metadata_path.exists():
        print(f"Loading metadata from: {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✓ Metadata loaded")
        print(f"  Sampling frequency: {metadata.get('SamplingFrequency', 'N/A')} Hz")
        print(f"  Number of channels: {metadata.get('NumberOfChannels', 'N/A')}")
        print(f"  Number of frequencies: {metadata.get('NumberOfFrequencies', 'N/A')}")
        
        # Show cleaning information if available
        if metadata.get('CleaningApplied', False):
            original_timepoints = metadata.get('OriginalTimePoints', 'N/A')
            clean_timepoints = metadata.get('CleanTimePoints', 'N/A')
            data_retention = metadata.get('DataRetention', 'N/A')
            n_segments = metadata.get('NumberOfCleanSegments', 'N/A')
            
            print(f"  Cleaning information:")
            print(f"    Original timepoints: {original_timepoints:,}")
            print(f"    Clean timepoints: {clean_timepoints:,}")
            print(f"    Data retention: {data_retention:.1%}" if isinstance(data_retention, (int, float)) else f"    Data retention: {data_retention}")
            print(f"    Number of clean segments: {n_segments}")
        else:
            print(f"  ⚠️  No cleaning information found in metadata")
    else:
        print(f"⚠️  Metadata file not found: {metadata_path}")
    
    # Verify data consistency
    if len(column_names) != data.shape[1]:
        raise ValueError(f"Mismatch: {len(column_names)} column names vs {data.shape[1]} features")
    
    print(f"✓ Data consistency verified")
    
    return data, column_names, metadata


def load_idx_data(subject, session, task, acquisition, run):
    """
    Load pre-generated index data for clean TFR segments.
    
    Parameters
    ----------
    subject : str
        Subject ID (e.g., "14")
    session : str
        Session ID (e.g., "vr")
    task : str
        Task ID (e.g., "01")
    acquisition : str
        Acquisition parameter (e.g., "b")
    run : str
        Run ID (e.g., "006")
        
    Returns
    -------
    idx_data : np.ndarray
        Clean segment indices with shape (n_segments, 2)
        Each row contains [start_timepoint, end_timepoint] for a clean segment
    """
    print(f"\n=== LOADING INDEX DATA ===")
    
    # Define TFR data directory
    trf_dir = repo_root / "data" / "derivatives" / "trf"
    
    # Create filename for index data
    base_filename = f"sub-{subject}_ses-{session}_task-{task}_acq-{acquisition}_run-{run}"
    idx_filename = f"idx_data_{base_filename}.npy"
    idx_path = trf_dir / idx_filename
    
    if not idx_path.exists():
        raise FileNotFoundError(f"Index data file not found: {idx_path}")
    
    print(f"Loading index data from: {idx_path}")
    idx_data = np.load(idx_path)
    
    print(f"✓ Index data loaded successfully")
    print(f"  Index shape: {idx_data.shape}")
    print(f"  Number of clean segments: {idx_data.shape[0]}")
    
    # Show segment details
    total_timepoints = 0
    for i, (start, end) in enumerate(idx_data):
        segment_length = end - start
        total_timepoints += segment_length
        print(f"  Segment {i+1}: [{start:6d}-{end:6d}] ({segment_length:6d} timepoints)")
    
    print(f"  Total timepoints across segments: {total_timepoints:,}")
    
    return idx_data


def prepare_data_for_glhmm(data, subject, session, task, acquisition, run, test_mode=False):
    """
    Prepare clean TFR data for GLHMM analysis.
    
    This involves:
    1. Loading pre-generated segment indices from eeg_tfr.py processing
    2. Validating data consistency between TFR data and index files
    3. Ensuring data is in correct format (n_timepoints, n_features)
    4. Adjusting indices if in test mode
    
    Parameters
    ----------
    data : np.ndarray
        Clean TFR data with shape (n_timepoints, n_features)
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
    test_mode : bool
        If True, adjust indices for test data (first 500 timepoints)
        
    Returns
    -------
    data : np.ndarray
        Data ready for GLHMM (same as input)
    idx_data : np.ndarray
        Clean segment indices with shape (n_segments, 2), adjusted for test mode if needed
    """
    print(f"\n=== PREPARING DATA FOR GLHMM ===")
    print(f"Input clean TFR data shape: {data.shape}")
    if test_mode:
        print(f"🧪 Test mode: Data truncated to first 500 timepoints")
    
    # Load pre-generated index data
    idx_data = load_idx_data(subject, session, task, acquisition, run)
    
    # Adjust indices for test mode
    if test_mode:
        print(f"\n🧪 Adjusting indices for test mode...")
        original_idx_shape = idx_data.shape
        
        # Find segments that fit within first 500 timepoints
        data_length = data.shape[0]  # Should be 500 in test mode
        valid_segments = []
        
        for i, (start, end) in enumerate(idx_data):
            if start < data_length:
                # Clip end to data length if necessary
                adjusted_end = min(end, data_length)
                if adjusted_end > start:  # Only keep if segment has positive length
                    valid_segments.append([start, adjusted_end])
                    print(f"   Segment {i+1}: [{start:6d}-{end:6d}] -> [{start:6d}-{adjusted_end:6d}]")
                else:
                    print(f"   Segment {i+1}: [{start:6d}-{end:6d}] -> SKIPPED (no data)")
            else:
                print(f"   Segment {i+1}: [{start:6d}-{end:6d}] -> SKIPPED (beyond test data)")
        
        if not valid_segments:
            raise ValueError("No valid segments found in test mode (first 500 timepoints)")
        
        idx_data = np.array(valid_segments)
        print(f"   Original segments: {original_idx_shape[0]}")
        print(f"   Valid test segments: {idx_data.shape[0]}")
    
    # Validate data consistency
    print(f"\n=== VALIDATING DATA CONSISTENCY ===")
    
    # Check that the last timepoint of the last segment matches data length
    if len(idx_data) > 0:
        last_segment = idx_data[-1]
        last_timepoint = last_segment[1]  # End of last segment
        data_length = data.shape[0]
        
        print(f"Last segment end timepoint: {last_timepoint}")
        print(f"TFR data length: {data_length}")
        
        if last_timepoint == data_length:
            print(f"✓ Index consistency validation PASSED")
            print(f"  Last segment endpoint matches TFR data length")
        else:
            if test_mode:
                print(f"⚠️  Index consistency validation INFO: Test mode truncation")
                print(f"  Last segment endpoint ({last_timepoint}) may not match TFR data length ({data_length})")
                print(f"  This is expected in test mode")
            else:
                raise ValueError(f"Index consistency validation FAILED: "
                               f"Last segment endpoint ({last_timepoint}) does not match "
                               f"TFR data length ({data_length})")
    else:
        raise ValueError("No segments found in index data")
    
    # Additional validation: check that segments are valid and fit within data
    total_reconstructed_length = 0
    for i, (start, end) in enumerate(idx_data):
        segment_length = end - start
        
        # Check that segment boundaries are valid
        if start < 0 or end <= start:
            raise ValueError(f"Invalid segment {i+1}: [{start}, {end}]")
        
        # Check that segment fits within data
        if end > data.shape[0]:
            raise ValueError(f"Segment {i+1} extends beyond data: [{start}, {end}] vs data length {data.shape[0]}")
        
        # Check that this segment follows immediately after the previous one (unless in test mode)
        if not test_mode:
            if i == 0:
                if start != 0:
                    raise ValueError(f"First segment should start at 0, but starts at {start}")
            else:
                prev_end = idx_data[i-1][1]
                if start != prev_end:
                    raise ValueError(f"Gap found between segment {i} and {i+1}: "
                                   f"segment {i} ends at {prev_end}, segment {i+1} starts at {start}")
        
        total_reconstructed_length += segment_length
    
    # Final check that total length matches (relaxed for test mode)
    if not test_mode and total_reconstructed_length != data_length:
        raise ValueError(f"Total segment length ({total_reconstructed_length}) "
                        f"does not match TFR data length ({data_length})")
    
    print(f"✓ All consistency checks passed")
    print(f"  Number of segments: {len(idx_data)}")
    print(f"  Total timepoints in segments: {total_reconstructed_length:,}")
    if test_mode:
        print(f"  Test mode: Segments adjusted for first {data.shape[0]} timepoints")
    else:
        print(f"  Segments are consecutive and complete")
    
    # Data is already in correct format (n_timepoints, n_features)
    print(f"✓ Data format verified: (n_timepoints={data.shape[0]}, n_features={data.shape[1]})")
    
    return data, idx_data


def preprocess_data_for_glhmm(data, idx_data, pca_components=50):
    """
    Preprocess data for GLHMM using standard preprocessing pipeline.
    
    Parameters
    ----------
    data : np.ndarray
        TFR data with shape (n_timepoints, n_features)
    idx_data : np.ndarray
        Session indices with shape (n_sessions, 2)
        
    Returns
    -------
    processed_data : np.ndarray
        Preprocessed data
    scaler : object
        Scaler object for potential inverse transformation
    log : dict
        Preprocessing log
    """
    print(f"\n=== PREPROCESSING DATA ===")
    print(f"Input data shape: {data.shape}")
    print(f"Session indices shape: {idx_data.shape}")
    print(f"PCA components requested: {pca_components}")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(data)):
        print(f"⚠️  Found {np.sum(np.isnan(data))} NaN values in data")
    if np.any(np.isinf(data)):
        print(f"⚠️  Found {np.sum(np.isinf(data))} infinite values in data")
    
    # Print data statistics before preprocessing
    print(f"Data statistics before preprocessing:")
    print(f"  Mean: {np.mean(data):.6f}")
    print(f"  Std: {np.std(data):.6f}")
    print(f"  Min: {np.min(data):.6f}")
    print(f"  Max: {np.max(data):.6f}")
    
    # Preprocess data using GLHMM's preprocessing pipeline
    # We use standardization (mean=0, std=1) as the default option
    try:
        processed_data, scaler, log = preproc.preprocess_data(
            data,
            idx_data,
            standardise=True,
            pca=pca_components
        )
        
        print(f"✓ Data preprocessing completed using GLHMM preproc.preprocess_data()")
        print(f"  Output data shape: {processed_data.shape}")
        print(f"  PCA reduction: {data.shape[1]} → {processed_data.shape[1]} features")
        
    except Exception as e:
        print(f"❌ Error in GLHMM preprocessing: {e}")
        print("Falling back to manual standardization...")
        raise
    
    # Print data statistics after preprocessing
    print(f"Data statistics after preprocessing:")
    print(f"  Mean: {np.mean(processed_data):.6f}")
    print(f"  Std: {np.std(processed_data):.6f}")
    print(f"  Min: {np.min(processed_data):.6f}")
    print(f"  Max: {np.max(processed_data):.6f}")
    
    print(f"✓ Preprocessing completed successfully")
    
    return processed_data, scaler, log


def initialize_and_train_hmm(data, idx_data, K=10, preproclog=None, random_seed=42):
    """
    Initialize and train a Gaussian HMM using GLHMM.
    
    Parameters
    ----------
    data : np.ndarray
        Preprocessed data with shape (n_timepoints, n_features)
    idx_data : np.ndarray
        Session indices with shape (n_sessions, 2)
    K : int
        Number of states (default: 4)
    preproclog : dict
        Preprocessing log from preprocess_data
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    hmm : glhmm.glhmm
        Trained HMM model
    """
    print(f"\n=== INITIALIZING AND TRAINING HMM ===")
    print(f"Number of states (K): {K}")
    print(f"Data shape: {data.shape}")
    print(f"Session indices shape: {idx_data.shape}")
    print(f"Random seed: {random_seed}")
    
    # Initialize HMM
    # We do not want to model an interaction between two sets of variables, so we set model_beta='no'
    print("Initializing GLHMM...")
    hmm = glhmm.glhmm(
        model_beta='no',  # No interaction between variables
        K=K,              # Number of states
        covtype='sharedfull',   # Full covariance matrix, "sharedfull" is another option
        preproclogY=preproclog  # Preprocessing log
    )
    
    print("✓ HMM initialized")
    
    # Train the HMM
    print("\nTraining HMM...")
    print("This may take several minutes depending on data size and number of states...")
    
    try:
        hmm.train(X=None, Y=data, indices=idx_data)
        print("✓ HMM training completed successfully")
            
    except Exception as e:
        print(f"❌ Error during HMM training: {e}")
        traceback.print_exc()
        raise
    
    return hmm

# SEGUIR DESDE ACA
# - Extender este codigo para que cuando le pase un participante completo,
#   sin especificar task, session, etc, concatene todos los datos de ese participante
#   y los procese juntos.
# - Dejar eso corriendo en Arete (si fallo dejarlo corriendo localmente)
def inspect_hmm_model(hmm, data, idx_data, column_names, original_n_features, save_dir=None, test_mode=False):
    """
    Inspect the trained HMM model by examining states, transitions, and dynamics.
    Also creates and saves visualizations.
    
    Parameters
    ----------
    hmm : glhmm.glhmm
        Trained HMM model
    data : np.ndarray
        Data used for training (after PCA if applied)
    column_names : list
        Names of the original features (before PCA)
    original_n_features : int
        Number of original features (before PCA)
    save_dir : Path, optional
        Directory to save plots
    test_mode : bool
        If True, add "_test" suffix to all output files
    """
    print(f"\n=== INSPECTING HMM MODEL ===")
    
    if save_dir is None:
        save_dir = repo_root / "data" / "derivatives" / "glhmm_results"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create suffix for test mode
    test_suffix = "_test" if test_mode else ""
    
    # Extract model parameters
    K = hmm.hyperparameters["K"]  # Number of states
    n_features_processed = data.shape[1]    # Number of features after PCA
    
    print(f"Model summary:")
    print(f"  Number of states: {K}")
    print(f"  Number of features (processed/PCA): {n_features_processed}")
    print(f"  Number of features (original space): {original_n_features}")
    print(f"  Number of timepoints: {data.shape[0]}")
    print(f"  Note: GLHMM returns results in original feature space (before PCA)")
    if test_mode:
        print(f"  🧪 Test mode: Output files will include '_test' suffix")
    
    # 1. Initial state probabilities
    print(f"\n1. Initial State Probabilities:")
    init_state_probs = hmm.Pi.copy()
    for k in range(K):
        print(f"  State {k+1}: {init_state_probs[k]:.4f}")
    
    # 2. Transition probability matrix
    print(f"\n2. Transition Probability Matrix:")
    # Use GLHMM's proper method to get transition probabilities
    TP = hmm.P.copy()  # the transition probability matrix [K, K]
    transition_probs = TP  # Keep for compatibility with rest of code
    
    print("   From\\To  ", end="")
    for k in range(K):
        print(f"State{k+1:2d}", end="  ")
    print()
    
    for i in range(K):
        print(f"   State {i+1:2d}  ", end="")
        for j in range(K):
            print(f"{TP[i, j]:7.4f}", end="  ")
        print()
        
    # Plot transition probabilities using GLHMM tutorial style
    plt.figure(figsize=(14, 6))
    
    # Define colormap
    cmap = 'Blues'
    
    # Plot 1: Original Transition Probabilities
    plt.subplot(1, 2, 1)
    plt.imshow(TP, cmap=cmap, interpolation='nearest')
    title_text = 'Transition Probabilities'
    if test_mode:
        title_text += '\n(Test Mode)'
    plt.title(title_text)
    plt.xlabel('To State')
    plt.ylabel('From State')
    
    # Add state labels
    state_labels = [str(i+1) for i in range(K)]
    plt.xticks(range(K), state_labels)
    plt.yticks(range(K), state_labels)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # Plot 2: Transition Probabilities without Self-Transitions
    plt.subplot(1, 2, 2)
    TP_noself = TP - np.diag(np.diag(TP))  # Remove self-transitions
    TP_noself2 = TP_noself / TP_noself.sum(axis=1, keepdims=True)  # Normalize probabilities
    
    plt.imshow(TP_noself2, cmap=cmap, interpolation='nearest')
    title_text = 'Transition Probabilities\nwithout Self-Transitions'
    if test_mode:
        title_text = 'Transition Probabilities\nwithout Self-Transitions\n(Test Mode)'
    plt.title(title_text)
    plt.xlabel('To State')
    plt.ylabel('From State')
    
    # Add state labels
    plt.xticks(range(K), state_labels)
    plt.yticks(range(K), state_labels)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.tight_layout()  # Adjust layout for better spacing
    
    transition_plot_path = save_dir / f'transition_probabilities{test_suffix}.png'
    plt.savefig(transition_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Transition probability plot saved: {transition_plot_path}")
    
    # Additional analysis: print self-transition vs non-self-transition statistics
    self_transitions = np.diag(TP)
    avg_self_transition = np.mean(self_transitions)
    avg_non_self_transition = np.mean(TP_noself2[TP_noself2 > 0])
    
    print(f"   Transition analysis:")
    print(f"     Average self-transition probability: {avg_self_transition:.4f}")
    print(f"     Average non-self-transition probability: {avg_non_self_transition:.4f}")
    print(f"     Ratio (self/non-self): {avg_self_transition/avg_non_self_transition:.2f}")
    
    # Show strongest non-self transitions
    TP_noself_flat = TP_noself2.flatten()
    strongest_indices = np.argsort(TP_noself_flat)[-3:]  # Top 3 non-self transitions
    print(f"   Strongest non-self transitions:")
    for idx in reversed(strongest_indices):
        if TP_noself_flat[idx] > 0:
            from_state = idx // K + 1
            to_state = idx % K + 1
            prob = TP_noself_flat[idx]
            print(f"     State {from_state} → State {to_state}: {prob:.4f}")
    
    # 3. State means: Time-varying amplitude patterns
    print(f"\n3. State Means: Time-varying amplitude patterns")
    try:
        # The state means can be interpreted as time-varying patterns of amplitude (relative to the baseline)
        K = hmm.hyperparameters["K"]  # the number of states
        n_channels = data.shape[1]  # the number of parcels/channels (after PCA)
        n_features = original_n_features  # the number of original features
        
        mu = np.zeros(shape=(n_features, K))
        mu = hmm.get_means()  # the state means in the shape (no. features, no. states)
        
        print(f"   State means shape: {mu.shape}")
        print(f"   Number of states: {K}")
        print(f"   Number of original features: {n_features}")
        print(f"   ✓ State means retrieved in original feature space")
        
        # Show sample of state means
        n_show = min(5, mu.shape[0])
        print(f"   Sample state means (first {n_show} features):")
        print("   Feature      ", end="")
        for k in range(K):
            print(f"State{k+1:2d}", end="    ")
        print()
        
        for i in range(n_show):
            if i < len(column_names):
                feature_name = column_names[i][:12]  # Truncate long names
            else:
                feature_name = f"Feature_{i+1}"[:12]
            print(f"   {feature_name:12s} ", end="")
            for k in range(K):
                print(f"{mu[i, k]:7.4f}", end="  ")
            print()
        
        if mu.shape[0] > n_show:
            print(f"   ... and {mu.shape[0] - n_show} more features")
            
        # Plot state mean activation following GLHMM tutorial style
        cmap = "coolwarm"
        
        # Calculate appropriate figure width based on number of states
        fig_width = max(16, K * 2.5)  # Minimum 16 inches, then 2.5 inches per state
        fig_height = 16  # Fixed height
        
        plt.figure(figsize=(fig_width, fig_height))
        plt.imshow(mu, cmap=cmap, interpolation="none")
        plt.colorbar(label='Activation Level')  # Label for color bar
        title_text = "State mean activation"
        if test_mode:
            title_text += " (Test Mode)"
        plt.title(title_text)
        plt.xticks(np.arange(K), np.arange(1, K+1))
        plt.gca().set_xlabel('State')
        plt.gca().set_ylabel('Features')
        plt.tight_layout()  # Adjust layout for better spacing
        
        means_plot_path = save_dir / f'state_means{test_suffix}.png'
        plt.savefig(means_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ State means plot saved: {means_plot_path}")
        
        # Assign state_means for use in later sections
        state_means = mu
            
    except Exception as e:
        print(f"   ❌ Error getting state means: {e}")
        print(f"   This might be due to a dimensionality mismatch between model and data")
        traceback.print_exc()
        mu = None
        state_means = mu  # Keep compatibility with rest of code
    
    # 4. State covariances: Time-varying functional connectivity
    print(f"\n4. State Covariances: Time-varying functional connectivity")
    try:
        # Only show if state_means worked
        if state_means is not None:
            # The state covariances represent the time-varying functional connectivity patterns
            n_channels = state_means.shape[0]  # number of features/channels
            Sigma = np.zeros(shape=(n_channels, n_channels, K))
            for k in range(K):
                Sigma[:, :, k] = hmm.get_covariance_matrix(k=k)  # the state covariance matrices in the shape (no. features, no. features, no. states)
            
            print(f"   State covariances shape: {Sigma.shape}")
            print(f"   Number of features: {n_channels}")
            print(f"   Number of states: {K}")
            print(f"   ✓ State covariance matrices extracted successfully")
            
            # Plot the covariance (i.e., functional connectivity) of each state
            # These are square matrices showing the brain region by brain region functional connectivity patterns
            cmap = "coolwarm"
            
            # Determine subplot arrangement based on number of states
            if K <= 4:
                n_rows, n_cols = 2, 2
                figsize = (10, 8)
            elif K <= 6:
                n_rows, n_cols = 2, 3
                figsize = (15, 8)
            elif K <= 9:
                n_rows, n_cols = 3, 3
                figsize = (15, 12)
            else:
                n_rows, n_cols = 4, 3
                figsize = (15, 16)
            
            plt.figure(figsize=figsize)
            for k in range(K):
                plt.subplot(n_rows, n_cols, k+1)
                plt.imshow(Sigma[:, :, k], cmap=cmap)
                plt.xlabel('Features')
                plt.ylabel('Features')
                plt.colorbar()
                title_text = "State covariance\nstate #%s" % (k+1)
                if test_mode:
                    title_text += " (Test)"
                plt.title(title_text)
            
            plt.subplots_adjust(hspace=0.7, wspace=0.8)
            
            covariances_plot_path = save_dir / f'state_covariances{test_suffix}.png'
            plt.savefig(covariances_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ State covariances plot saved: {covariances_plot_path}")
            
            # Assign for compatibility with rest of code
            state_covariances = Sigma
            
        else:
            print(f"   ⚠️  Skipping covariance display due to state means error")
            state_covariances = None
            
    except Exception as e:
        print(f"   ❌ Error getting state covariances: {e}")
        traceback.print_exc()
        state_covariances = None
    
    # 5. Viterbi path (most likely state sequence)
    print(f"\n5. State Sequence Analysis:")
    try:
        # Compute Viterbi path
        viterbi_path = hmm.decode(X=None, Y=data, indices=idx_data, viterbi=True)
        print(f"   Viterbi path computed for {len(viterbi_path)} timepoints")
        print(f"   Viterbi path shape: {viterbi_path.shape}")
        print(f"   Viterbi path dtype: {viterbi_path.dtype}")
        
        # Handle different possible shapes of viterbi_path
        if viterbi_path.ndim == 2:
            # If 2D, likely (n_timepoints, n_states) - convert to state indices
            viterbi_states = np.argmax(viterbi_path, axis=1)
            print(f"   Converted 2D Viterbi path to state indices")
        elif viterbi_path.ndim == 1:
            # If 1D, should already be state indices
            viterbi_states = viterbi_path
            print(f"   Using 1D Viterbi path as state indices")
        else:
            raise ValueError(f"Unexpected Viterbi path shape: {viterbi_path.shape}")
        
        print(f"   State sequence shape: {viterbi_states.shape}")
        
        # Count state occupancy
        unique_states, counts = np.unique(viterbi_states, return_counts=True)
        print("   State occupancy:")
        for state, count in zip(unique_states, counts):
            if isinstance(state, (int, np.integer)):
                state_num = int(state) + 1  # Convert to 1-based indexing
            else:
                state_num = state + 1
            percentage = (count / len(viterbi_states)) * 100
            print(f"     State {state_num}: {count:6d} timepoints ({percentage:5.1f}%)")
        
        # Show first 20 states as example
        print(f"   First 20 states: {viterbi_states[:20]}")
        
        # Plot Viterbi path
        vpath_plot_path = save_dir / f'viterbi_path{test_suffix}.png'
        try:
            graphics.plot_vpath(viterbi_path, 
                              idx_data=idx_data,
                              title="Viterbi Path Analysis" + (" (Test Mode)" if test_mode else ""),
                              save_path=vpath_plot_path)
            print(f"   ✓ Viterbi path plot saved to: {vpath_plot_path}")
        except (AttributeError, ImportError) as e:
            print(f"   ⚠️  graphics.plot_vpath not available: {e}")
            
    except Exception as e:
        print(f"   ⚠️  Could not compute Viterbi path: {e}")
        traceback.print_exc()
    
    return {
        'K': K,
        'init_state_probs': init_state_probs,
        'transition_probs': transition_probs,
        'state_means': state_means,
        'state_covariances': state_covariances,
    }


def save_model_results(hmm, model_params, save_dir=None, test_mode=False):
    """
    Save model results and parameters to files.
    
    Parameters
    ----------
    hmm : glhmm.glhmm
        Trained HMM model
    model_params : dict
        Extracted model parameters
    save_dir : Path, optional
        Directory to save results
    test_mode : bool
        If True, add "_test" suffix to all output files
    """
    print(f"\n=== SAVING MODEL RESULTS ===")
    
    if save_dir is None:
        save_dir = repo_root / "data" / "derivatives" / "glhmm_results"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create suffix for test mode
    test_suffix = "_test" if test_mode else ""
    if test_mode:
        print(f"🧪 Test mode: All result files will include '_test' suffix")
    
    # Save transition probabilities
    transition_path = save_dir / f'transition_probabilities{test_suffix}.npy'
    np.save(transition_path, model_params['transition_probs'])
    print(f"✓ Transition probabilities saved: {transition_path}")
    
    # Save state means (if available)
    if model_params['state_means'] is not None:
        means_path = save_dir / f'state_means{test_suffix}.npy'
        np.save(means_path, model_params['state_means'])
        print(f"✓ State means saved: {means_path}")
    else:
        print(f"⚠️  State means not available (model inspection error)")
    
    # Save state covariances (if available)
    if model_params['state_covariances'] is not None:
        covariances_path = save_dir / f'state_covariances{test_suffix}.npy'
        np.save(covariances_path, model_params['state_covariances'])
        print(f"✓ State covariances saved: {covariances_path}")
    else:
        print(f"⚠️  State covariances not available (model inspection error)")
    
    # Save initial state probabilities
    init_probs_path = save_dir / f'initial_state_probabilities{test_suffix}.npy'
    np.save(init_probs_path, model_params['init_state_probs'])
    print(f"✓ Initial state probabilities saved: {init_probs_path}")
    
    
    # Save model summary as JSON
    
    summary = {
        'hyperparameters': hmm.hyperparameters,
        'number_of_states': model_params['K'],
        'data_shape': hmm.hyperparameters.get('data_shape', 'unknown'),
        'model_type': 'Gaussian HMM',
        'covariance_type': 'sharedfull',
        'pca_components': model_params.get('pca_components', None),
        'test_mode': test_mode,
    }
    
    summary_path = save_dir / f'model_summary{test_suffix}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4, default=str)
    print(f"✓ Model summary saved: {summary_path}")


def main():
    """
    Main function to execute GLHMM analysis pipeline.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="GLHMM analysis for EEG time-frequency data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train new model
    python scripts/glhmm/test_glhmm.py --subject 14 --session vr --task 01 --acquisition b --run 006
    python scripts/glhmm/test_glhmm.py --subject 14 --session vr --task 01 --acquisition b --run 006 --states 6
    python scripts/glhmm/test_glhmm.py --subject 14 --session vr --task 01 --acquisition b --run 006 --states 4 --seed 42
    python scripts/glhmm/test_glhmm.py --subject 14 --session vr --task 01 --acquisition b --run 006 --pca-components 30

    # Quick testing with only first 500 timepoints
    python scripts/glhmm/test_glhmm.py --subject 14 --session vr --task 01 --acquisition b --run 006 --test
    python scripts/glhmm/test_glhmm.py --subject 14 --session vr --task 01 --acquisition b --run 006 --test --states 5
        """
    )
    
    parser.add_argument('--subject', type=str, required=True,
                       help='Subject ID (e.g., 14)')
    parser.add_argument('--session', type=str, default='vr',
                       help='Session ID (e.g., vr)')
    parser.add_argument('--task', type=str, required=True,
                       help='Task ID (e.g., 01)')
    parser.add_argument('--acquisition', type=str, required=True,
                       help='Acquisition parameter (e.g., b)')
    parser.add_argument('--run', type=str, required=True,
                       help='Run ID (e.g., 006)')
    parser.add_argument('--states', '-K', type=int, default=10,
                       help='Number of HMM states (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--pca-components', type=int, default=50,
                       help='Number of PCA components to keep (default: 50)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip creating visualization plots')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: use only first 500 timepoints for quick testing')
    
    # Parse arguments
    args = parser.parse_args()
    
    print("="*80)
    print("GLHMM ANALYSIS FOR EEG TIME-FREQUENCY DATA")
    print("="*80)
    print(f"Subject: {args.subject}")
    print(f"Session: {args.session}")
    print(f"Task: {args.task}")
    print(f"Acquisition: {args.acquisition}")
    print(f"Run: {args.run}")
    print(f"Number of states: {args.states}")
    print(f"Random seed: {args.seed}")
    print(f"PCA components: {args.pca_components}")
    print(f"Test mode: {args.test}")
    
    try:        
        # Step 1: Load clean TFR data
        data, column_names, _ = load_tfr_data(
            args.subject, args.session, args.task, args.acquisition, args.run
        )
        
        # Apply test mode if requested
        if args.test:
            original_shape = data.shape
            data = data[:500]  # Use only first 500 timepoints
            print(f"\n🧪 TEST MODE ACTIVE")
            print(f"   Original data shape: {original_shape}")
            print(f"   Test data shape: {data.shape}")
            print(f"   Using only first 500 timepoints for quick testing")
        
        # Step 2: Prepare data for GLHMM (load indices and validate consistency)
        data, idx_data = prepare_data_for_glhmm(
            data, args.subject, args.session, args.task, args.acquisition, args.run, test_mode=args.test
        )
        
        # Step 3: Preprocess data
        processed_data, _, preproclog = preprocess_data_for_glhmm(
            data, idx_data, pca_components=args.pca_components
        )
        
        print(f"\n📊 Data Pipeline Summary:")
        print(f"  Original features: {data.shape[1]}")
        print(f"  PCA components: {processed_data.shape[1]}")
        print(f"  Training data shape: {processed_data.shape}")
        print(f"  Note: Model results will be in original {data.shape[1]}-feature space")
        
        print(f"✓ Data ready for training")
        
        # Step 4: Initialize and train HMM
        model_save_dir = repo_root / "data" / "derivatives" / "glhmm_results"
        model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 5: Train new model
        hmm = initialize_and_train_hmm(
            processed_data, idx_data, K=args.states, 
            preproclog=preproclog, random_seed=args.seed
        )
        
        # Step 6: Save the trained model immediately to avoid re-training
        test_suffix = "_test" if args.test else ""
        model_path = model_save_dir / f"hmm_model_sub-{args.subject}_run-{args.run}_states-{args.states}{test_suffix}.pkl"
        
        print(f"\n💾 Saving trained model...")
        with open(model_path, 'wb') as f:
            pickle.dump(hmm, f)
        print(f"✓ Model saved: {model_path}")
        if not args.test:
            print(f"   You can manually reload this model if needed")
        
        # Step 7: Inspect model
        model_params = inspect_hmm_model(hmm, processed_data, idx_data, column_names, data.shape[1], test_mode=args.test)
        
        # Add PCA information to model_params
        model_params['pca_components'] = args.pca_components
        model_params['original_column_names'] = column_names
        model_params['test_mode'] = args.test
                    
        save_model_results(hmm, model_params, test_mode=args.test)
        
        print("\n" + "="*80)
        print("✓ GLHMM ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Results saved in: {repo_root / 'data' / 'derivatives' / 'glhmm_results'}")
        print(f"\nData processing summary:")
        print(f"  ✓ Clean TFR data loaded (bad segments excluded)")
        if args.test:
            print(f"  ✓ TEST MODE: Used only first 500 timepoints")
        print(f"  ✓ Segment indices validated for consistency")
        print(f"  ✓ GLHMM trained on concatenated clean segments")
        print(f"  ✓ PCA dimensionality reduction applied ({args.pca_components} components)")
        print(f"  ✓ Model with {args.states} states successfully trained")
        
    except Exception as e:
        print(f"\n❌ ERROR in GLHMM analysis: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
