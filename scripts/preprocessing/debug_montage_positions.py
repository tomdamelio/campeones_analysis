#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SIMPLIFIED MONTAGE DIAGNOSTIC SCRIPT
====================================

This script investigates why electrode positions may be missing or invalid
for interpolation in the CAMPEONES analysis pipeline.
"""

import os
import sys
from git import Repo
import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids

# Add src directory to Python path
repo = Repo(os.getcwd(), search_parent_directories=True)
repo_root = repo.git.rev_parse("--show-toplevel")
src_path = os.path.join(repo_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def analyze_montage_and_positions():
    """
    Simplified analysis of montage and electrode positions.
    """
    print("=== CAMPEONES MONTAGE POSITION DIAGNOSTIC ===")
    print("Investigating why electrode positions may be missing for interpolation")
    print("=" * 70)
    
    # Define file paths
    subject = "14"
    session = "vr"
    task = "01"
    acquisition = "b"
    run = "006"
    
    raw_data_folder = "data/raw"
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        acquisition=acquisition,
        run=run,
        datatype="eeg",
        suffix="eeg",
        extension=".vhdr",
        root=os.path.join(repo_root, raw_data_folder),
    )
    
    print(f"Analyzing file: {bids_path}")
    
    # 1. Load raw data
    print("\n1. LOADING RAW DATA")
    print("-" * 30)
    try:
        raw = read_raw_bids(bids_path)
        print(f"✓ Raw data loaded successfully")
        print(f"  - Channels: {len(raw.ch_names)}")
        print(f"  - Duration: {raw.times[-1]:.1f}s")
        print(f"  - Sampling rate: {raw.info['sfreq']} Hz")
    except Exception as e:
        print(f"✗ Error loading raw data: {e}")
        return
    
    # 2. Check original montage
    print("\n2. ORIGINAL MONTAGE STATUS")
    print("-" * 30)
    if raw.info['dig'] is not None:
        n_eeg_positions = sum(1 for d in raw.info['dig'] if d['kind'] == 3)
        print(f"✓ Original montage found: {n_eeg_positions} EEG positions")
    else:
        print("⚠️ No original montage found")
    
    # 3. Test montage files
    print("\n3. TESTING MONTAGE FILES")
    print("-" * 30)
    
    montage_files = [
        os.path.join(repo_root, "BC-32_FCz_modified.bvef"),
        os.path.join(repo_root, "data", "BC-32.bvef"),
        os.path.join(repo_root, "BC-32.bvef")
    ]
    
    working_montage = None
    
    for montage_file in montage_files:
        print(f"\nTesting: {montage_file}")
        if os.path.exists(montage_file):
            try:
                montage = mne.channels.read_custom_montage(montage_file)
                print(f"  ✓ Loaded: {len(montage.ch_names)} channels")
                
                # Test montage application
                raw_test = raw.copy()
                
                # Set channel types first (CRITICAL!)
                channel_type_mapping = {
                    'ECG': 'ecg', 'R_EYE': 'eog', 'L_EYE': 'eog',
                    'GSR': 'gsr', 'RESP': 'resp', 'AUDIO': 'misc',
                    'PHOTO': 'misc', 'X': 'misc', 'Y': 'misc', 'Z': 'misc',
                    'joystick_x': 'misc', 'joystick_y': 'misc'
                }
                
                existing_mapping = {ch: channel_type_mapping[ch] 
                                  for ch in channel_type_mapping 
                                  if ch in raw_test.ch_names}
                
                if existing_mapping:
                    raw_test.set_channel_types(existing_mapping)
                    print(f"  ✓ Channel types set for {len(existing_mapping)} channels")
                
                # Apply montage
                raw_test.set_montage(montage)
                print(f"  ✓ Montage applied successfully")
                working_montage = montage_file
                break
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        else:
            print(f"  ✗ File not found")
    
    if not working_montage:
        print("\n⚠️ No working montage found!")
        return
    
    # 4. Test bad channel scenarios
    print(f"\n4. BAD CHANNEL INTERPOLATION TEST")
    print("-" * 30)
    
    # Test the exact scenario from the error
    test_bad_channels = ['Fp1', 'O1', 'FC1', 'Cz']
    
    print(f"Testing bad channels: {test_bad_channels}")
    
    raw_test = raw.copy()
    
    # Set channel types
    existing_mapping = {ch: channel_type_mapping[ch] 
                      for ch in channel_type_mapping 
                      if ch in raw_test.ch_names}
    
    if existing_mapping:
        raw_test.set_channel_types(existing_mapping)
    
    # Apply montage
    montage = mne.channels.read_custom_montage(working_montage)
    raw_test.set_montage(montage)
    
    # Set bad channels
    raw_test.info['bads'] = test_bad_channels
    
    # Analyze each bad channel
    print("\nAnalyzing each bad channel:")
    montage_obj = raw_test.get_montage()
    eeg_picks = mne.pick_types(raw_test.info, eeg=True)
    eeg_ch_names = [raw_test.ch_names[i] for i in eeg_picks]
    
    for bad_ch in test_bad_channels:
        print(f"  {bad_ch}:", end=" ")
        
        if bad_ch not in raw_test.ch_names:
            print("NOT IN DATA ✗")
            continue
            
        if bad_ch not in eeg_ch_names:
            print("NOT EEG ⚠️")
            continue
            
        if montage_obj is None:
            print("NO MONTAGE ✗")
            continue
            
        if bad_ch not in montage_obj.ch_names:
            print("NOT IN MONTAGE ✗")
            continue
            
        try:
            pos = montage_obj.get_positions()['ch_pos'][bad_ch]
            if np.isnan(pos).any():
                print("NaN POSITION ✗")
            elif np.isinf(pos).any():
                print("INF POSITION ✗")
            elif np.allclose(pos, [0, 0, 0]):
                print("ORIGIN POSITION ⚠️")
            else:
                print(f"VALID POSITION ✓")
        except Exception as e:
            print(f"ERROR: {e}")
    
    # Test interpolation
    print(f"\n5. INTERPOLATION TEST")
    print("-" * 30)
    
    try:
        raw_interp = raw_test.copy().interpolate_bads(reset_bads=True)
        print(f"✓ Interpolation SUCCESSFUL!")
        print(f"  Original bads: {test_bad_channels}")
        print(f"  Final bads: {raw_interp.info['bads']}")
    except Exception as e:
        print(f"✗ Interpolation FAILED: {e}")
        
        # Try separating EEG vs non-EEG channels
        eeg_bads = [ch for ch in test_bad_channels if ch in eeg_ch_names]
        non_eeg_bads = [ch for ch in test_bad_channels if ch not in eeg_ch_names]
        
        print(f"\nTrying with only EEG channels:")
        print(f"  EEG bads: {eeg_bads}")
        print(f"  Non-EEG bads: {non_eeg_bads}")
        
        if eeg_bads:
            raw_test_eeg = raw_test.copy()
            raw_test_eeg.info['bads'] = eeg_bads
            
            try:
                raw_interp_eeg = raw_test_eeg.copy().interpolate_bads(reset_bads=True)
                print(f"  ✓ EEG-only interpolation SUCCESSFUL!")
            except Exception as e2:
                print(f"  ✗ EEG-only interpolation also failed: {e2}")
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    analyze_montage_and_positions() 