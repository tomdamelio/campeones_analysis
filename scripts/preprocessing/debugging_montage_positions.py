#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MONTAGE POSITION DIAGNOSTIC SCRIPT
==================================

This script investigates why electrode positions may be missing or invalid
for interpolation in the CAMPEONES analysis pipeline.

Author: AI Assistant
Date: 2024
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
    Comprehensive analysis of montage and electrode positions.
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
    
    # 2. Analyze original montage status
    print("\n2. ORIGINAL MONTAGE STATUS")
    print("-" * 30)
    if raw.info['dig'] is not None:
        print("✓ Original montage/digitization found in raw data")
        dig_counts = {}
        for d in raw.info['dig']:
            kind = d['kind']
            if kind == 1:
                dig_counts['LPA'] = dig_counts.get('LPA', 0) + 1
            elif kind == 2: 
                dig_counts['RPA'] = dig_counts.get('RPA', 0) + 1
            elif kind == 3:
                dig_counts['EEG'] = dig_counts.get('EEG', 0) + 1
            elif kind == 4:
                dig_counts['Head'] = dig_counts.get('Head', 0) + 1
            else:
                dig_counts[f'Kind_{kind}'] = dig_counts.get(f'Kind_{kind}', 0) + 1
        
        print(f"  Original digitization points: {dig_counts}")
        
        # Check if EEG positions are valid
        n_eeg_positions = sum(1 for d in raw.info['dig'] if d['kind'] == 3)
        eeg_channels = [ch for ch in raw.ch_names if 'EEG' in str(mne.io.pick.channel_type(raw.info, raw.ch_names.index(ch)))]
        print(f"  EEG channels in data: {len(eeg_channels)}")
        print(f"  EEG positions in dig: {n_eeg_positions}")
        
        if n_eeg_positions > 0:
            print("  ✓ Original EEG positions available")
        else:
            print("  ⚠️ No EEG positions in original montage")
    else:
        print("⚠️ No original montage/digitization in raw data")
    
    # 3. Test montage files
    print("\n3. TESTING MONTAGE FILES")
    print("-" * 30)
    
    montage_files = [
        os.path.join(repo_root, "BC-32_FCz_modified.bvef"),
        os.path.join(repo_root, "data", "BC-32.bvef"),
        os.path.join(repo_root, "BC-32.bvef")
    ]
    
    for montage_file in montage_files:
        print(f"\nTesting: {montage_file}")
        if os.path.exists(montage_file):
            print(f"  ✓ File exists")
            try:
                montage = mne.channels.read_custom_montage(montage_file)
                print(f"  ✓ Montage loaded successfully")
                print(f"  - Channels in montage: {len(montage.ch_names)}")
                print(f"  - First 10 channels: {montage.ch_names[:10]}")
                
                # Check for problematic positions
                positions = montage.get_positions()
                ch_pos = positions['ch_pos']
                
                problematic_channels = []
                valid_channels = []
                
                for ch_name, pos in ch_pos.items():
                    if np.isnan(pos).any():
                        problematic_channels.append(f"{ch_name}: NaN")
                    elif np.isinf(pos).any():
                        problematic_channels.append(f"{ch_name}: Inf")
                    elif np.allclose(pos, [0, 0, 0]):
                        problematic_channels.append(f"{ch_name}: Origin")
                    else:
                        valid_channels.append(ch_name)
                
                print(f"  - Valid positions: {len(valid_channels)}")
                print(f"  - Problematic positions: {len(problematic_channels)}")
                
                if problematic_channels:
                    print(f"  ⚠️ Problematic channels: {problematic_channels[:5]}...")
                
                # Test montage application
                try:
                    raw_test = raw.copy()
                    raw_test.set_montage(montage)
                    print(f"  ✓ Montage application successful")
                    
                    # Check final positions
                    final_montage = raw_test.get_montage()
                    if final_montage is not None:
                        final_positions = final_montage.get_positions()['ch_pos']
                        print(f"  ✓ Final positions available: {len(final_positions)}")
                    else:
                        print(f"  ✗ No montage after application")
                        
                except Exception as e:
                    print(f"  ✗ Montage application failed: {e}")
                
            except Exception as e:
                print(f"  ✗ Could not load montage: {e}")
        else:
            print(f"  ✗ File not found")
    
    # 4. Test standard montages
    print("\n4. TESTING STANDARD MONTAGES")
    print("-" * 30)
    
    standard_montages = ['biosemi32', 'biosemi64', 'standard_1020', 'standard_1005']
    
    for montage_name in standard_montages:
        print(f"\nTesting standard montage: {montage_name}")
        try:
            montage = mne.channels.make_standard_montage(montage_name)
            print(f"  ✓ Standard montage '{montage_name}' loaded")
            print(f"  - Channels: {len(montage.ch_names)}")
            
            # Check overlap with our data
            our_channels = [ch for ch in raw.ch_names]
            montage_channels = montage.ch_names
            overlap = set(our_channels) & set(montage_channels)
            
            print(f"  - Overlap with our data: {len(overlap)}/{len(our_channels)}")
            print(f"  - Overlapping channels: {list(overlap)[:10]}...")
            
            if len(overlap) > 20:  # Good coverage
                print(f"  ✓ Good coverage for our dataset")
            else:
                print(f"  ⚠️ Limited coverage for our dataset")
                
        except Exception as e:
            print(f"  ✗ Could not load standard montage '{montage_name}': {e}")
    
    # 5. Analyze channel types and names
    print("\n5. CHANNEL ANALYSIS")
    print("-" * 30)
    
    print("Channel types in data:")
    for i, ch_name in enumerate(raw.ch_names):
        ch_type = mne.io.pick.channel_type(raw.info, i)
        print(f"  {ch_name:12} → {ch_type}")
    
    # Separate by type
    eeg_channels = [ch for i, ch in enumerate(raw.ch_names) 
                   if mne.io.pick.channel_type(raw.info, i) == 'eeg']
    non_eeg_channels = [ch for i, ch in enumerate(raw.ch_names) 
                       if mne.io.pick.channel_type(raw.info, i) != 'eeg']
    
    print(f"\nEEG channels ({len(eeg_channels)}): {eeg_channels}")
    print(f"Non-EEG channels ({len(non_eeg_channels)}): {non_eeg_channels}")
    
    # 6. Simulate bad channel scenarios
    print("\n6. BAD CHANNEL SCENARIOS")
    print("-" * 30)
    
    # Test with different combinations of bad channels
    test_scenarios = [
        ['Fp1'],                    # Single EEG channel
        ['Fp1', 'O1'],             # Multiple EEG channels  
        ['ECG'],                    # Non-EEG channel
        ['Fp1', 'ECG'],            # Mixed EEG + non-EEG
        ['NonExistent'],            # Non-existent channel
        ['Fp1', 'O1', 'FC1', 'Cz'] # Many channels (from actual error)
    ]
    
    # Use the best montage found
    best_montage_file = os.path.join(repo_root, "BC-32_FCz_modified.bvef")
    if os.path.exists(best_montage_file):
        print(f"Using montage: {best_montage_file}")
        montage = mne.channels.read_custom_montage(best_montage_file)
        
        for scenario in test_scenarios:
            print(f"\nTesting bad channels: {scenario}")
            
            raw_test = raw.copy()
            
            # Set channel types (critical step!)
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
                print(f"  ✓ Channel types set: {existing_mapping}")
            
            # Apply montage
            try:
                raw_test.set_montage(montage)
                print(f"  ✓ Montage applied")
            except Exception as e:
                print(f"  ✗ Montage application failed: {e}")
                continue
            
            # Set bad channels
            valid_bads = [ch for ch in scenario if ch in raw_test.ch_names]
            invalid_bads = [ch for ch in scenario if ch not in raw_test.ch_names]
            
            if invalid_bads:
                print(f"  ⚠️ Invalid bad channels (not in data): {invalid_bads}")
            
            if valid_bads:
                raw_test.info['bads'] = valid_bads
                print(f"  → Bad channels set: {valid_bads}")
                
                # Analyze each bad channel
                for bad_ch in valid_bads:
                    ch_idx = raw_test.ch_names.index(bad_ch)
                    ch_type = mne.io.pick.channel_type(raw_test.info, ch_idx)
                    
                    print(f"    {bad_ch}: type={ch_type}", end="")
                    
                    # Check if in montage
                    test_montage = raw_test.get_montage()
                    if test_montage is not None and bad_ch in test_montage.ch_names:
                        pos = test_montage.get_positions()['ch_pos'][bad_ch]
                        if np.isnan(pos).any():
                            print(" → NaN position ✗")
                        elif np.isinf(pos).any():
                            print(" → Inf position ✗")
                        elif np.allclose(pos, [0, 0, 0]):
                            print(" → Origin position ⚠️")
                        else:
                            print(f" → Valid position ✓ {pos}")
                    else:
                        print(" → Not in montage ✗")
                
                # Test interpolation
                try:
                    raw_interp = raw_test.copy().interpolate_bads(reset_bads=True)
                    print(f"  ✓ Interpolation successful")
                except Exception as e:
                    print(f"  ✗ Interpolation failed: {e}")
            else:
                print(f"  → No valid bad channels to test")
    else:
        print(f"⚠️ Best montage file not found: {best_montage_file}")
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    analyze_montage_and_positions() 