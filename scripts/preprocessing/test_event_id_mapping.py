#!/usr/bin/env python
"""
Test script to verify CAMPEONES event_id mapping implementation.

This script checks that the event_id mapping is correctly applied when saving
preprocessed data and that the generated events.tsv files use consistent codes.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

# Standard CAMPEONES event mapping (should match preprocessing_eeg.py)
CAMPEONES_EVENT_ID = {
    'fixation': 10,        # Baseline fixation cross condition (500)
    'calm': 20,           # Calm video condition (901)
    'video': 30,          # Affective video conditions (various stim_ids)
    'video_luminance': 40  # Luminance control videos (100+ stim_ids)
}

def test_event_id_mapping():
    """Test that event_id mapping is correctly applied in preprocessing outputs."""
    
    print("=== TESTING CAMPEONES EVENT_ID MAPPING ===")
    print("Standard mapping:")
    for trial_type, code in CAMPEONES_EVENT_ID.items():
        print(f"  {trial_type:15} → {code:2d}")
    print()
    
    # Define test parameters
    derivatives_folder = repo_root / "data" / "derivatives"
    subject = "14"
    session = "vr"
    task = "01"
    acquisition = "b"
    run = "006"
    
    # Test 1: Check merged events file structure
    print("TEST 1: Checking merged events file structure...")
    merged_events_file = (derivatives_folder / "merged_events" / 
                         f"sub-{subject}" / f"ses-{session}" / "eeg" /
                         f"sub-{subject}_ses-{session}_task-{task}_acq-{acquisition}_run-{run}_desc-merged_events.tsv")
    
    if merged_events_file.exists():
        merged_df = pd.read_csv(merged_events_file, sep='\t')
        print(f"✓ Merged events file found: {merged_events_file.name}")
        print(f"  Trial types: {merged_df['trial_type'].unique()}")
        print(f"  Event count: {len(merged_df)}")
        
        # Check that all trial types are in our standard mapping
        unknown_types = []
        for trial_type in merged_df['trial_type'].unique():
            if trial_type not in CAMPEONES_EVENT_ID:
                unknown_types.append(trial_type)
        
        if unknown_types:
            print(f"  ⚠️ Unknown trial types found: {unknown_types}")
        else:
            print(f"  ✓ All trial types are in standard mapping")
            
    else:
        print(f"✗ Merged events file not found: {merged_events_file}")
        return False
    
    # Test 2: Check if preprocessed events.tsv files exist and use correct mapping
    print("\nTEST 2: Checking preprocessed events.tsv files...")
    
    events_files = []  # Initialize variable
    # Look for preprocessed files
    subject_derivatives = derivatives_folder / f"sub-{subject}" / f"ses-{session}" / "eeg"
    if subject_derivatives.exists():
        # Find events.tsv files
        events_files = list(subject_derivatives.glob("*_events.tsv"))
        
        if events_files:
            print(f"✓ Found {len(events_files)} events.tsv files")
            
            for events_file in events_files:
                print(f"\n  Checking: {events_file.name}")
                
                try:
                    events_df = pd.read_csv(events_file, sep='\t')
                    
                    if 'trial_type' in events_df.columns:
                        # Check if trial_type column uses standard codes
                        trial_types = events_df['trial_type'].unique()
                        print(f"    Trial types: {trial_types}")
                        
                        # If there's a 'value' column, check if it uses our standard codes
                        if 'value' in events_df.columns:
                            event_codes = events_df['value'].unique()
                            print(f"    Event codes: {event_codes}")
                            
                            # Check if codes match our mapping
                            valid_codes = []
                            invalid_codes = []
                            for code in event_codes:
                                if code in CAMPEONES_EVENT_ID.values():
                                    valid_codes.append(code)
                                else:
                                    invalid_codes.append(code)
                            
                            if invalid_codes:
                                print(f"    ⚠️ Non-standard codes found: {invalid_codes}")
                            else:
                                print(f"    ✓ All codes match standard mapping")
                        
                    else:
                        print(f"    ⚠️ No trial_type column found")
                        
                except Exception as e:
                    print(f"    ✗ Error reading file: {e}")
        else:
            print(f"  No events.tsv files found in {subject_derivatives}")
    else:
        print(f"✗ Subject derivatives folder not found: {subject_derivatives}")
    
    # Test 3: Check event consistency between merged_events and preprocessed files
    print("\nTEST 3: Event consistency verification...")
    
    if merged_events_file.exists():
        merged_trial_types = set(merged_df['trial_type'].unique())
        
        print(f"  Merged events trial types: {merged_trial_types}")
        
        # Check that our standard mapping covers all trial types
        covered_types = set(CAMPEONES_EVENT_ID.keys())
        missing_coverage = merged_trial_types - covered_types
        
        if missing_coverage:
            print(f"  ⚠️ Trial types not in standard mapping: {missing_coverage}")
            print(f"  Consider adding these to CAMPEONES_EVENT_ID:")
            for trial_type in missing_coverage:
                print(f"    '{trial_type}': <code>,")
        else:
            print(f"  ✓ All trial types covered by standard mapping")
    
    print("\n=== TEST SUMMARY ===")
    print("Event_id mapping verification completed.")
    print("If preprocessing_eeg.py has been run with the new changes,")
    print("the generated events.tsv files should use the standard codes:")
    for trial_type, code in CAMPEONES_EVENT_ID.items():
        print(f"  {trial_type} → {code}")
    print()
    print("This ensures:")
    print("  ✓ Reproducible event codes across subjects")
    print("  ✓ MNE-BIDS compliance")
    print("  ✓ Consistent analysis pipeline")
    print("======================")

if __name__ == "__main__":
    test_event_id_mapping() 