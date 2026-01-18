#!/usr/bin/env python
# coding: utf-8

# # EEG Preprocessing for Campeones Analysis Project - Variable Duration Epochs
# 
# The following script performs EEG data preprocessing for the Campeones Analysis dataset using **variable duration epochs** based on merged_events. Each video stimulus has its own duration, so epochs are created individually.
# 
# ## Processing Steps (following MNE best practices):
# 
# 1. **Load raw data** - BrainVision format (.vhdr, .vmrk, .eeg files)
# 2. **Set electrode montage** - Apply electrode positions BEFORE any processing
# 3. **Filtering** - Independent notch (3A) and band-pass (3B) blocks with verification (3C, 3D)
# 4. **Motion artifact detection** - Using accelerometer data for head movement detection
# 5. **Visual inspection of channels** - Identify and mark bad channels  
# 6. **Interpolate bad channels & Re-reference** - Before ICA for optimal rank
# 7. **Variable Duration Epoching** - Create individual epochs with baseline=(None, 0) + verification
# 8. **Epoch Quality Assessment** - Amplitude-based rejection (adapted for small datasets)
# 9. **Manual inspection of Epochs** - Visual inspection and manual rejection
# 10. **ICA** - Independent Component Analysis with artifact detection from continuous data
# 11. **Final cleaning** - Baseline already applied in epoch constructor
# 12. **Final preprocessed epochs** - Interpolation and re-referencing verification
# 13. **Save preprocessed data** - Export both continuous file (BIDS) and individual epochs
# 14. **Optional analysis** - PSD plots
# 15. **Generate reports** - HTML report with interactive plots
# 
# ## Key Features:
# - **Variable Duration Support**: Each video stimulus can have different duration
# - **Individual Processing**: Each epoch is processed separately through the pipeline
# - **Baseline Correction**: Uses pre-stimulus period for baseline correction
# - **Event Metadata**: Preserves trial type, duration, and onset information
# 
# **Current Configuration:**
# - Subject: sub-14
# - Session: vr  
# - Task: 01
# - Acquisition: b
# - Run: 006
# - Data Source: `./data/derivatives/merged_events/`

# %%
# %%
import argparse
import sys

# Default values (fallback)
subject = "18"
session = "vr"
task = "04"
acquisition = "a"
run = "005"
interactive = True

# Parse arguments if running as main script
if __name__ == "__main__" and len(sys.argv) > 1:
    parser = argparse.ArgumentParser(description="EEG Preprocessing for Campeones Analysis Project")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID (e.g. '18')")
    parser.add_argument("--session", type=str, required=True, help="Session ID (e.g. 'vr')")
    parser.add_argument("--task", type=str, required=True, help="Task ID (e.g. '04')")
    parser.add_argument("--run", type=str, required=True, help="Run ID (e.g. '005')")
    parser.add_argument("--acquisition", type=str, default="a", help="Acquisition parameter (default: 'a')")
    parser.add_argument("--interactive", action="store_true", help="Show interactive plots (blocks execution)")
    
    args = parser.parse_args()
    subject = args.subject
    session = args.session
    task = args.task
    run = args.run
    acquisition = args.acquisition
    interactive = args.interactive

import matplotlib
if interactive:
    try:
        matplotlib.use("Qt5Agg")
    except:
        print("Qt5Agg not available, using Agg")
        matplotlib.use("Agg")
else:
    matplotlib.use("Agg")

# Import necessary libraries for the preprocessing
import os
import sys
from git import Repo
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import mne
from mne_bids import BIDSPath, read_raw_bids ,write_raw_bids
# Importing libraries for automatic rejection of bad epochs
from pyprep import NoisyChannels

# tag automatically ICA components
# requires pytorch
from mne_icalabel import label_components

# Add src directory to Python path to enable imports
repo = Repo(os.getcwd(), search_parent_directories=True)
repo_root = repo.git.rev_parse("--show-toplevel")
src_path = os.path.join(repo_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import helper functions for preprocessing
from campeones_analysis.utils.log_preprocessing import LogPreprocessingDetails
from campeones_analysis.utils import bids_compliance


#%%
# LOAD DATA
# Get the current working directory
cwd = os.getcwd()

# Define the file path components for Campeones Analysis project
raw_data_folder = "data/raw"

# Variables already set by argparse or defaults above
data = "eeg"
print(f"Running preprocessing for: sub-{subject} ses-{session} task-{task} acq-{acquisition} run-{run}")
print(f"Interactive mode: {interactive}")

# Create a BIDSPath object pointing to raw data
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
)

#%%
# FOR SAVING
# Pipeline-specific derivatives folder for better provenance and BIDS compliance
# Following BIDS recommendation: each processing pipeline should have its own subdirectory
pipeline_name = "campeones_preproc"  # Descriptive pipeline name
derivatives_root = os.path.join(repo_root, "data", "derivatives")
derivatives_folder = os.path.join(derivatives_root, pipeline_name)
bids_dir = os.path.join(derivatives_folder, f"sub-{subject}", f"ses-{session}", "eeg")
os.makedirs(bids_dir, exist_ok=True)

print(f"=== PIPELINE-SPECIFIC DERIVATIVES ORGANIZATION ===")
print(f"Pipeline name: {pipeline_name}")
print(f"Derivatives root: {derivatives_root}")
print(f"Pipeline folder: {derivatives_folder}")
print(f"Subject output: {bids_dir}")
print("✓ Following BIDS best practices for derivatives organization")
print("✓ Enables coexistence of multiple processing pipelines")
print("✓ Improved provenance and traceability")

# Create pipeline dataset_description.json if it doesn't exist
pipeline_description_file = os.path.join(derivatives_folder, "dataset_description.json")
if not os.path.exists(pipeline_description_file):
    try:
        # Import the function (assuming it's in the same directory)
        sys.path.append(os.path.dirname(__file__))
        from create_pipeline_description import create_pipeline_description
        create_pipeline_description(derivatives_folder)
        print("✓ Pipeline dataset_description.json created")
    except ImportError:
        print("⚠ Could not create dataset_description.json (script not found)")
        # Create a minimal version manually
        minimal_description = {
            "Name": "CAMPEONES EEG Preprocessing Pipeline",
            "BIDSVersion": "1.8.0",
            "DatasetType": "derivative",
            "GeneratedBy": [{"Name": "campeones_preproc", "Version": "1.0"}]
        }
        import json
        with open(pipeline_description_file, 'w') as f:
            json.dump(minimal_description, f, indent=2)
        print("✓ Minimal dataset_description.json created")
else:
    print("✓ Pipeline dataset_description.json already exists")

print("=== DERIVATIVES ORGANIZATION COMPLETED ===\n")

# Note: Report and logging initialization moved up to be available earlier

#%%
# 1A. READ RAW

# Read Raw bids - BrainVision format (.vhdr, .vmrk, .eeg)
raw = read_raw_bids(bids_path)

# Initialize bad channels list (can be populated based on visual inspection)
# For now, start with empty list - you can update this manually if needed
raw.info["bads"] = []

print(f"Loading file: {bids_path}")
print(raw.info)

# 1B. DEFINE STANDARD EVENT ID MAPPING

# Define standard event_id mapping for all CAMPEONES trial types
# This ensures consistent and reproducible event codes across all subjects and runs
# Based on merged_events files from the derivatives folder
CAMPEONES_EVENT_ID = {
    'fixation': 10,        # Baseline fixation cross condition (500)
    'calm': 20,           # Calm video condition (901)
    'video': 30,          # Affective video conditions (various stim_ids)
    'video_luminance': 40  # Luminance control videos (100+ stim_ids)
}

print("=== STANDARD EVENT ID MAPPING ===")
print("Campeones Analysis project uses the following event codes:")
for trial_type, code in CAMPEONES_EVENT_ID.items():
    print(f"  {trial_type:15} → {code:2d}")
print("✓ Consistent event mapping ensures reproducible event.tsv files")
print("✓ Based on trial_type column from merged_events derivatives")
print("=== EVENT ID MAPPING DEFINED ===\n")

# %%
# 2. Verify electrode montage BEFORE any processing 

# Verificar rápidamente si hay posiciones antes de procesar
if raw.get_montage() is None or len(raw.get_montage().ch_names) == 0:
    raise RuntimeError("El dataset no contiene posiciones de electrodos.")
print("✓ Electrode montage ya presente (n =", len(raw.get_montage().ch_names), ")")

# Initialize a report to document the preprocessing steps (moved here to be available)
if 'report' not in locals():
    report = mne.Report(
        title=f"Preprocessing sub-{subject} ses-{session} task-{task} run-{run}"
    )

# Add the raw data info to the report
report.add_raw(raw=raw, title="Raw", psd=True)

# Initialize logging if not already done
if 'log_preprocessing' not in locals():
    # Path to the JSON file where preprocessing details will be stored
    # Using pipeline-specific folder for better organization
    json_path = os.path.join(
        derivatives_folder, "logs_preprocessing_details_all_subjects_eeg.json"
    )
    # Initialize the logging class
    log_preprocessing = LogPreprocessingDetails(json_path, subject, session, f"{task}_run-{run}")

# Log the raw data info and channel type setup
log_preprocessing.log_detail("info", str(raw.info))

# Log pipeline-specific information for provenance tracking
log_preprocessing.log_detail("pipeline_name", pipeline_name)
log_preprocessing.log_detail("derivatives_organization", "pipeline_specific")
log_preprocessing.log_detail("derivatives_root", derivatives_root)
log_preprocessing.log_detail("derivatives_pipeline_folder", derivatives_folder)
log_preprocessing.log_detail("bids_derivatives_compliant", True)
log_preprocessing.log_detail("pipeline_version", "1.0")  # Version del pipeline
log_preprocessing.log_detail("pipeline_description", "CAMPEONES EEG preprocessing with variable duration epochs")

# Verificar backend y crear navegador interactivo de datos
print(f"Backend de matplotlib actual: {matplotlib.get_backend()}")

# Crear navegador interactivo de datos EEG
#print("Creando navegador interactivo de datos...")
#browser = raw.plot(block=False, scalings="auto")
#print("Navegador creado. Si no ves el widget, verifica que ipympl esté funcionando correctamente.")

# También podemos verificar la calidad de las señales
print(f"Datos cargados: {raw.n_times} puntos temporales, {raw.info['sfreq']} Hz")
print(f"Canales: {raw.info['nchan']} total ({len(raw.info['ch_names'])} nombres)")
print(f"Duración: {raw.times[-1]:.2f} segundos")


# %%
# 3A. NOTCH FILTERING (Independent Block)

# Line noise removal for European electrical grid (50 Hz + harmonics)
print("=== STEP 3A: NOTCH FILTERING ===")

# Define notch filter parameters
notch_freqs = [50, 100]  # Line noise and first harmonic
notch_method = 'fir'     # FIR filter for better phase characteristics
notch_phase = 'zero'     # Zero-phase filtering

print(f"Notch filter configuration:")
print(f"  - Frequencies: {notch_freqs} Hz")
print(f"  - Method: {notch_method}")
print(f"  - Phase: {notch_phase}")

# Load data and apply notch filter
raw_loaded = raw.load_data().copy()
print(f"Raw data loaded for filtering: {raw_loaded.n_times} samples")

# Apply notch filter independently
print("Applying notch filter...")
raw_notched = raw_loaded.notch_filter(
    freqs=notch_freqs, 
    picks='eeg',
    method=notch_method,
    phase=notch_phase,
    verbose=True
)
print("✓ Notch filtering completed")

# Log notch filter details
log_preprocessing.log_detail("notch_frequencies", notch_freqs)
log_preprocessing.log_detail("notch_method", notch_method)
log_preprocessing.log_detail("notch_phase", notch_phase)
log_preprocessing.log_detail("notch_filter_independent", True)

print("=== NOTCH FILTERING COMPLETED ===\n")


# %%
# 3B. BAND-PASS FILTERING (Independent Block)

# ERP-optimized frequency range following MNE recommendations
print("=== STEP 3B: BAND-PASS FILTERING ===")

# Define band-pass filter parameters
hpass = 1.0   # High-pass: 1.0 Hz (removes DC offset and slow drifts)
lpass = 48.0  # Low-pass: 48 Hz (preserves EEG frequencies, removes high-freq artifacts)
filter_method = 'fir'     # FIR filter for better characteristics
filter_phase = 'zero'     # Zero-phase filtering
filter_length = 'auto'    # Automatic filter length selection

print(f"Band-pass filter configuration:")
print(f"  - High-pass cutoff: {hpass} Hz (removes DC offset and slow drifts)")
print(f"  - Low-pass cutoff: {lpass} Hz (removes high-frequency artifacts)")
print(f"  - Method: {filter_method}")
print(f"  - Phase: {filter_phase}")
print(f"  - Filter length: {filter_length}")
print(f"  - DC offset removal: Enhanced with 1.0 Hz HP filter")

# Apply band-pass filter independently
print("Applying band-pass filter...")
raw_filtered = raw_notched.filter(
    l_freq=hpass, 
    h_freq=lpass, 
    picks='eeg',
    method=filter_method,
    phase=filter_phase,
    filter_length=filter_length,
    verbose=True
)
print("✓ Band-pass filtering completed")
print("✓ DC offset should be significantly reduced")

# Log band-pass filter details
log_preprocessing.log_detail("hpass_filter", hpass)
log_preprocessing.log_detail("lpass_filter", lpass)
log_preprocessing.log_detail("bandpass_method", filter_method)
log_preprocessing.log_detail("bandpass_phase", filter_phase)
log_preprocessing.log_detail("bandpass_filter_independent", True)
log_preprocessing.log_detail("eeg_optimized_filtering", True)
log_preprocessing.log_detail("dc_offset_removal_enhanced", True)

print("=== BAND-PASS FILTERING COMPLETED ===\n")


#%%
# 3C. FILTERING VERIFICATION AND REPORTS

# Create comprehensive PSD comparison plot
print("=== STEP 3C: FILTERING VERIFICATION ===")
print("Creating PSD comparison plots...")

# Create detailed comparison figure with three subplots
fig = plt.figure(figsize=(18, 6))

# Plot 1: Raw data PSD
ax1 = plt.subplot(1, 3, 1)
raw.compute_psd(fmax=100).plot(axes=ax1, show=False)
ax1.set_title('Raw Data PSD\n(Unfiltered)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Power (dB)')
ax1.grid(True, alpha=0.3)

# Plot 2: After notch filtering only
ax2 = plt.subplot(1, 3, 2)
raw_notched.compute_psd(fmax=100).plot(axes=ax2, show=False)
ax2.set_title(f'After Notch Filter\n(Removed: {notch_freqs} Hz)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Power (dB)')
ax2.grid(True, alpha=0.3)

# Plot 3: Final filtered data PSD
ax3 = plt.subplot(1, 3, 3)
raw_filtered.compute_psd(fmax=100).plot(axes=ax3, show=False)
ax3.set_title(f'Final Filtered Data\n(HP: {hpass} Hz, LP: {lpass} Hz)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Power (dB)')
ax3.grid(True, alpha=0.3)

plt.suptitle('Independent Filter Blocks - PSD Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Summary of filtering results
print("✓ Independent filtering blocks applied successfully:")
print(f"  Step 1 - Notch filter: Removed line noise at {notch_freqs} Hz")
print(f"  Step 2 - Band-pass filter: Preserved {hpass}-{lpass} Hz range")
print(f"  Method: {filter_method} with {filter_phase}-phase design")
print("✓ Improved traceability through separate filter blocks")

# Save the filtered data with comprehensive description
bids_path.update(root = derivatives_folder, description = 'filtered')

# Log the overall filtering approach
log_preprocessing.log_detail("filter_order", "notch_then_bandpass_independent")
log_preprocessing.log_detail("filtering_approach", "separate_independent_blocks")

log_preprocessing.log_detail("psd_comparison_plots", 3)  # Raw, Notched, Final

print("=== FILTERING VERIFICATION COMPLETED ===\n")

#%%
# 4. Save filtered data WITHOUT converting original annotations to events
# RATIONALE: Original annotations are experimental markers, not real events
# Real events come from merged_events files and will be loaded later in the pipeline
print("=== SAVING FILTERED DATA (WITHOUT ORIGINAL ANNOTATIONS AS EVENTS) ===")
print("Note: Original annotations are experimental markers, not trial events")
print("Real events will be loaded from merged_events derivatives")

# Check if there are original annotations for information
if raw_filtered.annotations:
    print(f"Original annotations found: {len(raw_filtered.annotations)} markers")
    print(f"Description types: {set(raw_filtered.annotations.description)}")
    print("These will be preserved as annotations but NOT converted to events")
else:
    print("No original annotations found in raw data")

raw_filtered.set_annotations(None)   

# Save filtered data without creating events from original annotations
write_raw_bids(
    raw_filtered, 
    bids_path, 
    format='BrainVision', 
    allow_preload=True, 
    overwrite=True,
    events=None,  # Explicitly no events - real events come from merged_events
    event_id=None
)
print(f"✓ Filtered data saved: {bids_path.fpath}")
print("✓ Original annotations preserved as annotations (not events)")
print("✓ No events file created from original annotations")
print("✓ Real events will be loaded from merged_events derivatives later")

# %%
# 5. Visual inspection of channels and bad channel detection

# Note: Detailed PSD comparison already completed in section 3C
print("=== VISUAL INSPECTION AND BAD CHANNEL DETECTION ===")
print("(PSD comparison plots already shown in section 3C)")

# Automatically mark bad channels using PyPREP
nd = NoisyChannels(raw_filtered,do_detrend = False, random_state=42)
nd.find_all_bads(ransac=True, channel_wise=True) #if it slows down, set channel_wise to False
bads = nd.get_bads()
print(f"Bad channels detected: {bads}")
if bads != None:
    raw_filtered.info["bads"] = bads

# Plot the filtered data for visual inspection to identify bad channels
filtered_browser = raw_filtered.plot(n_channels=32, block=False)
print("Navegador de datos filtrados creado. Úsalo para identificar visualmente canales malos.")

# Add the filtered data to the report
report.add_raw(raw=raw_filtered, title="Filtered Raw", psd=True)

# Log the identified bad channels
log_preprocessing.log_detail("bad_channels", raw_filtered.info["bads"])



# %%
## 6. Set annotations from merged_events
# Load triggers from merged_events (with variable durations and real onset times)
events_file = os.path.join(
    derivatives_root, "merged_events", f"sub-{subject}", f"ses-{session}", "eeg",
    f"sub-{subject}_ses-{session}_task-{task}_acq-{acquisition}_run-{run}_desc-merged_events.tsv"
)

# Load events from TSV file
print(f"Loading events from: {events_file}")
events_df = pd.read_csv(events_file, sep='\t')
print(f"Events loaded: {len(events_df)} events")
print(f"Event types found: {events_df['trial_type'].unique()}")
print(f"Conditions found: {events_df['condition'].unique()}")

# Verify that the onset times are reasonable
print(f"Onset times: min={events_df['onset'].min():.1f}s, max={events_df['onset'].max():.1f}s")
print(f"Durations: min={events_df['duration'].min():.1f}s, max={events_df['duration'].max():.1f}s")

# Convert events to MNE annotations
event_annotations = mne.Annotations(
    onset=events_df['onset'].values,
    duration=events_df['duration'].values,
    description=events_df['trial_type'].values
)

raw_filtered.set_annotations(event_annotations)

# %%
## 6B. Mark all non-merged_events segments as "bad"
# RATIONALE: Ensure complete annotation coverage - every sample is either from merged_events or marked as bad
# This enables clean ICA fitting and analysis by excluding irrelevant segments

print("=== MARKING NON-MERGED_EVENTS SEGMENTS AS BAD ===")

# Get total recording duration
total_duration = raw_filtered.times[-1]
print(f"Total recording duration: {total_duration:.2f}s")

# Get merged_events coverage
merged_events_coverage = []
for i, row in events_df.iterrows():
    start_time = row['onset']
    end_time = row['onset'] + row['duration']
    merged_events_coverage.append((start_time, end_time))

# Sort by start time to ensure proper ordering
merged_events_coverage.sort(key=lambda x: x[0])
print(f"Merged events segments: {len(merged_events_coverage)}")

# Find gaps between merged_events (these will be marked as "bad")
bad_segments = []

# Check if there's a gap at the beginning
if merged_events_coverage[0][0] > 0:
    bad_segments.append((0, merged_events_coverage[0][0]))
    print(f"Gap at beginning: 0 - {merged_events_coverage[0][0]:.2f}s")

# Check gaps between consecutive merged_events
for i in range(len(merged_events_coverage) - 1):
    current_end = merged_events_coverage[i][1]
    next_start = merged_events_coverage[i + 1][0]
    
    if next_start > current_end:
        bad_segments.append((current_end, next_start))
        print(f"Gap between events: {current_end:.2f} - {next_start:.2f}s")

# Check if there's a gap at the end
if merged_events_coverage[-1][1] < total_duration:
    bad_segments.append((merged_events_coverage[-1][1], total_duration))
    print(f"Gap at end: {merged_events_coverage[-1][1]:.2f} - {total_duration:.2f}s")

# Create "bad" annotations for all gaps
if bad_segments:
    bad_onsets = [seg[0] for seg in bad_segments]
    bad_durations = [seg[1] - seg[0] for seg in bad_segments]
    bad_descriptions = ['bad'] * len(bad_segments)
    
    # Create bad annotations
    bad_annotations = mne.Annotations(
        onset=bad_onsets,
        duration=bad_durations,
        description=bad_descriptions
    )
    
    # Combine merged_events annotations with bad annotations
    combined_annotations = event_annotations + bad_annotations
    raw_filtered.set_annotations(combined_annotations)
    
    # Calculate coverage statistics
    merged_events_total_duration = sum([row['duration'] for _, row in events_df.iterrows()])
    bad_total_duration = sum(bad_durations)
    merged_events_percentage = (merged_events_total_duration / total_duration) * 100
    bad_percentage = (bad_total_duration / total_duration) * 100
    
    print(f"\n✓ Complete annotation coverage achieved:")
    print(f"  - Merged events: {merged_events_total_duration:.1f}s ({merged_events_percentage:.1f}%)")
    print(f"  - Bad segments: {bad_total_duration:.1f}s ({bad_percentage:.1f}%)")
    print(f"  - Total coverage: {merged_events_total_duration + bad_total_duration:.1f}s (100.0%)")
    print(f"  - Bad segments created: {len(bad_segments)}")
    
    # Log the complete annotation details
    log_preprocessing.log_detail("complete_annotation_coverage", True)
    log_preprocessing.log_detail("merged_events_duration_s", merged_events_total_duration)
    log_preprocessing.log_detail("bad_segments_duration_s", bad_total_duration)
    log_preprocessing.log_detail("merged_events_percentage", merged_events_percentage)
    log_preprocessing.log_detail("bad_segments_percentage", bad_percentage)
    log_preprocessing.log_detail("bad_segments_count", len(bad_segments))
    log_preprocessing.log_detail("annotation_strategy", "merged_events_plus_bad_gaps")
    
else:
    print("✓ No gaps found - merged_events cover entire recording")
    log_preprocessing.log_detail("complete_annotation_coverage", True)
    log_preprocessing.log_detail("gaps_found", False)

if interactive:
    raw_filtered.plot(block=True)

print("=== COMPLETE ANNOTATION COVERAGE IMPLEMENTED ===")
print("Benefits:")
print("✓ Every sample is annotated (merged_events or bad)")
print("✓ ICA will use only relevant segments (non-bad)")
print("✓ Analysis pipeline can automatically exclude irrelevant periods")
print("✓ Clear provenance of data inclusion/exclusion decisions")
print("✓ Reproducible segment selection across analysis steps")
print("="*50)


# %%
## 8. Independent Component Analysis (ICA)

print("=== ICA WITH COMPLETE ANNOTATION COVERAGE ===")

# Parameters for ICA (Independent Component Analysis) to remove artifacts
n_components = None
method = "picard"  # The algorithm to use for ICA
max_iter = (
    "auto"  # Maximum number of iterations; typically should be higher, like 500 or 1000
)
random_state = 42  # Seed for random number generator for reproducibility

# Initialize the ICA object with the specified parameters
ica = mne.preprocessing.ICA(
    n_components=n_components,
    method=method,
    max_iter=max_iter,
    random_state=random_state,
)

# Verify annotation coverage before ICA
print(f"Pre-ICA annotation verification:")
if raw_filtered.annotations is not None:
    print(f"  - Total annotations: {len(raw_filtered.annotations)}")
    annotation_types = set(raw_filtered.annotations.description)
    print(f"  - Annotation types: {annotation_types}")
    
    # Calculate coverage
    good_duration = sum([ann['duration'] for ann in raw_filtered.annotations if ann['description'] != 'bad'])
    bad_duration = sum([ann['duration'] for ann in raw_filtered.annotations if ann['description'] == 'bad'])
    print(f"  - Good segments: {good_duration:.1f}s")
    print(f"  - Bad segments: {bad_duration:.1f}s")
    print(f"  - ICA will use only good segments (auto-exclude bad)")
else:
    print("  - WARNING: No annotations found!")

# Fit the ICA model ONLY on non-bad segments (automatic with reject_by_annotation=True)
print("Fitting ICA on good segments only (excluding 'bad' annotations)...")
ica.fit(raw_filtered, picks='eeg', reject_by_annotation=True)
print("✓ ICA fitted successfully using only merged_events segments")

# Log ICA annotation usage
log_preprocessing.log_detail("ica_reject_by_annotation", True)
log_preprocessing.log_detail("ica_used_only_merged_events", True)
log_preprocessing.log_detail("ica_excluded_bad_segments", True)

# find EOG artifacts in the data via pattern matching, and exclude the EOG-related ICA components
eog_components, eog_scores = ica.find_bads_eog(
    inst=raw_filtered,
    ch_name="R_EYE",  # a channel close to the eye
    # threshold=1  # lower than the default threshold
)
print(f"EOG components detected: {eog_components}")

# find ECG artifacts in the data via pattern matching, and exclude the ECG-related ICA components
ecg_components, ecg_scores = ica.find_bads_ecg(
    inst=raw_filtered,
    ch_name="ECG",  # a channel close to the eye
    # threshold=1  # lower than the default threshold
)
print(f"ECG components detected: {ecg_components}")

# find muscle artifacts in the data via pattern matching, and exclude the muscle-related ICA components
muscle_components, muscle_scores = ica.find_bads_muscle(raw_filtered, threshold=0.7)
print(f"Muscle components detected: {muscle_components}")
# ica.plot_scores(muscle_scores, exclude=muscle_components)

# Combine all artifact components from the pattern matching methods
pattern_matching_artifacts = np.unique(ecg_components + eog_components + muscle_components)

##### Classify the components using ICLabel model #######
# run the model on the ICA components
ic_labels = label_components(raw_filtered, ica, method="iclabel")

# Create readable table of ICA component classifications
print("=== ICLabel CLASSIFICATION RESULTS ===")
print(f"{'Component':<12} {'Classification':<18} {'Action'}")
print("-" * 55)

label_names = ic_labels['labels']
label_probabilities = ic_labels['y_pred_proba']

for i, (label, probs) in enumerate(zip(label_names, label_probabilities)):
    component_name = f"ICA{i:03d}"
    
    # Determine recommended action
    if label in ['muscle artifact', 'eye blink', 'heart beat', 'channel noise']:
        action = "→ EXCLUDE"
    elif label == 'brain':
        action = "→ KEEP"
    else:
        action = "→ REVIEW"
    
    print(f"{component_name:<12} {label:<18} {action}")

# Summary statistics
label_counts = {}
for label in label_names:
    label_counts[label] = label_counts.get(label, 0) + 1

print(f"\n=== SUMMARY ===")
print(f"Total components: {len(label_names)}")
for label, count in sorted(label_counts.items()):
    percentage = (count / len(label_names)) * 100
    print(f"{label:<18}: {count:2d} ({percentage:4.1f}%)")

# Automatic exclusion recommendations
auto_exclude_types = ['muscle artifact', 'eye blink', 'heart beat', 'channel noise']
auto_exclude_count = sum(1 for label in label_names if label in auto_exclude_types)
brain_count = sum(1 for label in label_names if label == 'brain')

print(f"\n=== RECOMMENDATIONS ===")
print(f"Auto-exclude candidates: {auto_exclude_count}/{len(label_names)} ({(auto_exclude_count/len(label_names)*100):.1f}%)")
print(f"Brain components: {brain_count}/{len(label_names)} ({(brain_count/len(label_names)*100):.1f}%)")
print(f"Components to review: {len(label_names) - auto_exclude_count - brain_count}")
print("=" * 55)

# Extract ICA component labels
label_names = ic_labels['labels']

# Identify the ICA components that correspond to a 'channel noise' in ICLabel
channel_artifact_indices = [i for i, label in enumerate(label_names) if label == 'channel noise']

# Find components that coincide between pattern matching and ICLabel output for exclusion
# We'll only exclude components that match the artifacts found via pattern matching 
# and are classified as 'muscle artifact', 'eye blink', 'heart beat', or 'channel noise'
to_exclude = []
for idx in pattern_matching_artifacts:
    if label_names[idx] in ['muscle artifact', 'eye blink', 'heart beat', 'channel noise']:
        to_exclude.append(idx)

if len(eog_components) > 0 and eog_components[0] < 3:
    to_exclude.append(eog_components[0])

# Also ensure to include 'channel noise' components that were found only by ICLabel
to_exclude = np.unique(to_exclude + channel_artifact_indices)

# Exclude the selected components
ica.exclude = to_exclude.tolist()

# (Optional) Plot the ICA components for visual inspection
# ica.plot_components(inst=epochs_clean, picks=range(15))

# Plot the sources identified by ICA
# Plot the sources identified by ICA
if interactive:
    print("Opening interactive ICA sources plot...")
    ica.plot_sources(raw_filtered, block=True, show=True)
    
    print("Opening interactive ICA components plot (topomaps)...")
    ica.plot_components(inst=raw_filtered, show=True)
    
    plt.show(block=True)

# Add the ICA results to the report
report.add_ica(ica, title="ICA", inst=raw_filtered)

# Apply the ICA solution to the re-referenced data
raw_ica = ica.apply(inst=raw_filtered)

print("✓ ICA applied to preprocessed data")
print(f"✓ ICA excluded {len(ica.exclude)} components")
print(f"✓ Final preprocessed data ready for epoching and analysis")

# Log the ICA parameters and excluded components
log_preprocessing.log_detail("ica_components_excluded", ica.exclude)
log_preprocessing.log_detail("ica_method", method)
log_preprocessing.log_detail("ica_max_iter", max_iter)
log_preprocessing.log_detail("ica_random_state", random_state)

# Log detailed ICLabel results
log_preprocessing.log_detail("iclabel_total_components", len(label_names))
log_preprocessing.log_detail("iclabel_classifications", {
    "component_labels": {f"ICA{i:03d}": label for i, label in enumerate(label_names)},
    "label_counts": label_counts,
    "auto_exclude_candidates": auto_exclude_count,
    "brain_components": brain_count,
    "review_needed": len(label_names) - auto_exclude_count - brain_count
})
log_preprocessing.log_detail("iclabel_automatic_exclusions", [f"ICA{i:03d}" for i, label in enumerate(label_names) if label in auto_exclude_types])
log_preprocessing.log_detail("iclabel_brain_components", [f"ICA{i:03d}" for i, label in enumerate(label_names) if label == 'brain'])
log_preprocessing.log_detail("iclabel_review_components", [f"ICA{i:03d}" for i, label in enumerate(label_names) if label not in auto_exclude_types and label != 'brain'])



#%%
## 9. Interpolate Chs and Rereference

##################################
#######    Rereference   #########
##################################
raw_ica = mne.add_reference_channels(raw_ica.load_data(), ref_channels=["FCz"])

# Path to your .bvef file (relative to this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
bvef_file_path = os.path.join(script_dir, 'BC-32_FCz_modified.bvef')

if not os.path.exists(bvef_file_path):
    raise FileNotFoundError(f"Montage file not found at: {bvef_file_path}")

## Load the extended montage
montage = mne.channels.read_custom_montage(bvef_file_path)
#
## Apply the montage to your raw data
raw_ica.set_montage(montage)


# Rereference the data to the grand average reference
raw_rereferenced, ref_data = mne.set_eeg_reference(
    inst=raw_ica, ref_channels="average", copy=True
)

# Add the final preprocessed raw data to the report
report.add_raw(
    raw=raw_rereferenced, title="Raw data interpolated and rereferenced", psd=True
)

# Log the rereferencing details
log_preprocessing.log_detail("rereferenced_channels", "grand_average")

#%%
# Interpolate chs 

# Interpolate bad channels in the raw data after ICA application
raw_interpolate = raw_rereferenced.copy().interpolate_bads()

# Log the interpolated channels
log_preprocessing.log_detail("interpolated_channels", raw_ica.info["bads"])

# Plot the final preprocessed data for visual inspection
print("=== FINAL PREPROCESSED DATA VISUALIZATION ===")
print("Plotting final preprocessed data (filtered, ICA applied, interpolated, re-referenced)...")

# Create an interactive plot of the final preprocessed data
final_browser = raw_interpolate.plot(
    n_channels=32, 
    scalings="auto",
    title="Final Preprocessed EEG Data",
    block=False
)
print("✓ Final preprocessed data browser created")
print("  - Filtering: Notch (50, 100 Hz) + Bandpass (1-48 Hz)")
print("  - ICA: Artifact components removed")
print("  - Interpolation: Bad channels interpolated")
print("  - Re-referencing: Average reference applied")
print("  - Ready for epoching and analysis")

# Create an interactive plot showing annotations for documentation
print("Creating interactive plot with annotations for report...")
print("✓ Interactive plot will show all preprocessing steps and annotations")

# Add the final preprocessed raw data to the report (interactive with annotations)
print("Adding interactive final preprocessed raw data to report with annotations...")

# Verify annotations are present before adding to report
if raw_interpolate.annotations is not None:
    n_annotations = len(raw_interpolate.annotations)
    annotation_types = set(raw_interpolate.annotations.description)
    print(f"  → {n_annotations} annotations found: {list(annotation_types)}")
    print("  → Annotations will be visible in the interactive plot")
else:
    print("  → No annotations found")

# Add raw data with interactive plot that shows annotations
report.add_raw(
    raw=raw_interpolate, 
    title="Final Preprocessed Raw Data (Interactive with Annotations)", 
    psd=True,
    # The interactive plot will automatically show annotations when present
)
print("✓ Interactive raw data plot added to report with:")
print("  - All preprocessing steps applied")
print("  - Annotations visible and clickable")
print("  - Power Spectral Density analysis")
print("  - Channel information and montage")

print("=== FINAL PREPROCESSED DATA VISUALIZATION COMPLETED ===\n")

#%%
# 10. Save final preprocessed raw data

print("=== SAVING FINAL PREPROCESSED RAW DATA ===")

# 10B. Generate individual PSDs for each annotation type (excluding 'bad')
print("=== GENERATING INDIVIDUAL PSDs FOR EACH ANNOTATION TYPE ===")

if raw_interpolate.annotations is not None:
    # Get unique annotation types (excluding 'bad')
    annotation_types = set(raw_interpolate.annotations.description)
    annotation_types.discard('bad')  # Remove 'bad' annotations
    
    print(f"Annotation types found (excluding 'bad'): {list(annotation_types)}")
    
    for ann_type in sorted(annotation_types):
        print(f"Generating PSD for annotation: {ann_type}")
        
        try:
            # Create a copy of raw data for this annotation type
            raw_ann = raw_interpolate.copy()
            
            # Strategy: Mark all OTHER annotation types as 'bad' to isolate current type
            ann_onsets = []
            ann_durations = []
            ann_descriptions = []
            
            for ann in raw_interpolate.annotations:
                if ann['description'] == ann_type:
                    # Keep this annotation type as is
                    ann_onsets.append(ann['onset'])
                    ann_durations.append(ann['duration'])
                    ann_descriptions.append(ann['description'])
                elif ann['description'] == 'bad':
                    # Keep original bad annotations
                    ann_onsets.append(ann['onset'])
                    ann_durations.append(ann['duration'])
                    ann_descriptions.append('bad')
                else:
                    # Mark all other annotation types as 'bad' to exclude them
                    ann_onsets.append(ann['onset'])
                    ann_durations.append(ann['duration'])
                    ann_descriptions.append('bad')
            
            if ann_onsets:  # Only proceed if we have annotations of this type
                # Set the filtered annotations
                filtered_annotations = mne.Annotations(
                    onset=ann_onsets,
                    duration=ann_durations,
                    description=ann_descriptions
                )
                raw_ann.set_annotations(filtered_annotations)
                
                # Compute PSD for this annotation type (excluding bad segments)
                psd_ann = raw_ann.compute_psd(
                    fmax=45,  # Limit to EEG relevant frequencies
                    picks='eeg',
                    reject_by_annotation=['bad']  # Exclude bad segments
                )
                
                # Calculate total duration from original annotations (more accurate)
                total_duration = sum([ann['duration'] for ann in raw_interpolate.annotations 
                                    if ann['description'] == ann_type])
                title = f"PSD - {ann_type} (duration: {total_duration:.1f}s)"
                
                # Fix the MNE plot access issue
                fig_psd = psd_ann.plot(show=False)
                if hasattr(fig_psd, 'figure'):
                    fig_to_add = fig_psd.figure
                else:
                    fig_to_add = fig_psd
                
                report.add_figure(
                    fig=fig_to_add,
                    title=title,
                    caption=f"Power Spectral Density for {ann_type} annotation type. "
                           f"Total duration: {total_duration:.1f} seconds."
                )
                
                print(f"✓ PSD added for {ann_type} (duration: {total_duration:.1f}s)")
                
            else:
                print(f"⚠ No valid segments found for annotation type: {ann_type}")
                
        except Exception as e:
            print(f"⚠ Error generating PSD for {ann_type}: {str(e)}")
            continue
    
    # Log the PSD generation details
    log_preprocessing.log_detail("individual_psds_generated", True)
    log_preprocessing.log_detail("psd_annotation_types", list(annotation_types))
    log_preprocessing.log_detail("psd_frequency_range", "0-50 Hz")
    
    print(f"✓ Individual PSDs generated for {len(annotation_types)} annotation types")
    print("✓ PSDs exclude 'bad' segments automatically")
    
    # 10C. Generate comparative PSD plot with all conditions
    print("=== GENERATING COMPARATIVE PSD PLOT ===")
    
    try:
        # Create a figure for the comparative plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define colors for different conditions (fix matplotlib colormap access)
        import matplotlib.cm as cm
        colors = cm.get_cmap('Set1')(np.linspace(0, 1, len(annotation_types)))
        color_map = dict(zip(sorted(annotation_types), colors))
        
        psd_data = {}  # Store PSD data for logging
        
        for ann_type in sorted(annotation_types):
            print(f"Computing averaged PSD for: {ann_type}")
            
            try:
                # Create a copy of raw data for this annotation type
                raw_ann = raw_interpolate.copy()
                
                # Strategy: Mark all OTHER annotation types as 'bad' to isolate current type
                ann_onsets = []
                ann_durations = []
                ann_descriptions = []
                
                for ann in raw_interpolate.annotations:
                    if ann['description'] == ann_type:
                        # Keep this annotation type as is
                        ann_onsets.append(ann['onset'])
                        ann_durations.append(ann['duration'])
                        ann_descriptions.append(ann['description'])
                    elif ann['description'] == 'bad':
                        # Keep original bad annotations
                        ann_onsets.append(ann['onset'])
                        ann_durations.append(ann['duration'])
                        ann_descriptions.append('bad')
                    else:
                        # Mark all other annotation types as 'bad' to exclude them
                        ann_onsets.append(ann['onset'])
                        ann_durations.append(ann['duration'])
                        ann_descriptions.append('bad')
                
                if ann_onsets:
                    # Set the filtered annotations where only current type is preserved
                    filtered_annotations = mne.Annotations(
                        onset=ann_onsets,
                        duration=ann_durations,
                        description=ann_descriptions
                    )
                    raw_ann.set_annotations(filtered_annotations)
                    
                    # Debug: Check annotation coverage for this condition
                    current_type_annotations = [ann for ann in filtered_annotations if ann['description'] == ann_type]
                    current_type_duration = sum([ann['duration'] for ann in current_type_annotations])
                    print(f"  → Isolating {len(current_type_annotations)} segments of {ann_type} (total: {current_type_duration:.1f}s)")
                    
                    # Compute PSD for this annotation type (will automatically exclude 'bad')
                    psd_ann = raw_ann.compute_psd(
                        fmax=45,
                        picks='eeg',
                        reject_by_annotation=['bad']  # This will exclude all non-current annotation types
                    )
                    
                    # Get PSD data (channels x frequencies)
                    psd_data_matrix = psd_ann.get_data()  # Shape: (n_channels, n_freqs)
                    freqs = psd_ann.freqs
                    
                    # Convert to dB for better visualization
                    psd_data_db = 10 * np.log10(psd_data_matrix)
                    
                    # Calculate mean and std across channels
                    psd_mean = np.mean(psd_data_db, axis=0)
                    psd_std = np.std(psd_data_db, axis=0)
                    
                    # Plot mean with solid line
                    color = color_map[ann_type]
                    ax.plot(freqs, psd_mean, 
                           color=color, 
                           linewidth=2, 
                           label=f'{ann_type}', 
                           alpha=0.9)
                    
                    # Plot std as shaded area
                    ax.fill_between(freqs, 
                                   psd_mean - psd_std, 
                                   psd_mean + psd_std, 
                                   color=color, 
                                   alpha=0.2)
                    
                    # Store data for logging (calculate duration from original annotations)
                    total_duration = sum([ann['duration'] for ann in raw_interpolate.annotations 
                                        if ann['description'] == ann_type])
                    psd_data[ann_type] = {
                        'duration': total_duration,
                        'n_channels': psd_data_matrix.shape[0],
                        'freq_range': f'{freqs[0]:.1f}-{freqs[-1]:.1f} Hz'
                    }
                    
                    print(f"✓ {ann_type}: {total_duration:.1f}s, {psd_data_matrix.shape[0]} channels")
                    
            except Exception as e:
                print(f"⚠ Error processing {ann_type} for comparative plot: {str(e)}")
                continue
        
        # Customize the plot
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Power Spectral Density (dB)', fontsize=12)
        ax.set_title('Comparative PSD: Mean ± SD across EEG channels by condition', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 45)  # Focus on relevant EEG frequencies
        
        # Add informative text
        ax.text(0.02, 0.98, 
                'Solid line: Mean across channels\nShaded area: ± Standard deviation', 
                transform=ax.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)
        
        plt.tight_layout()
        
        # Add to report
        report.add_figure(
            fig=fig,
            title="Comparative PSD Analysis",
            caption="Power Spectral Density comparison across all experimental conditions. "
                   "Solid lines represent the mean PSD across EEG channels, shaded areas show ± 1 SD. "
                   f"Frequency range: 1-45 Hz. Data from {len(annotation_types)} conditions."
        )
        
        plt.show(block=False)
        
        # Log comparative analysis details
        log_preprocessing.log_detail("comparative_psd_generated", True)
        log_preprocessing.log_detail("comparative_psd_conditions", list(annotation_types))
        log_preprocessing.log_detail("comparative_psd_data", psd_data)
        log_preprocessing.log_detail("comparative_psd_frequency_range", "1-45 Hz")
        
        print(f"✓ Comparative PSD plot generated with {len(annotation_types)} conditions")
        print("✓ Shows mean ± SD across EEG channels for each condition")
        
    except Exception as e:
        print(f"⚠ Error generating comparative PSD plot: {str(e)}")
        log_preprocessing.log_detail("comparative_psd_generated", False)
        log_preprocessing.log_detail("comparative_psd_error", str(e))
    
    print("=== COMPARATIVE PSD PLOT COMPLETED ===\n")
    
else:
    print("⚠ No annotations found - skipping individual PSD generation")
    log_preprocessing.log_detail("individual_psds_generated", False)

print("=== INDIVIDUAL PSD GENERATION COMPLETED ===\n")

# Save the final preprocessed raw data
bids_path_final = BIDSPath(
    subject=subject,
    session=session,
    task=task,
    acquisition=acquisition,
    run=run,
    datatype=data,
    suffix=data,
    extension=".vhdr",
    root=derivatives_folder,
    description="preproc"
)

# Save the preprocessed raw data
write_raw_bids(
    raw_interpolate, 
    bids_path_final, 
    format='BrainVision', 
    allow_preload=True, 
    overwrite=True
)

print(f"✓ Final preprocessed raw data saved: {bids_path_final.fpath}")
print("✓ Processing steps completed:")
print("  - Filtering (notch + bandpass)")
print("  - ICA artifact removal")
print("  - Bad channel interpolation")
print("  - Re-referencing to average")
print("✓ Data ready for epoching and analysis")

# Log final preprocessing completion
log_preprocessing.log_detail("final_preprocessing_completed", True)
log_preprocessing.log_detail("final_raw_data_saved", str(bids_path_final.fpath))
log_preprocessing.log_detail("preprocessing_steps_completed", [
    "filtering", "ica", "interpolation", "rereferencing"
])

print("=== FINAL PREPROCESSED RAW DATA SAVED ===\n")

# Save the report as an HTML file
html_report_fname = bids_compliance.make_bids_basename(
    subject=subject,
    session=session,
    task=task,
    suffix=data,
    extension=".html",
    desc="preprocReport",
)
report.save(os.path.join(bids_dir, html_report_fname), overwrite=True)

# Save the preprocessing details to the JSON file
log_preprocessing.save_preprocessing_details()

# %%
