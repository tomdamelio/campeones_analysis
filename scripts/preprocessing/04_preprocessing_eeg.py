#!/usr/bin/env python
# coding: utf-8

# # EEG Preprocessing for Campeones Analysis Project - Variable Duration Epochs
# 
# The following script performs EEG data preprocessing for the Campeones Analysis dataset using **variable duration epochs** based on merged_events. Each video stimulus has its own duration, so epochs are created individually.
# 
# ## Processing Steps (following MNE best practices):
# 
# 1. **Load raw data** - BrainVision format (.vhdr, .vmrk, .eeg files)
# 2. **Filtering** - Independent notch (3A) and band-pass (3B) blocks with verification (3C, 3D)
# 3. **Visual inspection of channels** - Identify and mark bad channels  
# 4. **Set Annotations** - Load events from merged_events and mark gaps as 'bad'
# 5. **ICA** - Independent Component Analysis on continuous data (excluding 'bad' segments)
# 6. **Set electrode montage** - Apply electrode positions
# 7. **Interpolate bad channels** - Using clean data after ICA
# 8. **Re-reference** - To average reference (after interpolation)
# 9. **Save preprocessed data** - Export continuous file (BIDS) with annotations
# 10. **Generate reports** - HTML report with interactive plots and PSD analysis
# 
# Note: "Variable Duration Epochs" are supported by saving continuous data 
# with precise annotations, allowing downstream analysis to slice as needed.
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
    parser.add_argument("--auto", action="store_true", help="Run automatically without interactive plots")
    
    args = parser.parse_args()
    subject = args.subject
    session = args.session
    task = args.task
    run = args.run
    acquisition = args.acquisition
    interactive = not args.auto

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
#
# R-4 (two-copy pattern): the HPF used here defines the ANALYSIS copy — the data that
# gets saved to disk as the final preprocessed file and that ICA weights are applied to.
# We use HP 0.1 Hz to preserve slow ERP components (LPP, late positivity on affective
# videos, sustained responses). A second copy with HPF 1 Hz is created just before the
# ICA fit (see section 8) because ICA needs an aggressive HPF for a good decomposition
# (Winkler et al. 2015, EMBC; MNE ICA tutorial). Filtering is linear, so the ICA solution
# fitted on the 1 Hz copy is mathematically valid to apply to the 0.1 Hz copy
# (Luck, Appendix 3: "It's OK that we've double-filtered the data. The original filtering
# is so much milder that it will be dwarfed by the new filter").
hpass = 0.1   # High-pass: 0.1 Hz — preserves slow ERP components for the analysis copy
lpass = 48.0  # Low-pass: 48 Hz. Kept over the 40 Hz default because (a) 8 Hz extra
              # high-beta bandwidth may carry decoding info, (b) the notch at 50 Hz
              # still covers line noise inside the LP transition band, (c) marginal
              # benefit for find_bads_muscle detection range. See R-5 in review doc.
filter_method = 'fir'     # FIR filter for better characteristics
filter_phase = 'zero'     # Zero-phase filtering
filter_length = 'auto'    # Automatic filter length selection

print(f"Band-pass filter configuration (ANALYSIS copy):")
print(f"  - High-pass cutoff: {hpass} Hz (preserves slow ERP components, R-4 two-copy pattern)")
print(f"  - Low-pass cutoff: {lpass} Hz (removes high-frequency artifacts)")
print(f"  - Method: {filter_method}")
print(f"  - Phase: {filter_phase}")
print(f"  - Filter length: {filter_length}")
print(f"  - A second copy with HPF 1 Hz will be created later for ICA fit")

# Apply band-pass filter independently.
# NOTE: .copy() is important here (R-12). Without it, raw_notched.filter() would
# operate in-place and raw_notched would end up in the bandpassed state too,
# destroying the notched-only signal we need later to create the wide-band ICA
# copy. As a side benefit, the PSD plot labeled "After Notch Filter" below now
# actually shows the notched-only state (previously mislabeled).
print("Applying band-pass filter...")
raw_filtered = raw_notched.copy().filter(
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
log_preprocessing.log_detail("two_copy_pattern", "analysis_0.1-48Hz_ica_1-100Hz")

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

# R-11: Ensure EEG channel positions are available BEFORE PyPREP runs.
# PyPREP's RANSAC criterion uses spherical spline interpolation over 3D electrode
# positions (see find_bad_by_ransac in pyprep/find_noisy_channels.py). Without a
# montage PyPREP would either crash or silently skip RANSAC, losing one of the
# six bad-channel criteria — specifically the one that catches floating/high-
# impedance electrodes with consistent EM pickup that the other five criteria
# tend to miss. We set the canonical BC-32 montage here with on_missing='ignore'
# because FCz is not yet present (it is added in section 7 per R-1), and we
# don't need FCz in the montage for PyPREP since FCz is the reference and thus
# not a channel in the data at this point. Section 7 will re-apply the full
# montage (including FCz) after add_reference_channels.
print("Loading canonical BC-32 montage for PyPREP (R-11)...")
script_dir = os.path.dirname(os.path.abspath(__file__))
bvef_file_path_for_prep = os.path.join(script_dir, 'BC-32_FCz_modified.bvef')
if not os.path.exists(bvef_file_path_for_prep):
    raise FileNotFoundError(f"Montage file not found at: {bvef_file_path_for_prep}")
montage_for_prep = mne.channels.read_custom_montage(bvef_file_path_for_prep)
raw_filtered.set_montage(montage_for_prep, on_missing='ignore')

# Sanity check: count EEG channels with valid 3D positions. Channels without
# positions have loc[:3] all zeros (MNE default) or NaN.
eeg_picks = mne.pick_types(raw_filtered.info, eeg=True)
positions = np.array([raw_filtered.info["chs"][i]["loc"][:3] for i in eeg_picks])
n_with_pos = int(np.sum(
    ~(np.all(positions == 0, axis=1) | np.any(np.isnan(positions), axis=1))
))
print(f"✓ Montage applied: {n_with_pos}/{len(eeg_picks)} EEG channels have valid positions")
if n_with_pos < len(eeg_picks):
    missing = [raw_filtered.ch_names[i] for i, p in zip(eeg_picks, positions)
               if np.all(p == 0) or np.any(np.isnan(p))]
    print(f"  ⚠ Channels WITHOUT positions (RANSAC won't use them): {missing}")
log_preprocessing.log_detail("pyprep_montage_applied_before_prep", True)
log_preprocessing.log_detail("pyprep_channels_with_positions", n_with_pos)

# Automatically mark bad channels using PyPREP
nd = NoisyChannels(raw_filtered, do_detrend=False, random_state=42)
nd.find_all_bads(ransac=True, channel_wise=True)  # if it slows down, set channel_wise to False
bads = nd.get_bads()
print(f"Bad channels detected (union of all criteria): {bads}")

# R-17: introspect all bad_by_* attributes populated by find_all_bads
# to log WHICH criteria triggered for each bad channel. This gives per-channel
# traceability ("ch X was flagged by RANSAC and correlation") and per-criterion
# totals that surface which artifact type is dominating a given run. Using
# introspection (rather than hard-coded attribute names) makes this robust to
# future PyPREP versions that may add new criteria. Sibling to R-10 (ICA
# component logging), applied here to the PyPREP bad-channel detection stage.
bads_by_criterion = {}
for attr in dir(nd):
    if not attr.startswith("bad_by_") or attr.startswith("_"):
        continue
    val = getattr(nd, attr)
    if isinstance(val, (list, tuple, set)):
        bads_by_criterion[attr.replace("bad_by_", "")] = list(val)

# Reverse mapping: for each flagged channel, which criteria triggered?
criteria_per_channel = {}
for ch in bads:
    criteria_per_channel[ch] = sorted([
        crit for crit, chs in bads_by_criterion.items() if ch in chs
    ])

# Human-readable summary
print("\n--- Bad channel criteria breakdown ---")
if bads:
    for ch in bads:
        crits = ", ".join(criteria_per_channel[ch]) or "(none — unexpected)"
        print(f"  {ch:<8} → {crits}")
else:
    print("  (no bad channels detected)")

print("--- Criterion totals ---")
for crit in sorted(bads_by_criterion.keys()):
    chs = bads_by_criterion[crit]
    print(f"  bad_by_{crit:<14} n={len(chs):2d}  {chs if chs else ''}")

if bads:
    raw_filtered.info["bads"] = bads

# Plot the filtered data for visual inspection to identify bad channels
filtered_browser = raw_filtered.plot(n_channels=32, block=False)
print("Navegador de datos filtrados creado. Úsalo para identificar visualmente canales malos.")

# Add the filtered data to the report
report.add_raw(raw=raw_filtered, title="Filtered Raw", psd=True)

# Log the identified bad channels + the per-criterion and per-channel breakdowns.
# This gives post-hoc QC the ability to answer: "which PyPREP criterion is
# flagging most channels across subjects?" and "was channel X flagged by one
# weak criterion or by several strong ones?".
log_preprocessing.log_detail("bad_channels", raw_filtered.info["bads"])
log_preprocessing.log_detail("bad_channels_by_criterion", bads_by_criterion)
log_preprocessing.log_detail("bad_channels_criteria_per_channel", criteria_per_channel)



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

# R-13: measure snap-to-sample jitter for merged_events onsets before converting to
# MNE annotations. When onsets in the TSV are stored with sub-sample precision, setting
# them on a raw sampled at sfreq Hz rounds to the nearest sample — the residual is the
# jitter between the "true" event time and the MNE-stored event time. At 500 Hz the
# max possible jitter is 1 ms (half a sample). For decoding with time-windowed SVC
# this is negligible, but it's worth logging once per run so that future cross-subject
# comparisons can verify jitter stayed bounded.
sfreq = raw_filtered.info['sfreq']
onsets_samples_float = events_df['onset'].values * sfreq
onsets_samples_int = np.round(onsets_samples_float).astype(int)
jitter_samples = np.abs(onsets_samples_float - onsets_samples_int)
jitter_ms = jitter_samples * 1000.0 / sfreq
print(f"Snap-to-sample jitter (events → {sfreq:.0f} Hz grid): "
      f"max={jitter_ms.max():.4f} ms, mean={jitter_ms.mean():.4f} ms")
log_preprocessing.log_detail("event_onset_sfreq", float(sfreq))
log_preprocessing.log_detail("event_onset_jitter_max_ms", float(jitter_ms.max()))
log_preprocessing.log_detail("event_onset_jitter_mean_ms", float(jitter_ms.mean()))

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
## 7. Reference, montage, and interpolation (run BEFORE ICA to satisfy ICLabel's CAR requirement)
#
# RATIONALE (R-1 + R-3 Ruta A, review v3):
# - ICLabel is designed to classify ICs from data referenced to a common average and filtered
#   between 1-100 Hz (see mne_icalabel/iclabel/label_components.py docstring; Pion-Tonachini
#   et al. 2019, NeuroImage — training set was built entirely with common average reference).
#   Running ICLabel on data with a different reference is out-of-distribution for the model.
# - Average-referencing must happen AFTER interpolating bad channels, otherwise the bad
#   channel's contamination gets spread to all other channels through the mean subtraction
#   (Cohen, 2014, Analyzing Neural Time Series Data, §7.9).
# Therefore the canonical order is: add FCz -> set montage -> interpolate bads -> set
# average reference -> ICA fit -> ICLabel.

print("\n=== REFERENCE BLOCK: add FCz + montage + interpolate + average reference ===")

# Add back the original reference channel (FCz) as an all-zero channel so it can enter the
# average reference. It gets real values once set_eeg_reference subtracts the mean.
raw_filtered = mne.add_reference_channels(raw_filtered.load_data(), ref_channels=["FCz"])

# Load the extended montage (BC-32 with FCz position) and apply it.
script_dir = os.path.dirname(os.path.abspath(__file__))
bvef_file_path = os.path.join(script_dir, 'BC-32_FCz_modified.bvef')
if not os.path.exists(bvef_file_path):
    raise FileNotFoundError(f"Montage file not found at: {bvef_file_path}")
montage = mne.channels.read_custom_montage(bvef_file_path)
raw_filtered.set_montage(montage)

# Interpolate the bad channels detected by PyPREP. reset_bads=False keeps their names in
# info["bads"] for trazabilidad (downstream code and the JSON log still see which channels
# were interpolated), while the underlying data is already clean.
print(f"Interpolating bad channels: {raw_filtered.info['bads']}")
raw_filtered.interpolate_bads(reset_bads=False)
log_preprocessing.log_detail("interpolated_channels", raw_filtered.info["bads"])

# Re-reference to common average. set_eeg_reference with projection=False applies the
# reference directly to the data (not as a projector).
raw_filtered, _ = mne.set_eeg_reference(
    raw_filtered, ref_channels="average", copy=False
)
log_preprocessing.log_detail("rereferenced_channels", "grand_average")
log_preprocessing.log_detail("reference_applied_before_ica", True)
print("✓ Data is now in common average reference + bad channels interpolated")
print("✓ Ready for ICA fit and ICLabel classification (CAR requirement satisfied)")


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

# Initialize the ICA object with the specified parameters.
# R-18: fit_params=dict(ortho=False, extended=True) configures Picard as "picard-o"
# (Ablin et al. 2018, IEEE TSP), which is mathematically equivalent to extended infomax
# but ~10x faster. This matches the decomposition ICLabel was trained on (Pion-Tonachini
# et al. 2019, NeuroImage — trained on ~6000 recordings decomposed with extended infomax)
# and eliminates the runtime warning "The provided ICA instance was fitted with a 'picard'
# algorithm. ICLabel was designed with extended infomax ICA decompositions." Without these
# flags, Picard defaults to ortho=True, extended=False (FastICA-like), which gives a
# slightly out-of-distribution decomposition for ICLabel and mis-calibrates the
# probabilities that R-7 Variant A relies on (ICLABEL_THRESHOLD=0.85, BRAIN_FLOOR=0.30).
ica = mne.preprocessing.ICA(
    n_components=n_components,
    method=method,
    fit_params=dict(ortho=False, extended=True),
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

# R-4 + R-12: wide-band two-copy pattern for ICA.
#
# The analysis copy (raw_filtered) stays at 0.1-48 Hz for downstream decoding. For the
# ICA decomposition we want a SECOND copy with a wider passband (1-100 Hz), because:
#   - find_bads_muscle works on the 20-100 Hz range (muscle peaks above our 48 Hz LP).
#   - ICLabel was trained on data filtered between 1 and 100 Hz (Pion-Tonachini 2019).
#     Giving it 1-48 Hz is out-of-distribution.
#   - ICA separates muscle components better when the 48-100 Hz range is preserved.
#
# We start from raw_notched (not raw_filtered) because filtering is destructive: the
# 48-100 Hz content was removed in section 3B and cannot be recovered. raw_notched
# still has the full spectrum minus the 50/100 Hz line noise.
#
# All the spatial preprocessing applied to raw_filtered in sections 5 (bad-channel
# detection) and 7 (FCz + montage + interpolation + CAR) must be replicated on
# raw_for_ica so the two copies differ ONLY in filter band. Otherwise ICLabel, which
# assumes common average reference, would receive the wrong spatial state.
#
# See Winkler et al. 2015 (EMBC, PMID 26737196) on HPF 1-2 Hz for ICA, and the MNE
# ICA tutorial on linearity of the unmixing matrix across filter choices.

hpass_for_ica = 1.0
lpass_for_ica = 100.0
print(f"Creating wide-band ICA copy ({hpass_for_ica}-{lpass_for_ica} Hz from raw_notched)...")

# 1. Bandpass the notched-only data to 1-100 Hz.
raw_for_ica = raw_notched.copy().filter(
    l_freq=hpass_for_ica,
    h_freq=lpass_for_ica,
    picks='eeg',
    method='fir',
    phase='zero',
    verbose=False,
)

# 2. Replicate the spatial preprocessing of raw_filtered.
#    (a) Add FCz back as an all-zero channel so it can enter the average reference.
raw_for_ica = mne.add_reference_channels(raw_for_ica.load_data(), ref_channels=["FCz"])
#    (b) Apply the same montage used in section 7.
raw_for_ica.set_montage(montage)
#    (c) Carry the same bad channels detected by PyPREP on raw_filtered.
raw_for_ica.info["bads"] = list(raw_filtered.info["bads"])
#    (d) Interpolate the bads (reset_bads=False to keep trazabilidad, matching section 7).
raw_for_ica.interpolate_bads(reset_bads=False)
#    (e) Set common average reference to match raw_filtered.
raw_for_ica, _ = mne.set_eeg_reference(raw_for_ica, ref_channels="average", copy=False)
#    (f) Copy annotations so ica.fit(..., reject_by_annotation=True) excludes the same
#        bad segments that the analysis pipeline excludes.
raw_for_ica.set_annotations(raw_filtered.annotations)

print(f"✓ raw_for_ica ready: CAR + interpolated + 1-100 Hz (same spatial state as raw_filtered)")
log_preprocessing.log_detail("ica_fit_hpass", hpass_for_ica)
log_preprocessing.log_detail("ica_fit_lpass", lpass_for_ica)
log_preprocessing.log_detail("analysis_hpass", hpass)
log_preprocessing.log_detail("analysis_lpass", lpass)
log_preprocessing.log_detail("ica_wide_band_from_raw_notched", True)

# Fit the ICA model ONLY on non-bad segments (automatic with reject_by_annotation=True).
# Note: we fit on raw_for_ica (HPF 1 Hz), but ica.apply later runs on raw_filtered (HPF 0.1 Hz).
print("Fitting ICA on good segments only (excluding 'bad' annotations)...")
ica.fit(raw_for_ica, picks='eeg', reject_by_annotation=True)
print("✓ ICA fitted successfully using only merged_events segments")

# Log ICA annotation usage
log_preprocessing.log_detail("ica_reject_by_annotation", True)
log_preprocessing.log_detail("ica_used_only_merged_events", True)
log_preprocessing.log_detail("ica_excluded_bad_segments", True)

# find EOG artifacts in the data via pattern matching, and exclude the EOG-related ICA components
eog_components, eog_scores = ica.find_bads_eog(
    inst=raw_for_ica,
    ch_name="R_EYE",  # a channel close to the eye
    # threshold=1  # lower than the default threshold
)
print(f"EOG components detected: {eog_components}")

# find ECG artifacts in the data via pattern matching, and exclude the ECG-related ICA components
ecg_components, ecg_scores = ica.find_bads_ecg(
    inst=raw_for_ica,
    ch_name="ECG",  # a channel close to the eye
    # threshold=1  # lower than the default threshold
)
print(f"ECG components detected: {ecg_components}")

# R-8: find muscle artifacts via ICA pattern matching. threshold=0.7 is more conservative
# than MNE's default of 0.5 — higher threshold means fewer components are flagged as
# muscle. Chosen because (a) CAMPEONES is a VR paradigm with expected head/neck movement,
# so a looser default risks flagging legitimate posterior/frontal brain components that
# have some high-frequency content from residual micro-movements, and (b) the final
# artifact decision is re-filtered through ICLabel + brain-floor logic below (R-7),
# so find_bads_muscle acts as a candidate generator, not a final gate. If visual
# inspection of the HTML report shows clearly-muscle components being missed, lower
# to 0.5; if brain components are being flagged, raise to 0.8.
muscle_components, muscle_scores = ica.find_bads_muscle(raw_for_ica, threshold=0.7)
print(f"Muscle components detected: {muscle_components}")
log_preprocessing.log_detail("find_bads_muscle_threshold", 0.7)
log_preprocessing.log_detail("find_bads_muscle_components", muscle_components)
# ica.plot_scores(muscle_scores, exclude=muscle_components)

# Combine all artifact components from the pattern matching methods
pattern_matching_artifacts = np.unique(ecg_components + eog_components + muscle_components)

##### Classify the components using ICLabel model #######
# run the model on the ICA components. ICLabel receives raw_for_ica (CAR + HPF 1 Hz),
# which matches its training data specification (see mne_icalabel docstring).
# We call iclabel_label_components directly to obtain the full (n_components, 7)
# probability matrix — the high-level label_components() only returns the top-class
# probability, which isn't enough for the brain-floor logic below (R-7).
from mne_icalabel.iclabel import iclabel_label_components
iclabel_proba = iclabel_label_components(raw_for_ica, ica)  # shape (n_components, 7)

# ICLabel class order (see iclabel_label_components docstring):
#   0: brain, 1: muscle artifact, 2: eye blink, 3: heart beat,
#   4: line noise, 5: channel noise, 6: other
ICLABEL_CLASSES = [
    "brain", "muscle artifact", "eye blink", "heart beat",
    "line noise", "channel noise", "other",
]
ARTIFACT_CLASSES = ["muscle artifact", "eye blink", "heart beat", "channel noise"]
BRAIN_IDX = ICLABEL_CLASSES.index("brain")

label_names = [ICLABEL_CLASSES[i] for i in iclabel_proba.argmax(axis=1)]
top_probabilities = iclabel_proba.max(axis=1)
brain_probabilities = iclabel_proba[:, BRAIN_IDX]

# ---- R-7 exclusion thresholds ----
ICLABEL_THRESHOLD = 0.85   # min top-class prob to trust an ICLabel artifact call
BRAIN_FLOOR = 0.30         # never exclude if brain prob >= this (Variante A)

# ---- Readable classification table ----
print("=== ICLabel CLASSIFICATION RESULTS ===")
print(f"{'Component':<12} {'Classification':<18} {'Top prob':<10} {'Brain prob':<11} {'Action'}")
print("-" * 75)
for i in range(len(label_names)):
    component_name = f"ICA{i:03d}"
    label = label_names[i]
    top_prob = top_probabilities[i]
    brain_prob = brain_probabilities[i]
    is_artifact_call = label in ARTIFACT_CLASSES and top_prob >= ICLABEL_THRESHOLD
    if is_artifact_call and brain_prob < BRAIN_FLOOR:
        action = "→ EXCLUDE"
    elif label == "brain":
        action = "→ KEEP"
    else:
        action = "→ REVIEW"
    print(f"{component_name:<12} {label:<18} {top_prob:<10.3f} {brain_prob:<11.3f} {action}")

# Summary statistics
label_counts = {}
for label in label_names:
    label_counts[label] = label_counts.get(label, 0) + 1

print(f"\n=== SUMMARY ===")
print(f"Total components: {len(label_names)}")
for label, count in sorted(label_counts.items()):
    percentage = (count / len(label_names)) * 100
    print(f"{label:<18}: {count:2d} ({percentage:4.1f}%)")

auto_exclude_types = ARTIFACT_CLASSES  # alias kept for downstream logging
auto_exclude_count = sum(1 for label in label_names if label in auto_exclude_types)
brain_count = sum(1 for label in label_names if label == "brain")

print(f"\n=== RECOMMENDATIONS ===")
print(f"Auto-exclude candidates: {auto_exclude_count}/{len(label_names)} ({(auto_exclude_count/len(label_names)*100):.1f}%)")
print(f"Brain components: {brain_count}/{len(label_names)} ({(brain_count/len(label_names)*100):.1f}%)")
print(f"Components to review: {len(label_names) - auto_exclude_count - brain_count}")
print("=" * 75)

# ---- R-7 Variante A: exclusion logic ----
# A component is a candidate for exclusion if EITHER:
#   (a) pattern matching flagged it (find_bads_eog/ecg/muscle), OR
#   (b) ICLabel classified it as artifact with top prob >= ICLABEL_THRESHOLD.
# A candidate is ACTUALLY excluded only if its brain probability is < BRAIN_FLOOR
# (the floor applies to BOTH pattern matching and ICLabel — Variante A).
# This replaces the previous AND-intersection logic and the arbitrary
# `eog_components[0] < 3` heuristic.
pattern_set = set(int(i) for i in pattern_matching_artifacts.tolist())
iclabel_set = set(
    int(i) for i in range(len(label_names))
    if label_names[i] in ARTIFACT_CLASSES and top_probabilities[i] >= ICLABEL_THRESHOLD
)
candidate_exclusions = pattern_set | iclabel_set

to_exclude = sorted([
    idx for idx in candidate_exclusions
    if brain_probabilities[idx] < BRAIN_FLOOR
])
vetoed_by_brain_floor = sorted(candidate_exclusions - set(to_exclude))

ica.exclude = to_exclude

print(f"\n=== R-7 EXCLUSION DECISION (threshold={ICLABEL_THRESHOLD}, brain floor={BRAIN_FLOOR}) ===")
print(f"Pattern-matching candidates:   {sorted(pattern_set)}")
print(f"ICLabel candidates (>={ICLABEL_THRESHOLD}): {sorted(iclabel_set)}")
print(f"Vetoed by brain floor (<{BRAIN_FLOOR}): {vetoed_by_brain_floor}")
print(f"Final ica.exclude:             {to_exclude}")
print("=" * 75)

# (Optional) Plot the ICA components for visual inspection
# ica.plot_components(inst=epochs_clean, picks=range(15))

# Plot the sources identified by ICA. We pass raw_for_ica (the copy on which ICA was
# fitted) so the source time courses shown match the actual decomposition.
if interactive:
    print("Opening interactive ICA sources plot...")
    ica.plot_sources(raw_for_ica, block=True, show=True)

    print("Opening interactive ICA components plot (topomaps)...")
    ica.plot_components(inst=raw_for_ica, show=True)

    plt.show(block=True)

# R-14: Add the ICA results to the report with before/after overlay plots enabled.
# Passing inst=raw_for_ica (the fit copy, 1–100 Hz CAR) triggers MNE to generate:
#   (a) topographies for all components,
#   (b) source time courses for the excluded components,
#   (c) plot_overlay(): the signal with and without ica.exclude applied — this is
#       the critical sanity check to detect ICA over-correction (neural signal being
#       removed alongside artifact). Because ica.exclude was set above (line ~953),
#       the overlay shows the actual exclusion decision, not a placeholder.
# n_jobs=1 keeps the report deterministic; if reports become slow on long runs,
# raise to 2 or 4.
report.add_ica(
    ica,
    title="ICA",
    inst=raw_for_ica,
    n_jobs=1,
)

# Apply the ICA solution to the ANALYSIS copy (HPF 0.1 Hz). This is the key step of the
# two-copy pattern: the unmixing matrix learned from raw_for_ica is transferred here to
# remove artifact components from raw_filtered while preserving its slow ERP content.
raw_ica = ica.apply(inst=raw_filtered)

print("✓ ICA applied to preprocessed data")
print(f"✓ ICA excluded {len(ica.exclude)} components")
print(f"✓ Final preprocessed data ready for epoching and analysis")

# Save the full ICA object (unmixing matrix + ica.exclude) for post-hoc traceability.
# BIDS-derivatives naming convention: *_desc-ica_ica.fif
# Loading it back with mne.preprocessing.read_ica() recovers the exact decomposition
# and exclusion list — enables sensitivity analyses without re-fitting ICA.
ica_fname = os.path.join(
    derivatives_folder,
    f"sub-{subject}", f"ses-{session}", "eeg",
    f"sub-{subject}_ses-{session}_task-{task}_acq-{acquisition}_run-{run}_desc-ica_ica.fif",
)
os.makedirs(os.path.dirname(ica_fname), exist_ok=True)
ica.save(ica_fname, overwrite=True)
print(f"✓ ICA object saved: {ica_fname}")

# Log the ICA parameters and excluded components
log_preprocessing.log_detail("ica_components_excluded", ica.exclude)
log_preprocessing.log_detail("ica_method", method)
log_preprocessing.log_detail("ica_max_iter", max_iter)
log_preprocessing.log_detail("ica_random_state", random_state)
log_preprocessing.log_detail("ica_object_saved", ica_fname)

# R-7 exclusion decision parameters
log_preprocessing.log_detail("ica_exclusion_iclabel_threshold", ICLABEL_THRESHOLD)
log_preprocessing.log_detail("ica_exclusion_brain_floor", BRAIN_FLOOR)
log_preprocessing.log_detail("ica_exclusion_pattern_candidates", sorted(pattern_set))
log_preprocessing.log_detail("ica_exclusion_iclabel_candidates", sorted(iclabel_set))
log_preprocessing.log_detail("ica_exclusion_vetoed_by_brain_floor", vetoed_by_brain_floor)

# Mini-R-10: per-component detail for every excluded component
log_preprocessing.log_detail("ica_exclusions_detail", [
    {
        "component": f"ICA{idx:03d}",
        "iclabel_class": label_names[idx],
        "iclabel_top_prob": float(top_probabilities[idx]),
        "iclabel_brain_prob": float(brain_probabilities[idx]),
        "by_pattern_matching": idx in pattern_set,
        "by_iclabel": idx in iclabel_set,
    }
    for idx in ica.exclude
])

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
## 9. Final preprocessed data
#
# Reference, montage and interpolation were already applied in section 7 (before ICA fit),
# per R-1 + R-3 Ruta A of the review. At this point raw_ica is already in common average
# reference with bad channels interpolated and ICA artifacts removed — it is the final
# preprocessed raw.
# We keep the name `raw_interpolate` as an alias to avoid churn in the downstream plotting,
# reporting and saving code.
raw_interpolate = raw_ica

# Add the final preprocessed raw data to the report.
report.add_raw(
    raw=raw_interpolate, title="Final (CAR + interp + ICA)", psd=True
)

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
