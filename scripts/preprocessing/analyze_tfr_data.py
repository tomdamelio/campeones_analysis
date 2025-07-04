#!/usr/bin/env python
# coding: utf-8

# # TFR Data Visualization for Campeones Analysis Project
# 
# This script loads concatenated TFR data for a single subject and creates
# a spectrogram-like visualization with segment boundaries marked.

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

# Add src directory to Python path
repo = Repo(os.getcwd(), search_parent_directories=True)
repo_root = repo.git.rev_parse("--show-toplevel")
src_path = os.path.join(repo_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

#%%
# LOAD CONCATENATED TFR DATA
print(f"=== LOADING CONCATENATED TFR DATA FOR SUBJECT {subject} ===")

# Define paths
derivatives_root = os.path.join(repo_root, "data", "derivatives")
trf_folder = os.path.join(derivatives_root, "trf", f"sub-{subject}")

# File paths
tfr_file = os.path.join(trf_folder, f"sub-{subject}_desc-morlet_tfr_concatenated.npz")
indices_file = os.path.join(trf_folder, f"idx_data_sub-{subject}_concatenated.npz")
columns_file = os.path.join(trf_folder, f"sub-{subject}_desc-morlet_columns_concatenated.tsv")

# Check if files exist
if not os.path.exists(tfr_file):
    raise FileNotFoundError(f"TFR file not found: {tfr_file}")
if not os.path.exists(indices_file):
    raise FileNotFoundError(f"Indices file not found: {indices_file}")
if not os.path.exists(columns_file):
    raise FileNotFoundError(f"Columns file not found: {columns_file}")

print(f"Loading TFR data from: {os.path.basename(tfr_file)}")
print(f"Loading indices from: {os.path.basename(indices_file)}")
print(f"Loading column names from: {os.path.basename(columns_file)}")

#%%
# LOAD DATA
# Load TFR data
tfr_data = np.load(tfr_file)
tfr_matrix = tfr_data['arr_0']  # Shape: (timepoints, features)
print(f"TFR data shape: {tfr_matrix.shape}")

# Load indices
indices_data = np.load(indices_file)
segment_indices = indices_data['arr_0']  # Shape: (n_segments, 2)
print(f"Segment indices shape: {segment_indices.shape}")
print(f"Number of segments: {segment_indices.shape[0]}")

# Load column names
columns_df = pd.read_csv(columns_file, sep='\t')
feature_names = columns_df['column_name'].tolist()
print(f"Number of features: {len(feature_names)}")

# Verify data consistency
if tfr_matrix.shape[1] != len(feature_names):
    raise ValueError(f"Mismatch: TFR has {tfr_matrix.shape[1]} features but columns file has {len(feature_names)}")

#%%
# ORGANIZE FEATURES BY CHANNEL AND FREQUENCY
print(f"\n=== ORGANIZING FEATURES BY CHANNEL AND FREQUENCY ===")

# Parse feature names to extract channel and frequency
channels = []
frequencies = []
feature_info = []

for feature in feature_names:
    # Feature names are like "Fp1_2hz", "Fp1_4hz", etc.
    parts = feature.split('_')
    if len(parts) == 2:
        channel = parts[0]
        freq = parts[1].replace('hz', '')
        channels.append(channel)
        frequencies.append(int(freq))
        feature_info.append({'channel': channel, 'frequency': int(freq), 'name': feature})
    else:
        print(f"Warning: Could not parse feature name: {feature}")

# Get unique channels and frequencies
unique_channels = sorted(list(set(channels)))
unique_frequencies = sorted(list(set(frequencies)))

print(f"Unique channels ({len(unique_channels)}): {unique_channels}")
print(f"Unique frequencies ({len(unique_frequencies)}): {unique_frequencies}")

# Create feature organization for plotting
feature_groups = {}
for i, info in enumerate(feature_info):
    channel = info['channel']
    if channel not in feature_groups:
        feature_groups[channel] = []
    feature_groups[channel].append({
        'index': i,
        'frequency': info['frequency'],
        'name': info['name']
    })

# Sort features within each channel by frequency
for channel in feature_groups:
    feature_groups[channel].sort(key=lambda x: x['frequency'])

#%%
# CREATE SPECTROGRAM VISUALIZATION
print(f"\n=== CREATING SPECTROGRAM VISUALIZATION ===")

# Prepare data for visualization
# We'll organize the matrix so that features are grouped by channel and ordered by frequency
organized_features = []
feature_labels = []
channel_boundaries = []

current_idx = 0
for channel in unique_channels:
    channel_start = current_idx
    for feature_info in feature_groups[channel]:
        organized_features.append(feature_info['index'])
        feature_labels.append(f"{channel}_{feature_info['frequency']}Hz")
        current_idx += 1
    channel_boundaries.append((channel_start, current_idx - 1, channel))

# Reorder the TFR matrix according to our organization
tfr_organized = tfr_matrix[:, organized_features]

# Convert to log scale for better visualization (add small constant to avoid log(0))
tfr_log = np.log10(tfr_organized + 1e-10)

print(f"Organized TFR shape: {tfr_organized.shape}")
print(f"Feature organization complete with {len(unique_channels)} channel groups")

#%%
# CREATE THE PLOT
fig, ax = plt.subplots(figsize=(20, 12))

# Create time vector (assuming 250 Hz sampling rate)
sampling_rate = 250.0  # Hz
time_vector = np.arange(tfr_matrix.shape[0]) / sampling_rate

# Create the spectrogram plot
im = ax.imshow(tfr_log.T, 
               aspect='auto',
               origin='lower',
               extent=[time_vector[0], time_vector[-1], 0, len(feature_labels)],
               cmap='viridis',
               interpolation='nearest')

# Add colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Log10(Power)', fontsize=12)

# Add vertical lines for segment boundaries
print(f"\nAdding {len(segment_indices)} segment boundaries:")
for i, (start_idx, end_idx) in enumerate(segment_indices):
    start_time = start_idx / sampling_rate
    end_time = end_idx / sampling_rate
    
    # Add vertical lines
    ax.axvline(x=start_time, color='red', linestyle='-', alpha=0.7, linewidth=1)
    ax.axvline(x=end_time, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    if i < 10:  # Print first 10 segments
        print(f"  Segment {i+1}: {start_time:.1f}s - {end_time:.1f}s ({end_time-start_time:.1f}s duration)")

if len(segment_indices) > 10:
    print(f"  ... and {len(segment_indices) - 10} more segments")

# Customize y-axis with channel grouping
y_ticks = []
y_labels = []
channel_dividers = []

for start_idx, end_idx, channel in channel_boundaries:
    # Add tick at middle of channel group
    mid_idx = (start_idx + end_idx) / 2
    y_ticks.append(mid_idx)
    y_labels.append(channel)
    
    # Add horizontal line to separate channels (except for last one)
    if end_idx < len(feature_labels) - 1:
        channel_dividers.append(end_idx + 0.5)

# Set y-axis ticks and labels
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)

# Add horizontal lines to separate channels
for divider in channel_dividers:
    ax.axhline(y=divider, color='white', linestyle='-', alpha=0.3, linewidth=0.5)

# Set labels and title
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('EEG Channels (grouped by frequency)', fontsize=12)
ax.set_title(f'Time-Frequency Representation: Subject {subject}\n'
             f'Concatenated TFR Data ({tfr_matrix.shape[0]} timepoints, {len(segment_indices)} segments)', 
             fontsize=14, fontweight='bold')

# Add frequency legend on the right
freq_text = "Frequencies:\n" + "\n".join([f"{freq} Hz" for freq in unique_frequencies])
ax.text(1.02, 0.98, freq_text, 
        transform=ax.transAxes, 
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
        fontsize=10)

# Add summary information
total_duration = time_vector[-1]
info_text = f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)\n"
info_text += f"Segments: {len(segment_indices)}\n"
info_text += f"Channels: {len(unique_channels)}\n"
info_text += f"Frequencies: {len(unique_frequencies)}\n"
info_text += "Red lines: segment boundaries"

ax.text(0.02, 0.98, info_text, 
        transform=ax.transAxes, 
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
        fontsize=10)

plt.tight_layout()
plt.show()

#%%
# DETAILED SEGMENT INFORMATION
print(f"\n=== DETAILED SEGMENT INFORMATION ===")
print(f"Subject: {subject}")
print(f"Total timepoints: {tfr_matrix.shape[0]}")
print(f"Total features: {tfr_matrix.shape[1]}")
print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Number of segments: {len(segment_indices)}")

# Calculate segment statistics
segment_durations = []
for start_idx, end_idx in segment_indices:
    duration = (end_idx - start_idx) / sampling_rate
    segment_durations.append(duration)

print(f"\nSegment duration statistics:")
print(f"  Mean duration: {np.mean(segment_durations):.1f}s")
print(f"  Median duration: {np.median(segment_durations):.1f}s")
print(f"  Min duration: {np.min(segment_durations):.1f}s")
print(f"  Max duration: {np.max(segment_durations):.1f}s")
print(f"  Total duration: {np.sum(segment_durations):.1f}s")

# Show feature organization
print(f"\nFeature organization:")
for channel in unique_channels[:5]:  # Show first 5 channels
    features = feature_groups[channel]
    feature_list = [f"{f['frequency']}Hz" for f in features]
    print(f"  {channel}: {', '.join(feature_list)}")

if len(unique_channels) > 5:
    print(f"  ... and {len(unique_channels) - 5} more channels")

print("\n=== TFR VISUALIZATION COMPLETED ===")

#%% 