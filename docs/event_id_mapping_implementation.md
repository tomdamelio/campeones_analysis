# Event ID Mapping Implementation for CAMPEONES Analysis

## Overview

This document describes the implementation of standardized event ID mapping in the EEG preprocessing pipeline for the CAMPEONES Analysis project. This enhancement ensures consistent and reproducible event codes across all subjects and analysis runs.

## Problem Statement

According to MNE-BIDS documentation, when `Raw` objects contain `Annotations`, it is **mandatory** to provide an `event_id` dictionary when calling `write_raw_bids()`. Without this mapping, MNE-BIDS generates arbitrary event codes, leading to:

- ❌ Inconsistent event codes across subjects
- ❌ Non-reproducible analysis pipelines  
- ❌ Difficulty in cross-subject comparisons
- ❌ BIDS compliance issues

## Solution: Standard CAMPEONES Event ID Mapping

### Event ID Dictionary

Based on the `trial_type` values found in `merged_events` files across all subjects, we define a standard mapping:

```python
CAMPEONES_EVENT_ID = {
    'fixation': 10,        # Baseline fixation cross condition (stim_id: 500)
    'calm': 20,           # Calm video condition (stim_id: 901)
    'video': 30,          # Affective video conditions (various stim_ids)
    'video_luminance': 40  # Luminance control videos (stim_ids: 100+)
}
```

### Code Assignment Rationale

- **10-series**: Baseline/control conditions
- **20-series**: Calm/neutral conditions  
- **30-series**: Primary experimental conditions (affective videos)
- **40-series**: Control conditions (luminance)

This spacing allows for future expansion within each category.

## Implementation Details

### 1. Definition in Preprocessing Script

The mapping is defined early in `preprocessing_eeg.py`:

```python
# %%
# 1.5. DEFINE STANDARD EVENT ID MAPPING

CAMPEONES_EVENT_ID = {
    'fixation': 10,        # Baseline fixation cross condition (500)
    'calm': 20,           # Calm video condition (901)
    'video': 30,          # Affective video conditions (various stim_ids)
    'video_luminance': 40  # Luminance control videos (100+ stim_ids)
}
```

### 2. Application in write_raw_bids Calls

The mapping is applied in **both** `write_raw_bids` calls:

#### First Call (Filtered Data)
```python
# Convert annotations to events for BIDS compliance
events_array = None
if raw_filtered.annotations:
    events, _ = mne.events_from_annotations(raw_filtered, event_id=CAMPEONES_EVENT_ID)
    if len(events) > 0:
        events_array = events

write_raw_bids(
    raw_filtered, 
    bids_path, 
    format='BrainVision', 
    allow_preload=True, 
    overwrite=True,
    events=events_array,
    event_id=CAMPEONES_EVENT_ID if events_array is not None else None
)
```

#### Second Call (Preprocessed Data)
```python
# Convert annotations to events for the preprocessed data
preproc_events_array = None
if raw_preproc.annotations:
    preproc_events, _ = mne.events_from_annotations(raw_preproc, event_id=CAMPEONES_EVENT_ID)
    if len(preproc_events) > 0:
        preproc_events_array = preproc_events

write_raw_bids(
    raw_preproc, 
    preproc_path, 
    overwrite=True, 
    format='FIF',
    events=preproc_events_array,
    event_id=CAMPEONES_EVENT_ID if preproc_events_array is not None else None
)
```

### 3. Epoching Integration

Individual epoch creation also uses the standard mapping:

```python
# Create temporary event for this epoch using standard CAMPEONES event_id mapping
event_code = CAMPEONES_EVENT_ID.get(trial_type, 999)  # Use 999 for unknown trial types
temp_event = np.array([[onset_sample, 0, event_code]])
temp_event_id = {trial_type: event_code}
```

### 4. Consolidated Epochs

When saving consolidated epochs, the mapping ensures proper event codes:

```python
# Create events and event_id for consolidated epochs using standard mapping
consolidated_events = []
for i, meta in enumerate(final_metadata_df.itertuples()):
    trial_type = meta.trial_type
    event_code = CAMPEONES_EVENT_ID.get(trial_type, 999)
    consolidated_events.append([i, 0, event_code])

# Build event_id dictionary using only the trial types present in this dataset
consolidated_event_id = {}
for trial_type in final_metadata_df['trial_type'].unique():
    if trial_type in CAMPEONES_EVENT_ID:
        consolidated_event_id[trial_type] = CAMPEONES_EVENT_ID[trial_type]
```

## Testing and Verification

### Test Script

A dedicated test script (`test_event_id_mapping.py`) verifies the implementation:

```bash
micromamba run -n campeones python scripts/preprocessing/test_event_id_mapping.py
```

The test verifies:
1. ✅ All trial types in merged_events are covered by the standard mapping
2. ✅ Generated events.tsv files use consistent codes
3. ✅ Cross-subject compatibility

### Expected Output Files

After running preprocessing with event_id mapping, the following files will contain standard event codes:

- `*_events.tsv` - BIDS-compliant events files with consistent codes
- `*_events.json` - Metadata describing the event_id mapping
- `*-epo.fif` - Epoch files with standard event codes

## Benefits

### ✅ Reproducibility
- Identical event codes across all subjects and runs
- Consistent analysis pipelines
- Version-controlled event mapping

### ✅ BIDS Compliance
- Proper events.tsv files with meaningful codes
- Documented event_id mapping in JSON sidecars
- Standards-compliant derivatives

### ✅ Analysis Efficiency
- Direct cross-subject comparisons
- Simplified group-level analyses
- Clear event code semantics

### ✅ Future-Proof
- Expandable mapping structure
- Clear documentation and testing
- Maintainable codebase

## Usage Examples

### Loading Epochs with Standard Codes
```python
import mne

# Load consolidated epochs
epochs = mne.read_epochs('sub-14_ses-vr_task-01_desc-preproc-consolidated_eeg.fif')

# Access events by standard codes
fixation_epochs = epochs[epochs.events[:, 2] == 10]  # fixation
video_epochs = epochs[epochs.events[:, 2] == 30]     # video
```

### Cross-Subject Analysis
```python
# Consistent event codes across subjects
for subject in subjects:
    epochs = load_subject_epochs(subject)
    
    # Same event codes work for all subjects
    fixation_erp = epochs[epochs.events[:, 2] == 10].average()
    video_erp = epochs[epochs.events[:, 2] == 30].average()
```

## Migration Notes

### Existing Data
- Files processed before this implementation may use different event codes
- Reprocessing with the updated script will apply standard codes
- Test script can identify files needing reprocessing

### Code Updates
- Any analysis scripts should use the standard `CAMPEONES_EVENT_ID` mapping
- Import the mapping from a central location for consistency
- Update any hardcoded event codes to use the standard mapping

## Technical Implementation Summary

| Component | Status | Description |
|-----------|---------|-------------|
| **Event ID Definition** | ✅ Implemented | Standard mapping defined in preprocessing script |
| **write_raw_bids Integration** | ✅ Implemented | Both calls use event_id parameter |
| **Epoching Integration** | ✅ Implemented | Individual epochs use standard codes |
| **Consolidation Integration** | ✅ Implemented | Consolidated epochs use standard mapping |
| **Testing Script** | ✅ Implemented | Verification of mapping implementation |
| **Documentation** | ✅ Implemented | Complete implementation guide |
| **Error Handling** | ✅ Implemented | Unknown trial types mapped to code 999 |
| **Logging** | ✅ Implemented | Event mapping details logged |

This implementation ensures that all CAMPEONES EEG data uses consistent, reproducible, and BIDS-compliant event codes for reliable scientific analysis. 