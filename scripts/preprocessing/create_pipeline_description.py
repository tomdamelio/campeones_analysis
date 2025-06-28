#!/usr/bin/env python
"""
Create dataset_description.json for the CAMPEONES preprocessing pipeline.

This script creates the required BIDS dataset_description.json file for the
campeones_preproc derivatives pipeline, following BIDS best practices.
"""

import json
import os
from pathlib import Path

def create_pipeline_description(derivatives_folder):
    """
    Create dataset_description.json for the CAMPEONES preprocessing pipeline.
    
    Parameters
    ----------
    derivatives_folder : str or Path
        Path to the pipeline-specific derivatives folder
    """
    
    # Dataset description for the CAMPEONES preprocessing pipeline
    dataset_description = {
        "Name": "CAMPEONES EEG Preprocessing Pipeline",
        "BIDSVersion": "1.8.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "campeones_preproc",
                "Version": "1.0",
                "Description": "CAMPEONES EEG preprocessing pipeline with variable duration epochs",
                "CodeURL": "https://github.com/tomdamelio/campeones_analysis",
                "Container": {
                    "Type": "micromamba",
                    "Tag": "campeones:latest"
                }
            }
        ],
        "SourceDatasets": [
            {
                "DOI": "doi:10.xxxx/xxxx",  # Update with actual DOI if available
                "URL": "https://github.com/tomdamelio/campeones_analysis",
                "Version": "1.0"
            }
        ],
        "HowToAcknowledge": "Please cite the CAMPEONES project and this preprocessing pipeline.",
        "License": "MIT",
        "ReferencesAndLinks": [
            "https://mne.tools/stable/index.html",
            "https://mne.tools/mne-bids/stable/index.html",
            "https://bids-specification.readthedocs.io/en/stable/05-derivatives/01-introduction.html"
        ],
        "DatasetDOI": "doi:10.xxxx/xxxx",  # Update with actual DOI
        "PipelineDescription": {
            "Name": "CAMPEONES EEG Preprocessing",
            "Version": "1.0",
            "Steps": [
                "Raw data loading (BrainVision format)",
                "Electrode montage verification and application",
                "Independent notch filtering (50Hz, 100Hz)",
                "Band-pass filtering (0.1-64Hz, ERP-optimized)",
                "Motion artifact detection using accelerometer",
                "Bad channel detection and interpolation",
                "Average reference",
                "Variable duration epoching based on merged_events",
                "Epoch quality assessment (amplitude-based)",
                "ICA with enhanced EOG/ECG detection",
                "Final preprocessing and consolidation"
            ],
            "Tools": [
                "MNE-Python",
                "MNE-BIDS", 
                "AutoReject",
                "PyPREP",
                "ICLabel"
            ],
            "EventIDMapping": {
                "fixation": 10,
                "calm": 20,
                "video": 30,
                "video_luminance": 40
            }
        }
    }
    
    # Create the JSON file
    description_file = Path(derivatives_folder) / "dataset_description.json"
    
    # Ensure directory exists
    description_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the file
    with open(description_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_description, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Pipeline dataset_description.json created: {description_file}")
    return description_file

if __name__ == "__main__":
    # For testing - create in current directory
    test_folder = Path("test_derivatives/campeones_preproc")
    create_pipeline_description(test_folder) 