import os

import mne
import numpy as np
from scipy import interpolate

#### EEG ####

def set_chs_montage(raw):
    rename_dict = {"FP1": "Fp1", "FP2": "Fp2", "FZ": "Fz", "CZ": "Cz", "PZ": "Pz"}

    # Rename the channels
    raw.rename_channels(rename_dict)

    channel_types = {
        "Fp1": "eeg",
        "Fp2": "eeg",
        "F3": "eeg",
        "F4": "eeg",
        "C3": "eeg",
        "C4": "eeg",
        "P3": "eeg",
        "P4": "eeg",
        "O1": "eeg",
        "O2": "eeg",
        "F7": "eeg",
        "F8": "eeg",
        "T7": "eeg",
        "T8": "eeg",
        "P7": "eeg",
        "P8": "eeg",
        "Fz": "eeg",
        "Cz": "eeg",
        "Pz": "eeg",
        "IO": "misc",
        "FC1": "eeg",
        "FC2": "eeg",
        "CP1": "eeg",
        "CP2": "eeg",
        "FC5": "eeg",
        "FC6": "eeg",
        "CP5": "eeg",
        "CP6": "eeg",
        "FT9": "eeg",
        "FT10": "eeg",
        "TP9": "eeg",
        "TP10": "eeg",
        "ECG": "ecg",
        "R_EYE": "eog",
        "L_EYE": "eog",
        "AUDIO": "misc",
        "PHOTO": "misc",
        "RESP": "resp",
        "GSR": "gsr",
        "triggerStream": "stim",
    }

    for i in range(1, 11):
        if f"muerto{i}" in raw.ch_names:
            channel_types[f"muerto{i}"] = "misc"

    raw.set_channel_types(channel_types)

    bads = ["IO"]

    for i in range(1, 11):
        if f"muerto{i}" in raw.ch_names:
            bads.append(f"muerto{i}")

    raw.info["bads"] = bads

    raw.drop_channels(raw.info["bads"])

    # Obtener la ruta del proyecto (3 niveles arriba del archivo actual)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    # Ruta absoluta al archivo .bvef
    bvef_file_path = os.path.join(project_root, "data", "BC-32.bvef")
    print(f"Cargando montaje desde: {bvef_file_path}")
    
    # Load the montage
    montage = mne.channels.read_custom_montage(bvef_file_path)
    
    # Apply the montage to your raw data
    raw.set_montage(montage)

    return raw


def make_joystick_mapping(joystick_data, joystick_timestamps, sfreq):
    """
    Create a Raw object from joystick data, resampled to match EEG data.

    Parameters
    ----------
    joystick_data : array
        Joystick data array with shape (n_samples, n_channels)
    joystick_timestamps : array
        Timestamps for joystick data
    sfreq : float
        Sampling frequency of the EEG data to match

    Returns
    -------
    raw : mne.io.Raw
        Raw object containing joystick data resampled to match EEG
    """
    # Extract X and Y axes (assuming they are the first two channels)
    if joystick_data.shape[1] < 2:
        raise ValueError("Joystick data must have at least 2 channels (X and Y)")
    
    # Extract X and Y data
    x_data = joystick_data[:, 0]
    y_data = joystick_data[:, 1]
    
    # Calculate relative timestamps (starting from 0)
    joystick_rel_time = joystick_timestamps - joystick_timestamps[0]
    
    # Create interpolation functions for X and Y
    f_x = interpolate.interp1d(joystick_rel_time, x_data, bounds_error=False, fill_value="extrapolate")
    f_y = interpolate.interp1d(joystick_rel_time, y_data, bounds_error=False, fill_value="extrapolate")
    
    # Create a new time array at the desired sampling frequency
    # Covering the same time span as the original joystick data
    duration = joystick_rel_time[-1]
    new_time = np.arange(0, duration, 1.0/sfreq)
    
    # Resample the data using the interpolation functions
    new_x = f_x(new_time)
    new_y = f_y(new_time)
    
    # Combine into a single array
    joystick_array = np.vstack([new_x, new_y])
    
    # Create info structure
    info = mne.create_info(
        ch_names=['joystick_x', 'joystick_y'],
        sfreq=sfreq,
        ch_types=['misc', 'misc']
    )
    
    # Create Raw object
    raw = mne.io.RawArray(joystick_array, info)
    
    print(f"Joystick data resampled from {len(joystick_timestamps)} points to {len(new_time)} points")
    print(f"Original duration: {duration:.2f}s, Resampled duration: {new_time[-1]:.2f}s")
    
    return raw


