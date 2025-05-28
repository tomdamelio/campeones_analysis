import os

import mne

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
        "AUDIO": "stim",
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

    # Path to your .bvef file
    bvef_file_path = os.path.join("data", "BC-32.bvef")
    # Load the montage
    montage = mne.channels.read_custom_montage(bvef_file_path)

    # Apply the montage to your raw data
    raw.set_montage(montage)

    return raw


def make_joystick_mapping(raw):
    """
    - Renames the first two channels to joystick_x / joystick_y.
    - Classifies them as 'misc' in arbitrary units.
    - Detects all joystick_* channels and marks as bad
      only those that are not joystick_x or joystick_y.
    - Verifies if the unit is 'au'; if not, warns and forces 'au'.
    """
    import warnings  # Added missing warnings import

    # 1) Rename first two channels
    orig = raw.info["ch_names"][:2]
    if len(orig) < 2:
        raise ValueError("At least 2 channels are needed for joystick.")
    raw.rename_channels({orig[0]: "joystick_x", orig[1]: "joystick_y"})

    # 2) Assign misc type and set unit to au
    raw.set_channel_types({"joystick_x": "misc", "joystick_y": "misc"})
    for ch in raw.info["chs"]:
        if ch["ch_name"] in ("joystick_x", "joystick_y"):
            unit = ch.get("unit", None)
            # Unit check
            if unit is not None:
                unit_str = str(unit).lower()
                if unit_str != "au":
                    warnings.warn(
                        f"Channel {ch['ch_name']} has unit '{unit}', "
                        "expected 'au'. Will use 'au'."
                    )
            # Force arbitrary unit
            ch["unit"] = "au"
            ch["unit_mul"] = 0

    # 3) Detect all joystick channels
    joystick_chs = [ch for ch in raw.info["ch_names"] if ch.startswith("joystick_")]

    # 4) Mark as bad those that are not joystick_x / joystick_y
    bads = [ch for ch in joystick_chs if ch not in ("joystick_x", "joystick_y")]
    raw.info["bads"] = bads.copy()

    return raw


def correct_channel_types(raw):
    """
    Correct channel types in the raw object, particularly for trigger channels.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data object to correct channel types.

    Returns
    -------
    raw : mne.io.Raw
        Raw data object with corrected channel types.
    """
    # Lista de canales que deben ser TRIG
    trigger_channels = ["AUDIO", "PHOTO", "triggerStream"]

    # Crear diccionario de mapeo para los canales que necesitan corrección
    ch_type_mapping = {}
    for ch in trigger_channels:
        if ch in raw.ch_names:
            ch_type_mapping[ch] = "stim"

    # Aplicar la corrección si hay canales para corregir
    if ch_type_mapping:
        raw.set_channel_types(ch_type_mapping)
        print("Channel types corrected for trigger channels")

    return raw
