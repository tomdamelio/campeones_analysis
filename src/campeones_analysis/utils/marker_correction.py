import os
import re
from typing import List

import mne
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from utils.exceptions import (
    BehNotFound,
    InconsistentAnnotationsWithBeh,
    StimChannelNotFound,
)


def correct_raw_markers(raw, beh_data, task):
    """
    Correct markers in the raw data based on the behavioral data, and onset times.

    Parameters:
    raw (mne.io.Raw): MNE Raw object.
    beh_data (pandas.DataFrame): DataFrame with behavioral data.
    task (str): Task identifier.

    """
    auditive_empirical_delta = 0.21
    visual_empirical_delta = 0.96
    total_markers = 468

    if task == "resting":
        return raw

    channel_audio_name = "AUDIO" if task != "sartvisual" else "PHOTO"
    descriptions_to_filter = [r"\d+"] if task == "narrative" else ["GO", "NOGO"]
    filtered_annotations, filtered_ids = filter_annotations(raw, descriptions_to_filter)
    sampling_rate = raw.info["sfreq"]
    is_manually_corrected = False

    new_marker_descriptions = propagate_and_generate_marker_names(beh_data, task)

    if len(filtered_annotations.onset) != len(new_marker_descriptions):
        if (
            len(filtered_annotations.onset) == 0
            and len(new_marker_descriptions) == total_markers
        ):
            print("-----------------------------")
            print(
                f"Tarea: {task}. No se han encontrado marcadores en el archivo de EEG. Se intentarán agregar los marcadores de la tarea manualmente"
            )
            print("-----------------------------\n")

            if task == "narrative" or task == "sartauditiva":
                raw = add_sinusoidal_markers(raw, channel_audio_name, 0.5)
            else:
                raw = add_pleateau_markers(raw, channel_audio_name, 0.1)
        else:
            raise InconsistentAnnotationsWithBeh(
                f"Number of annotations does not match with markers in Beh Data. Found {len(filtered_annotations.onset)} annotations in Raw and {len(new_marker_descriptions)} markers in beh"
            )

    try:
        stim_channel_signal_values = get_stim_channel_data(raw, channel_audio_name)
        marker_times = adjust_markers(
            stim_channel_signal_values, filtered_annotations.onset, sampling_rate
        )
    except StimChannelNotFound:
        print("-----------------------------")
        print(
            f"Tarea: {task}. No se ha encontrado el canal de estímulo. Se corregirán los onsets teniendo en cuenta el error promedio"
        )
        print("-----------------------------\n")

        if task == "sartvisual":
            marker_times = filtered_annotations.onset + visual_empirical_delta
        else:
            marker_times = filtered_annotations.onset + auditive_empirical_delta

        is_manually_corrected = True

    unique_times = set(marker_times)

    if len(unique_times) < len(marker_times):
        print("-----------------------------")
        print(
            f"Tarea: {task}. Se han encontrado tiempos repetidos. Es posible que el canal de estímulo no haya funcionado bien. Se corregirán los onsets teniendo en cuenta el error promedio"
        )
        print("-----------------------------\n")

        if task == "sartvisual":
            marker_times = filtered_annotations.onset + visual_empirical_delta
        else:
            marker_times = filtered_annotations.onset + auditive_empirical_delta

        is_manually_corrected = True

    if not is_manually_corrected:
        average_correction = np.mean(marker_times - filtered_annotations.onset)
        std_correction = np.std(marker_times - filtered_annotations.onset)
        print("-----------------------------")
        print(
            f"Tarea: {task}. Se han corregido exitosamente los onsets. Promedio: {average_correction}. Desvío estándar: {std_correction}"
        )
        print("-----------------------------\n")

    new_annotations = create_new_annotations(
        raw,
        filtered_annotations.onset,
        False,
        [],
        new_marker_descriptions,
        marker_times,
    )
    raw.annotations.delete(filtered_ids)
    raw.set_annotations(raw.annotations + new_annotations)

    return raw


def load_data(results_folder, subject, session, task, data):
    """Load data from a FIF file.

    Parameters:
    results_folder (str): Path to the results folder.
    subject (str): Subject identifier.
    session (str): Session identifier.
    task (str): Task identifier.
    data (str): Data type (Ej: 'eeg').

    Returns:
    mne.io.Raw: MNE Raw object with the loaded data.
    """
    filename = f"sub-{subject}/ses-{session}/{data}/sub-{subject}_ses-{session}_task-{task}_{data}.fif"
    file_path = os.path.join(results_folder, filename)
    return mne.io.read_raw(file_path, verbose=True, preload=True)


def load_beh_data(results_folder, subject, session, task, data):
    """Load Behavioral data from a CSV file.

    Parameters:
    results_folder (str): Path to the results folder.
    subject (str): Subject identifier.
    session (str): Session identifier.
    task (str): Task identifier.
    data (str): Data type (Ej: 'eeg').

    Returns:
    pandas.DataFrame: DataFrame with the loaded data.
    """
    filename = f"sub-{subject}/ses-{session}/{data}/sub-{subject}_ses-{session}_task-{task}_{data}.csv"
    file_path = os.path.join(results_folder, filename)
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise BehNotFound()


def filter_annotations(raw, descriptions):
    """Filter annotations in the data based on specified descriptions.

    Parameters:
    raw (mne.io.Raw): MNE Raw object.
    descriptions (list): List of descriptions to filter (e.g., ['GO', 'NOGO']). They can be strings
    or Regex patterns.

    Returns:
    tuple[mne.Annotations, list]: Tuple with the filtered annotations and their indices.
    """
    filtered_annotations = mne.Annotations(onset=[], duration=[], description=[])
    ids = []
    for id, annotation in enumerate(raw.annotations):
        if annotation["description"] in descriptions or any(
            re.match(description, annotation["description"])
            for description in descriptions
        ):
            filtered_annotations.append(
                annotation["onset"], annotation["duration"], annotation["description"]
            )
            ids.append(id)

    return filtered_annotations, ids


def get_stim_channel_data(raw, channel_name):
    """Get data from the specified audio channel.

    Parameters:
    raw (mne.io.Raw): MNE Raw object with data.
    channel_name (str): Name of the stim channel.

    Returns:
    array: Values of the stim signal.
    """

    try:
        channel_index = mne.pick_channels(raw.info["ch_names"], include=[channel_name])
        audio_channel_data = raw[channel_index]
        audio_signal_values = audio_channel_data[0][0]
        return audio_signal_values

    except ValueError:
        raise StimChannelNotFound()


def adjust_markers(stim_channel_signal_values, marker_onsets, sampling_rate):
    """Adjust markers based on peaks in the audio channel.

    Parameters:
    stim_channel_signal_values (array): Values of the stim signal.
    marker_onsets (array): Array with marker times.
    sampling_rate (float): Sampling rate of the data.

    Returns:
    list: List of adjusted marker times.
    """
    adjusted_marker_indices = []
    peaks, _ = find_peaks(
        np.abs(stim_channel_signal_values),
        height=np.quantile(np.abs(stim_channel_signal_values), q=0.95),
    )

    for marker_time in marker_onsets:
        marker_index = int(marker_time * sampling_rate)
        closest_peak_index = peaks[np.argmin(np.abs(peaks - marker_index))]
        adjusted_marker_indices.append(closest_peak_index)

    return [idx / sampling_rate for idx in adjusted_marker_indices]


def propagate_and_generate_marker_names(df_beh, task):
    """Propagate non GO/NO GO information and generate marker names based on the CSV file.

    Parameters:
    df_beh (pandas.DataFrame): DataFrame with behavioral data.
    task (str): Task identifier.

    Returns:
    list: List of marker names.
    """
    marker_names = []

    if task == "narrative":
        for _, row in df_beh.iterrows():
            if pd.notna(row["segment"]):
                current_segment = row["segment"]
                current_probe_response = row["probe_response"]
                current_confidence_response = row["confidence_response"]
                current_depth_response = row["depth_response"]
                marker_name = f"segment-{int(current_segment)}/{current_probe_response}/{current_confidence_response}/{current_depth_response}"
                marker_names.append(marker_name)
        return marker_names
    else:
        current_segment = 16
        current_probe_response = None
        current_confidence_response = None
        current_depth_response = None
        distance_to_segment_end = 0

        for _, row in df_beh[::-1].iterrows():
            condition = row["condition"]

            if pd.notna(row["probe_key"]):
                distance_to_segment_end = 0
                current_probe_response = row["probe_response"]
                current_confidence_response = row["confidence_response"]
                current_depth_response = row["depth_response"]
                current_segment -= 1

            if condition in ["go", "nogo"]:
                distance_to_segment_end += 1
                is_correct = (row["stim"] == 3 and row["key_response"] != "space") or (
                    row["stim"] != 3 and row["key_response"] == "space"
                )
                thisTrialN = row[".thisTrialN"]
                marker_name = f"{condition}/{'correct' if is_correct else 'incorrect'}/segment-{int(current_segment)}/{int(distance_to_segment_end)},{int(thisTrialN)}/{current_probe_response}/{current_confidence_response}/{current_depth_response}"
                marker_names.append(marker_name)

        return marker_names[::-1]


def create_new_annotations(
    raw: mne.io.Raw,
    marker_onsets: List[float],
    should_include_old_markers: bool,
    old_marker_descriptions: List[str],
    new_marker_descriptions: List[str],
    adjusted_marker_times: List[float],
):
    """Create new annotations based on the adjusted markers.

    Parameters:
    raw (mne.io.Raw): MNE Raw object.
    marker_onsets (list): List of marker times.
    should_include_old_markers (bool): Whether to include old markers in the new annotations.
    old_marker_descriptions (list): List of old marker descriptions.
    new_marker_descriptions (list): List of new marker descriptions.
    adjusted_marker_times (list): List of adjusted marker times.

    Returns:
    mne.Annotations: New annotations based on the adjusted markers.
    """
    sampling_rate = raw.info["sfreq"]
    old_marker_indices = [int(time * sampling_rate) for time in marker_onsets]

    if should_include_old_markers:
        total_markers = len(old_marker_indices) + len(adjusted_marker_times)
    else:
        total_markers = len(adjusted_marker_times)

    events = np.matrix(
        [[None] * 3] * total_markers
    )  # 3 columnas: onset, duration, event_description

    for i, idx in enumerate(adjusted_marker_times):
        events[i] = [int(idx * sampling_rate), 0, new_marker_descriptions[i]]

    if should_include_old_markers:
        for i, idx in enumerate(old_marker_indices, start=len(adjusted_marker_times)):
            events[i] = [
                idx,
                0,
                old_marker_descriptions[i - len(adjusted_marker_times)],
            ]

    event_times = events[:, 0] / sampling_rate
    annotations_description = np.reshape(events[:, 2], newshape=-1).tolist()[0]
    new_annotations = mne.Annotations(
        onset=np.reshape(event_times, newshape=-1).tolist()[0],
        duration=np.zeros(len(event_times)),
        description=annotations_description,
    )

    return new_annotations


### Manual Preprocessing functions


def _remove_consecutive_times(times):
    times_desc = np.array(times)[::-1]
    diff = np.diff(times_desc, prepend=np.inf)
    indices = diff < -2

    return times_desc[indices][::-1].tolist()


def add_sinusoidal_markers(raw, channel_name, duration, threshold=0.5):
    """Add markers to the raw object based on sinusoidal waves in the narrative task audio channel.
    Used only if the narrative task audio channel does not have markers.

    Parameters:
    raw (mne.io.Raw): MNE Raw object.
    channel_name (str): Name of the audio channel.
    duration (float): Duration of the sinusoidal wave in seconds.
    threshold (float): Amplitude threshold for detecting sinusoidal waves.

    Returns:
    mne.io.Raw: Raw object with new sinusoidal markers added.
    """
    audio_signal_values, _ = get_stim_channel_data(raw, channel_name)
    sampling_rate = raw.info["sfreq"]
    signal_length = len(audio_signal_values)
    window_size = int(sampling_rate * duration)
    detected_times = []

    for start in range(
        0, signal_length - window_size, window_size // 2
    ):  # Slide window
        segment = audio_signal_values[start : start + window_size]
        yf = fft(segment)
        xf = fftfreq(window_size, 1 / sampling_rate)
        amplitudes = np.abs(yf)
        significant_amplitudes = amplitudes[xf > 0]
        peak_index = np.argmax(significant_amplitudes)
        peak_amplitude = significant_amplitudes[peak_index]

        if (
            peak_amplitude > threshold * np.max(amplitudes)
            and np.sum(significant_amplitudes > (threshold * peak_amplitude)) <= 3
        ):
            detected_times.append(start / sampling_rate)

    detected_times = _remove_consecutive_times(detected_times)
    new_annotations = mne.Annotations(
        onset=detected_times,
        duration=[0] * len(detected_times),
        description=["Sinusoidal Wave"] * len(detected_times),
    )
    raw.set_annotations(raw.annotations + new_annotations)

    return raw


def add_pleateau_markers(raw, channel_name, duration, threshold=0.5):
    """Add markers to the raw object based on plateau waves in the narrative task audio channel.
    Used only if the narrative task audio channel does not have markers.

    Parameters:
    raw (mne.io.Raw): MNE Raw object.
    channel_name (str): Name of the audio channel.
    duration (float): Duration of the plateau wave in seconds.
    threshold (float): Amplitude threshold for detecting plateau waves.

    Returns:
    mne.io.Raw: Raw object with new plateau markers added.
    """
    audio_signal_values, _ = get_stim_channel_data(raw, channel_name)
    sampling_rate = raw.info["sfreq"]
    diff_signal = np.diff(audio_signal_values)

    detected_times = []
    for idx, value in enumerate(diff_signal):
        if value > threshold and diff_signal[idx + 1] < -threshold:
            detected_times.append((idx + 1) / sampling_rate)

    new_annotations = mne.Annotations(
        onset=detected_times,
        duration=[0] * len(detected_times),
        description=["GO"] * len(detected_times),
    )
    raw.set_annotations(raw.annotations + new_annotations)

    return raw
