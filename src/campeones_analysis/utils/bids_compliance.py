import json
import os
from pathlib import Path

import mne
import pandas as pd
from mne_bids import BIDSPath, write_raw_bids

#### BIDS ####


def make_bids_basename(
    subject, session, task, suffix, extension, desc=None, acq=None, run=None
):
    """
    Create a BIDS-compliant basename.

    Parameters:
    subject (str): Subject ID
    session (str): Session ID
    task (str): Task name
    suffix (str): Data suffix (e.g., 'eeg', 'gaze')
    extension (str): File extension
    desc (str, optional): Description for the data
    acq (str, optional): Acquisition parameter (e.g., 'a' for different setup)
    run (str, optional): Run number (e.g., '01', '02')

    Returns:
    str: BIDS-compliant basename
    """
    # Start with required entities
    basename = f"sub-{subject}_ses-{session}_task-{task}"

    # Add optional entities if provided
    if acq is not None:
        basename += f"_acq-{acq}"
    if run is not None:
        basename += f"_run-{run}"
    if desc is not None:
        basename += f"_desc-{desc}"

    # Add suffix and extension
    basename += f"_{suffix}{extension}"

    return basename


def save_raw_bids_compliant(subject, session, task, data_type, raw, root_folder):
    """
    Save raw data in BIDS format.

    Parameters
    ----------
    subject : str
        Subject ID.
    session : str
        Session ID.
    task : str
        Task name.
    data_type : str
        Type of data ('eeg', 'gaze', etc.).
    raw : mne.io.Raw
        Raw data object.
    root_folder : str
        Root folder for BIDS dataset.
    """
    if data_type == "eeg":
        # Define the BIDS path for EEG data
        bids_path = BIDSPath(
            subject=subject,
            session=session,
            task=task,
            datatype="eeg",
            root=root_folder,
            extension=".vhdr",
            check=True,
        )

        # Convert events to BrainVision format if they exist
        events_array = None
        event_id = None
        event_metadata = None
        extra_columns_descriptions = None

        if raw.annotations:
            events, event_id = mne.events_from_annotations(raw)
            if len(events) > 0:
                events_array = events
                # Create event metadata as DataFrame
                event_metadata = pd.DataFrame(
                    {
                        "onset": events[:, 0] / raw.info["sfreq"],  # Convert to seconds
                        "duration": events[:, 1]
                        / raw.info["sfreq"],  # Convert to seconds
                        "description": [str(desc) for desc in events[:, 2]],
                    }
                )
                # Create extra columns descriptions
                extra_columns_descriptions = {
                    "onset": "Event onset in seconds",
                    "duration": "Event duration in seconds",
                    "description": "Event description",
                }

        # Use the bids_path to save your data
        write_raw_bids(
            raw,
            bids_path,
            format="BrainVision",
            allow_preload=True,
            overwrite=True,
            events=events_array,
            event_id=event_id,
            event_metadata=event_metadata,
            extra_columns_descriptions=extra_columns_descriptions,
        )
        print(f"Data saved at BIDS path: {bids_path}")

    elif data_type == "gaze":
        # Define the BIDS path for gaze data
        bids_basename = make_bids_basename(
            subject=subject,
            session=session,
            task=task,
            suffix="gaze",
            extension=".vhdr",
        )
        gaze_dir = os.path.join(root_folder, f"sub-{subject}", f"ses-{session}", "gaze")
        os.makedirs(gaze_dir, exist_ok=True)
        gaze_filepath = os.path.join(gaze_dir, bids_basename)

        # raw.save(gaze_filepath, overwrite=True)
        mne.export.export_raw(
            gaze_filepath, raw, fmt="brainvision", add_ch_type=True, overwrite=True
        )
        print(f"Gaze data saved at: {gaze_filepath}")

        # Metadata for gaze data
        gaze_metadata = {
            "TaskName": task,
            "Manufacturer": "Gazepoint GP3",
            "PowerLineFrequency": "n/a",
            "SamplingFrequency": raw.info["sfreq"],
            "SoftwareFilters": "n/a",
            "RecordingDuration": raw.times[-1],
            "RecordingType": "continuous",
            "GazeChannelCount": len(raw.ch_names),
        }
        gaze_metadata_path = gaze_filepath.replace(".vhdr", ".json")
        with open(gaze_metadata_path, "w") as f:
            json.dump(gaze_metadata, f, indent=4)
        print(f"Gaze metadata saved at: {gaze_metadata_path}")

        # Event handling: save events to .tsv
        events, event_id = mne.events_from_annotations(raw)
        events_df = pd.DataFrame(events, columns=["onset", "duration", "description"])
        events_path = os.path.join(
            gaze_dir, f"sub-{subject}_ses-{session}_task-{task}_events.tsv"
        )
        events_df.to_csv(events_path, sep="\t", index=False)
        print(f"Events saved at: {events_path}")

        # Event metadata
        events_metadata = {
            "onset": {"Description": "Event onset", "Units": "seconds"},
            "duration": {"Description": "Event duration", "Units": "seconds"},
            "description": {"Description": "Event description", "event_id": event_id},
        }
        events_metadata_path = events_path.replace(".tsv", ".json")
        with open(events_metadata_path, "w") as f:
            json.dump(events_metadata, f, indent=4)
        print(f"Events metadata saved at: {events_metadata_path}")


def save_epoched_bids(
    epoched_data, root_path, subject, session, task, data, desc, events, event_id
):
    """
    Save epoched data in BIDS format.

    Parameters
    ----------
    epoched_data : mne.Epochs
        The epoched data to be saved.
    root_path : str
        The root path of the BIDS dataset.
    subject : str
        Subject ID.
    session : str
        Session ID.
    task : str
        Task name.
    desc : str
        Descriptor for the dataset.
    """
    if not isinstance(epoched_data, mne.Epochs):
        raise ValueError("epoched_data must be an instance of mne.Epochs")

    # Create BIDS path
    bids_fname = make_bids_basename(
        subject=subject,
        session=session,
        task=task,
        suffix=data,
        extension=".fif",
        desc=desc,
    )
    bids_directory = os.path.join(root_path, f"sub-{subject}", f"ses-{session}", data)
    # Ensure the directory exists
    if not os.path.exists(bids_directory):
        os.makedirs(bids_directory)

    bids_path = os.path.join(bids_directory, bids_fname)
    # Save the epoched data
    epoched_data.save(bids_path, overwrite=True)

    # Create events file
    event_id = event_id
    events_fname = os.path.join(
        bids_directory,
        make_bids_basename(
            subject=subject,
            session=session,
            task=task,
            suffix="events",
            extension=".tsv",
            desc=desc,
        ),
    )
    events_df = pd.DataFrame(events, columns=["onset", "duration", "description"])
    events_df.to_csv(events_fname, sep="\t", index=False)
    print(f"Events saved at: {events_fname}")

    # Event metadata
    events_metadata = {
        "onset": {"Description": "Event onset", "Units": "seconds"},
        "duration": {"Description": "Event duration", "Units": "seconds"},
        "description": {"Description": "Event description", "event_id": event_id},
    }
    events_metadata_path = events_fname.replace(".tsv", ".json")
    with open(events_metadata_path, "w") as f:
        json.dump(events_metadata, f, indent=4)
    print(f"Events metadata saved at: {events_metadata_path}")

    # Create JSON sidecar metadata for the epochs
    sidecar_json_fname = bids_path.replace(".fif", ".json")
    json_metadata = {
        "TaskName": task,
        "Manufacturer": "Brain Products",
        "RecordingType": "epoched",
        "SamplingFrequency": epoched_data.info["sfreq"],
        "PowerLineFrequency": epoched_data.info["line_freq"],
        "NotchFilter": epoched_data.info["line_freq"],
        "SoftwareFilters": str(epoched_data.info["highpass"])
        + "-"
        + str(epoched_data.info["lowpass"])
        + " Hz",
        "EEGReference": "CAR",
        "EEGGround": "AFz",
        "EEGPlacementScheme": "based on the extended 10/20 system",
        "EEGChannelCount": epoched_data.info["nchan"],
        "EOGChannelCount": 0,
        "ECGChannelCount": 0,
        "EMGChannelCount": 0,
        "MiscChannelCount": 0,
        "TriggerChannelCount": 0,
        "ICA": True,
        "Epoch_count": len(epoched_data),
    }

    with open(sidecar_json_fname, "w") as f:
        json.dump(json_metadata, f, indent=4)
    print(f"Epoch object metadata saved at: {sidecar_json_fname}")


def read_epochs(root_path, subject, session, task, data, desc=None):
    """
    Read epoched data from a BIDS-compliant file.

    Parameters:
    ----------
    root_path : str
        Root path to the BIDS dataset.
    subject : str
        Subject ID.
    session : str
        Session ID.
    task : str
        Task name.
    data : str
        Data type, e.g., 'eeg' or 'gaze'.
    desc : str, optional
        Description for the data (if available).

    Returns:
    --------
    epochs : mne.Epochs
        The loaded epochs object.
    events : np.ndarray
        The events array associated with the epochs.
    """
    # Construct BIDS-compliant file name
    bids_fname = make_bids_basename(
        subject=subject,
        session=session,
        task=task,
        suffix=data,
        extension=".fif",
        desc=desc,
    )
    bids_directory = os.path.join(root_path, f"sub-{subject}", f"ses-{session}", data)

    # Full file path to the epochs file
    bids_path = os.path.join(bids_directory, bids_fname)

    if not os.path.exists(bids_path):
        raise FileNotFoundError(f"File not found: {bids_path}")

    # Load the epochs file
    epochs = mne.read_epochs(bids_path, preload=True)

    # Load events
    events_fname = make_bids_basename(
        subject=subject,
        session=session,
        task=task,
        suffix="events",
        extension=".tsv",
        desc=desc,
    )
    events_path = os.path.join(bids_directory, events_fname)

    if not os.path.exists(events_path):
        raise FileNotFoundError(f"Events file not found: {events_path}")

    # Read events file
    events_df = pd.read_csv(events_path, sep="\t")
    events = events_df[["onset", "duration", "description"]].values

    print(f"Loaded epochs from: {bids_path}")
    print(f"Loaded events from: {events_path}")

    return epochs, events


def save_evoked(
    evoked,
    derivatives_folder,
    subject,
    session,
    task,
    data,
    stim,
    resp,
    mind,
    conf,
    immersion,
):
    """
    Saves the evoked object to a specified folder structure.

    Parameters:
    evoked (mne.Evoked): The evoked object to save.
    derivatives_folder (str): Root folder where the data will be saved.
    subject (str): The subject ID (e.g., '12').
    session (str): The session ID (e.g., 'b').
    task (str): The task name (e.g., 'sartauditiva').
    data (str): The type of data (e.g., 'eeg').
    stim (str): The stimulus condition (e.g., 'go').
    resp (str): The response condition (e.g., 'correct').
    mind (str): The mind condition (e.g., 'on-task').
    conf (str): The confidence condition (e.g., 'very confident').
    immersion (str): The immersion condition (e.g., 'completely immersed').

    Returns:
    str: The full path to the saved file.
    """
    # Define path for saving the evoked data
    output_folder = os.path.join(
        derivatives_folder, f"sub-{subject}", f"ses-{session}", data, "evoked"
    )

    # Create the directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Define the filename for the evoked file
    evoked_fname = os.path.join(
        output_folder,
        f"sub-{subject}_ses-{session}_task-{task}_cond-{stim}_{resp}_{mind}_{conf}_{immersion}-ave.fif",
    )

    # Save the evoked object
    evoked.save(evoked_fname)

    return evoked_fname  # Return the file path for confirmation or future use


def load_evoked(
    derivatives_folder, subject, session, task, data, stim, resp, mind, conf, immersion
):
    """
    Loads the evoked object from a specified folder structure.

    Parameters:
    derivatives_folder (str): Root folder where the data is stored.
    subject (str): The subject ID (e.g., '12').
    session (str): The session ID (e.g., 'b').
    task (str): The task name (e.g., 'sartauditiva').
    data (str): The type of data (e.g., 'eeg').
    stim (str): The stimulus condition (e.g., 'go').
    resp (str): The response condition (e.g., 'correct').
    mind (str): The mind condition (e.g., 'on-task').
    conf (str): The confidence condition (e.g., 'very confident').
    immersion (str): The immersion condition (e.g., 'completely immersed').

    Returns:
    evoked (mne.Evoked): The loaded evoked object, or None if file does not exist.
    """
    # Define path for loading the evoked data
    evoked_fname = os.path.join(
        derivatives_folder,
        f"sub-{subject}",
        f"ses-{session}",
        data,
        "evoked",
        f"sub-{subject}_ses-{session}_task-{task}_cond-{stim}_{resp}_{mind}_{conf}_{immersion}-ave.fif",
    )

    # Check if the file exists
    if os.path.exists(evoked_fname):
        # Load the evoked object
        evoked = mne.read_evokeds(
            evoked_fname, verbose=False
        )  # Assuming the first evoked object
        return evoked
    else:
        print(f"Evoked file not found: {evoked_fname}")
        return None  # Return None if the file doesn't exist


def create_bids_metadata(root_folder):
    """
    Create or update BIDS metadata files (dataset_description.json, participants.tsv, participants.json).

    Parameters
    ----------
    root_folder : str or Path
        Root folder of the BIDS dataset.
    """
    root_folder = Path(root_folder)

    # Create dataset_description.json if it doesn't exist
    dataset_json = root_folder / "dataset_description.json"
    if not dataset_json.exists():
        metadata = {
            "Name": "Campeones EEG Dataset",
            "BIDSVersion": "1.10.0",
            "DatasetType": "raw",
            "Authors": ["Your Name"],
            "HowToAcknowledge": "Please cite this dataset if you use it in your research",
            "Funding": ["Your Funding Source"],
            "EthicsApprovals": ["Your Ethics Committee"],
            "ReferencesAndLinks": ["Relevant URLs"],
            "DatasetDOI": "DOI when available",
        }
        with open(dataset_json, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Created dataset_description.json at {dataset_json}")

    # Create participants.tsv if it doesn't exist
    participants_tsv = root_folder / "participants.tsv"
    if not participants_tsv.exists():
        participants_df = pd.DataFrame(
            columns=["participant_id", "age", "sex", "hand", "weight", "height"]
        )
        participants_df.to_csv(participants_tsv, sep="\t", index=False)
        print(f"Created participants.tsv at {participants_tsv}")

    # Create participants.json if it doesn't exist
    participants_json = root_folder / "participants.json"
    if not participants_json.exists():
        participants_metadata = {
            "participant_id": {
                "Description": "Unique participant identifier",
                "LongName": "Participant ID",
            },
            "age": {"Description": "Age of the participant", "Units": "years"},
            "sex": {
                "Description": "Sex of the participant",
                "Levels": {"M": "male", "F": "female"},
            },
            "hand": {
                "Description": "Handedness of the participant",
                "Levels": {"R": "right", "L": "left", "A": "ambidextrous"},
            },
            "weight": {"Description": "Weight of the participant", "Units": "kg"},
            "height": {"Description": "Height of the participant", "Units": "cm"},
        }
        with open(participants_json, "w") as f:
            json.dump(participants_metadata, f, indent=4)
        print(f"Created participants.json at {participants_json}")


def update_participant_info(
    root_folder, subject_id, age=None, sex=None, hand=None, weight=None, height=None
):
    """
    Update participant information in participants.tsv.

    Parameters
    ----------
    root_folder : str or Path
        Root folder of the BIDS dataset.
    subject_id : str
        Subject ID (without 'sub-' prefix).
    age : int or float, optional
        Age of the participant in years.
    sex : str, optional
        Sex of the participant ('M' or 'F').
    hand : str, optional
        Handedness of the participant ('R', 'L', or 'A').
    weight : float, optional
        Weight of the participant in kg.
    height : float, optional
        Height of the participant in cm.
    """
    root_folder = Path(root_folder)
    participants_tsv = root_folder / "participants.tsv"

    # Read existing participants.tsv or create new DataFrame
    if participants_tsv.exists():
        participants_df = pd.read_csv(participants_tsv, sep="\t")
    else:
        participants_df = pd.DataFrame(
            columns=["participant_id", "age", "sex", "hand", "weight", "height"]
        )

    # Create or update participant row
    participant_id = f"sub-{subject_id}"
    if participant_id in participants_df["participant_id"].values:
        # Update existing row
        mask = participants_df["participant_id"] == participant_id
        if age is not None:
            participants_df.loc[mask, "age"] = age
        if sex is not None:
            participants_df.loc[mask, "sex"] = sex
        if hand is not None:
            participants_df.loc[mask, "hand"] = hand
        if weight is not None:
            participants_df.loc[mask, "weight"] = weight
        if height is not None:
            participants_df.loc[mask, "height"] = height
    else:
        # Add new row
        new_row = {
            "participant_id": participant_id,
            "age": age,
            "sex": sex,
            "hand": hand,
            "weight": weight,
            "height": height,
        }
        participants_df = pd.concat(
            [participants_df, pd.DataFrame([new_row])], ignore_index=True
        )

    # Save updated participants.tsv
    participants_df.to_csv(participants_tsv, sep="\t", index=False)
    print(f"Updated participant information in {participants_tsv}")
