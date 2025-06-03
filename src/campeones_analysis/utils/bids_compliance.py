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
            json.dump(gaze_metadata, f, indent=4, ensure_ascii=True)
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
            json.dump(events_metadata, f, indent=4, ensure_ascii=True)
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
        json.dump(events_metadata, f, indent=4, ensure_ascii=True)
    print(f"Events metadata saved at: {events_metadata_path}")

    # Create JSON sidecar metadata for the epochs
    sidecar_json_fname = bids_path.replace(".fif", ".json")
    json_metadata = {
        "TaskName": task,
        "Manufacturer": "Brain Products",
        "RecordingType": "epoched",
        "RecordingDuration": float(epoched_data.times[-1] - epoched_data.times[0]),
        "SamplingFrequency": float(epoched_data.info["sfreq"]),
        "PowerLineFrequency": 50.0,
        "SoftwareFilters": "n/a",
        "EpochLength": float(epoched_data.times[-1] - epoched_data.times[0]),
        "EpochCount": len(epoched_data),
    }
    with open(sidecar_json_fname, "w") as f:
        json.dump(json_metadata, f, indent=4, ensure_ascii=True)
    print(f"Metadata saved at: {sidecar_json_fname}")


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
    Creates or updates BIDS metadata files:
    - dataset_description.json
    - participants.tsv
    - participants.json

    Parameters
    ----------
    root_folder : str | Path
        Root folder of the BIDS dataset.
    """
    root_folder = Path(root_folder)

    # --------------------------------------------------
    # dataset_description.json
    # --------------------------------------------------
    dataset_json = root_folder / "dataset_description.json"
    if not dataset_json.exists():
        metadata = {
            "Name": "CAMPEONES: Continuous Annotation and Multimodal Processing of EmOtions in Naturalistic EnvironmentS",
            "BIDSVersion": "1.10.0",
            "DatasetType": "raw",
            "License": "CC BY-4.0",  # Added recommended license
            "EthicsApprovals": [
                "Comité de Ética para la Investigación Científica y Tecnológica de la Universidad Abierta Interamericana (CEICyT-UAI), Dictamen N° 0-1111, Aprobado: 08-04-2024"
            ],
        }
        dataset_json.write_text(json.dumps(metadata, indent=4, ensure_ascii=True))
        print(f"Created dataset_description.json at {dataset_json}")

    # --------------------------------------------------
    # participants.tsv
    # --------------------------------------------------
    participants_tsv = root_folder / "participants.tsv"
    if not participants_tsv.exists():
        cols = [
            "participant_id",
            "species",
            "birth_date",
            "age",
            "sex",  # Added recommended field
            "gender",
            "handedness",
            "education_level",
            "university_career",
            "nationality",
            "ethnicity",
            "residence_location",
            "vr_experience_count",
            "medical_conditions_current",
            "medical_conditions_past",
            "psychological_diagnosis",
            "current_treatment",
        ]
        pd.DataFrame(columns=cols).to_csv(participants_tsv, sep="\t", index=False)
        print(f"Created participants.tsv at {participants_tsv}")

    # --------------------------------------------------
    # participants.json
    # --------------------------------------------------
    participants_json = root_folder / "participants.json"
    if not participants_json.exists():
        participants_metadata = {
            "participant_id": {"Description": "Unique participant identifier"},
            "species": {
                "Description": "Binomial species name from NCBI Taxonomy; if omitted defaults to Homo sapiens"
            },
            "birth_date": {
                "Description": "Date of birth of the participant",
                "Units": "date",  # Changed from Format to Units
            },
            "age": {"Description": "Age of the participant", "Units": "years"},
            "sex": {
                "Description": "Biological sex of the participant",
                "Levels": {"M": "Male", "F": "Female", "n/a": "Not available"},
            },
            "gender": {
                "Description": "Self-identified gender",
                "Levels": {
                    "M": "Male",
                    "F": "Female",
                    "X": "Non-binary",
                    "prefer_not_to_say": "Prefer not to say",
                    "other": "Other (specify)",
                },
            },
            "education_level": {
                "Description": "Highest completed education level",
                "Levels": {
                    "primary_incomplete": "Primary incomplete",
                    "primary_complete": "Primary complete",
                    "secondary_incomplete": "Secondary incomplete",
                    "secondary_complete": "Secondary complete",
                    "tertiary_incomplete": "Tertiary/University incomplete",
                    "tertiary_complete": "Tertiary/University complete",
                    "postgraduate_incomplete": "Postgraduate/PhD incomplete",
                    "postgraduate_complete": "Postgraduate/PhD complete",
                },
            },
            "university_career": {
                "Description": "University degree or career being pursued (if applicable)",
                "Levels": {
                    "psychology": "Psychology",
                    "physics": "Physics",
                    "other": "Other (specify)",
                    "n/a": "Not applicable",
                },
            },
            "nationality": {
                "Description": "Country of nationality (ISO-3166-1 alpha-2 code recommended)",
                "Levels": {"AR": "Argentine", "other": "Other (specify)"},
            },
            "ethnicity": {
                "Description": "Self-reported ethnic origin (multiple selections allowed)",
                "Levels": {
                    "white": "Caucasian or White",
                    "black": "African descent or Black",
                    "latino": "Hispanic or Latino",
                    "asian": "Asian",
                    "native": "Native American or Indigenous",
                    "pacific_islander": "Pacific Islander",
                    "middle_eastern": "Arab or Middle Eastern",
                    "other": "Other (specify)",
                },
                "Delimiter": ";",
            },
            "residence_location": {
                "Description": "Participant's usual place of residence",
                "Levels": {
                    "caba": "Autonomous City of Buenos Aires",
                    "pba": "Province of Buenos Aires",
                    "other": "Other (specify)",
                },
            },
            "vr_experience_count": {
                "Description": "Number of prior experiences using virtual reality devices",
                "Units": "count",
            },
            "handedness": {
                "Description": "Handedness of the participant",
                "Levels": {
                    "left": "Left",
                    "right": "Right",
                    "ambidextrous": "Ambidextrous",
                    "n/a": "Not available",
                },
            },
            "medical_conditions_current": {
                "Description": "Current medical diagnoses or conditions (check all that apply)",
                "Levels": {
                    "pregnancy": "Pregnancy (including suspicion or possibility)",
                    "cardiac_pressure": "Cardiac / Blood pressure",
                    "neurological": "Neurological (e.g., epilepsy)",
                    "other_specify": "Other (please specify)",
                    "none_of_the_above": "None of the above",
                },
                "Delimiter": ";",
            },
            "medical_conditions_past": {
                "Description": "Past medical diagnoses or conditions (check all that apply)",
                "Levels": {
                    "cardiac_pressure": "Cardiac / Blood pressure",
                    "neurological": "Neurological (e.g., epilepsy)",
                    "other_specify": "Other (please specify)",
                    "none_of_the_above": "None of the above",
                },
                "Delimiter": ";",
            },
            "psychological_diagnosis": {
                "Description": "Current psychological diagnoses (as diagnosed by a professional)",
                "Levels": {
                    "anxiety_disorders": "Anxiety or other anxiety disorders (e.g., specific phobias, panic disorder, etc.)",
                    "mood_disorders": "Depression or other mood disorders",
                    "bipolar_disorder": "Bipolar disorder",
                    "schizophrenia": "Schizophrenia",
                    "personality_disorders": "Personality disorders",
                    "other_specify": "Other (please specify)",
                    "none_of_the_above": "None of the above",
                },
                "Delimiter": ";",
            },
            "current_treatment": {
                "Description": "Current ongoing treatment(s) (check all that apply): psychological, psychiatric, neurological, other medical, or none. Specify if 'other' or to add details.",
                "Levels": {
                    "none": "No ongoing treatment",
                    "psychological": "Psychological treatment",
                    "psychiatric": "Psychiatric treatment",
                    "neurological": "Neurological treatment",
                    "other_medical": "Other medical treatment (not psychiatric or neurological)",
                    "other_specify": "Other (please specify)",
                },
                "Delimiter": ";",
            },
        }
        participants_json.write_text(
            json.dumps(participants_metadata, indent=4, ensure_ascii=True)
        )
        print(f"Created participants.json at {participants_json}")


def update_participant_info(
    root_folder: str | Path,
    subject_id: str,
    *,
    birth_date: str | None = None,
    age: int | float | None = None,
    gender: str | None = None,
    handedness: str | None = None,
    education_level: str | None = None,
    university_career: str | None = None,
    nationality: str | None = None,
    ethnicity: str | None = None,
    residence_location: str | None = None,
    vr_experience_count: int | None = None,
    medical_conditions_current: str | None = None,
    medical_conditions_past: str | None = None,
    psychological_diagnosis: str | None = None,
    current_treatment: str | None = None,
):
    """
    Adds or updates a participant's row in participants.tsv
    following BIDS 1.10 standards.

    Missing values should be set as 'n/a'.
    """
    root_folder = Path(root_folder)
    participants_tsv = root_folder / "participants.tsv"
    # --- Columns to manage ---
    col_order = [
        "participant_id",
        "species",
        "birth_date",
        "age",
        "sex",
        "gender",
        "handedness",
        "education_level",
        "university_career",
        "nationality",
        "ethnicity",
        "residence_location",
        "vr_experience_count",
        "medical_conditions_current",
        "medical_conditions_past",
        "psychological_diagnosis",
        "current_treatment",
    ]

    # Load or create dataframe
    if participants_tsv.exists():
        df = pd.read_csv(participants_tsv, sep="\t")
    else:
        df = pd.DataFrame(columns=col_order)
    # Ensure all columns exist
    for c in col_order:
        if c not in df.columns:
            df[c] = "n/a"

    pid = f"sub-{subject_id}"
    if pid in df.participant_id.values:
        row_mask = df.participant_id == pid
    else:
        row_idx = len(df)
        df.loc[row_idx, "participant_id"] = pid
        df.loc[row_idx, "species"] = "Homo sapiens"  # Updated to proper binomial name
        df.loc[row_idx, "sex"] = "n/a"  # Added default value for recommended field

    # Safe assignments
    def set_if(val, col):
        if val is not None:
            if pid in df.participant_id.values:
                df.loc[row_mask, col] = val
            else:
                df.loc[row_idx, col] = val

    set_if(birth_date, "birth_date")
    set_if(age, "age")
    set_if(gender, "gender")
    set_if(handedness, "handedness")
    set_if(education_level, "education_level")
    set_if(university_career, "university_career")
    set_if(nationality, "nationality")
    set_if(ethnicity, "ethnicity")
    set_if(residence_location, "residence_location")
    set_if(vr_experience_count, "vr_experience_count")
    set_if(medical_conditions_current, "medical_conditions_current")
    set_if(medical_conditions_past, "medical_conditions_past")
    set_if(psychological_diagnosis, "psychological_diagnosis")
    set_if(current_treatment, "current_treatment")

    # Reorder columns and save
    df = df[col_order]
    df.to_csv(participants_tsv, sep="\t", index=False)
    print(f"Updated participant information in {participants_tsv}")


def load_participant_data(csv_path: str | Path) -> dict[str, dict]:
    """
    Loads participant data from a questionnaire CSV file and structures it for BIDS.

    This function extracts only the relevant fields from the questionnaire data
    that should be included in the BIDS participants.tsv file. It handles data
    transformations like date formatting and translating Spanish responses to English.

    Parameters
    ----------
    csv_path : str | Path
        Path to the CSV file containing questionnaire data

    Returns
    -------
    dict[str, dict]
        Dictionary with participant IDs as keys and their BIDS metadata as values
    """
    from pathlib import Path

    # Ensure path is a Path object
    csv_path = Path(csv_path)

    # Read CSV file
    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)

    # Define mapping from CSV columns to BIDS fields (for documentation purposes)
    _column_mapping = {
        "ID PARTICIPANTE": "subject_id",
        "Fecha de Nacimiento:": "birth_date",
        "Edad:": "age",
        "Género": "gender",
        "¿Cuál es tu mano dominante?": "handedness",
        "Nivel Educativo alcanzado:": "education_level",
        "Carrera:": "university_career",
        "Nacionalidad": "nationality",
        "Orígen étnico (es posible marcar mas de una opción)": "ethnicity",
        "Lugar de Residencia:": "residence_location",
        "Cantidad de experiencias previas con dispositivos de Realidad Virtual": "vr_experience_count",
        "¿Tenés algún/os de los siguientes diagnósticos o condiciones médicas ACTUALMENTE?": "medical_conditions_current",
        "¿Tuviste algún/os diagnósticos o condiciones médicas de las siguientes PREVIAMENTE?": "medical_conditions_past",
        "Tenés algún diagnóstico (por un profesional) en la actualidad de": "psychological_diagnosis",
        "¿Estás actualmente en tratamiento psicológico/psiquiátrico/neurológico?": "current_treatment",
    }

    # Define value mappings for categorical fields
    gender_mapping = {
        "Masculino": "M",
        "Femenino": "F",
        "No binario": "X",
        "Prefiero no decir": "prefer_not_to_say",
        "Otro (especifique)": "other",
    }

    handedness_mapping = {
        "Diestro/a": "right",
        "Zurdo/a": "left",
        "Ambidiestro/a": "ambidextrous",
    }

    education_mapping = {
        "Primario incompleto": "primary_incomplete",
        "Primario completo": "primary_complete",
        "Secundario incompleto": "secondary_incomplete",
        "Secundario completo": "secondary_complete",
        "Terciario/Universitario incompleto": "tertiary_incomplete",
        "Terciario/Universitario completo": "tertiary_complete",
        "Posgrado/Doctorado incompleto": "postgraduate_incomplete",
        "Posgrado/Doctorado completo": "postgraduate_complete",
    }

    nationality_mapping = {
        "Argentino/a": "AR",
        "Uruguaya": "UY",  # Added based on the example data
    }

    residence_mapping = {
        "Ciudad Autónoma de Buenos Aires": "caba",
        "Provincia de Buenos Aires": "pba",
    }

    # Process each participant
    participants_data = {}

    for _, row in df.iterrows():
        # Skip rows without participant ID
        if pd.isna(row["ID PARTICIPANTE"]):
            continue

        # Format subject ID properly
        subject_id = str(int(row["ID PARTICIPANTE"]))

        # Initialize data dictionary
        data = {}

        # Process birth date (convert to ISO format)
        try:
            birth_date = pd.to_datetime(row["Fecha de Nacimiento:"])
            data["birth_date"] = birth_date.strftime("%Y-%m-%d")
        except Exception:
            data["birth_date"] = None

        # Process age
        if "Edad:" in row and pd.notna(row["Edad:"]):
            data["age"] = int(row["Edad:"])

        # Process gender
        if "Género" in row and pd.notna(row["Género"]):
            data["gender"] = gender_mapping.get(row["Género"], "n/a")

        # Process handedness
        if "¿Cuál es tu mano dominante?" in row and pd.notna(
            row["¿Cuál es tu mano dominante?"]
        ):
            data["handedness"] = handedness_mapping.get(
                row["¿Cuál es tu mano dominante?"], "n/a"
            )

        # Process education level
        if "Nivel Educativo alcanzado:" in row:
            for col in df.columns:
                if col.startswith("Nivel Educativo alcanzado:") and row[col] == 1:
                    level = col.replace("Nivel Educativo alcanzado:", "").strip()
                    data["education_level"] = education_mapping.get(level, "n/a")
                    break

            # Alternative approach if the above doesn't work
            for level, code in education_mapping.items():
                if level in row and pd.notna(row[level]) and row[level] == 1:
                    data["education_level"] = code
                    break

        # Process university career
        if "Carrera:" in row and pd.notna(row["Carrera:"]):
            data["university_career"] = row["Carrera:"]

        # Process nationality
        if "Nacionalidad" in row:
            for col in df.columns:
                if col.startswith("Nacionalidad") and row[col] == 1:
                    nat = col.replace("Nacionalidad", "").strip()
                    data["nationality"] = nationality_mapping.get(nat, nat)
                    break

            # Alternative approach
            if pd.notna(row["Nacionalidad"]):
                data["nationality"] = nationality_mapping.get(
                    row["Nacionalidad"], row["Nacionalidad"]
                )

        # Process ethnicity (can be multiple selections)
        ethnicity_values = []
        ethnicity_cols = [c for c in df.columns if c.startswith("Orígen étnico")]
        for col in ethnicity_cols:
            if pd.notna(row[col]) and row[col] == 1:
                ethnicity = col.replace(
                    "Orígen étnico (es posible marcar mas de una opción)", ""
                ).strip()
                if ethnicity == "Caucásico o Blanco":
                    ethnicity_values.append("white")
                elif ethnicity == "Afrodescendiente o Negro":
                    ethnicity_values.append("black")
                elif ethnicity == "Hispano o Latino":
                    ethnicity_values.append("latino")
                elif ethnicity == "Asiático":
                    ethnicity_values.append("asian")
                elif ethnicity == "Nativo americano o indígena":
                    ethnicity_values.append("native")
                elif ethnicity == "De las islas del Pacífico":
                    ethnicity_values.append("pacific_islander")
                elif ethnicity == "Árabe o del Medio Oriente":
                    ethnicity_values.append("middle_eastern")
                elif ethnicity == "Otro (especifique)":
                    ethnicity_values.append("other")

        if ethnicity_values:
            data["ethnicity"] = ";".join(ethnicity_values)

        # Process residence location
        if "Lugar de Residencia:" in row:
            for col in df.columns:
                if col.startswith("Lugar de Residencia:") and row[col] == 1:
                    loc = col.replace("Lugar de Residencia:", "").strip()
                    data["residence_location"] = residence_mapping.get(loc, loc)
                    break

            # Alternative approach
            if pd.notna(row["Lugar de Residencia:"]):
                data["residence_location"] = residence_mapping.get(
                    row["Lugar de Residencia:"], row["Lugar de Residencia:"]
                )

        # Process VR experience count
        if (
            "Cantidad de experiencias previas con dispositivos de Realidad Virtual"
            in row
            and pd.notna(
                row[
                    "Cantidad de experiencias previas con dispositivos de Realidad Virtual"
                ]
            )
        ):
            data["vr_experience_count"] = int(
                row[
                    "Cantidad de experiencias previas con dispositivos de Realidad Virtual"
                ]
            )

        # Process medical conditions (current)
        medical_current = []
        med_current_cols = [
            c
            for c in df.columns
            if c.startswith(
                "¿Tenés algún/os de los siguientes diagnósticos o condiciones médicas ACTUALMENTE?"
            )
        ]
        for col in med_current_cols:
            if pd.notna(row[col]) and row[col] == 1:
                condition = col.replace(
                    "¿Tenés algún/os de los siguientes diagnósticos o condiciones médicas ACTUALMENTE?",
                    "",
                ).strip()
                if condition != "Ninguna de las anteriores":
                    if condition == "Embarazo (incluye sospecha o posibilidad)":
                        medical_current.append("pregnancy")
                    elif condition == "Cardíacos / Presión":
                        medical_current.append("cardiac_pressure")
                    elif condition == "Neurológicos (por ejemplo, epilepsia)":
                        medical_current.append("neurological")
                    elif condition == "Otro (especifique)":
                        medical_current.append("other_specify")

        if medical_current:
            data["medical_conditions_current"] = ";".join(medical_current)

        # Process medical conditions (past)
        medical_past = []
        med_past_cols = [
            c
            for c in df.columns
            if c.startswith(
                "¿Tuviste algún/os diagnósticos o condiciones médicas de las siguientes PREVIAMENTE?"
            )
        ]
        for col in med_past_cols:
            if pd.notna(row[col]) and row[col] == 1:
                condition = col.replace(
                    "¿Tuviste algún/os diagnósticos o condiciones médicas de las siguientes PREVIAMENTE?",
                    "",
                ).strip()
                if condition != "Ninguna de las anteriores":
                    if condition == "Cardíacos / Presión":
                        medical_past.append("cardiac_pressure")
                    elif condition == "Neurológicos (por ejemplo, epilepsia)":
                        medical_past.append("neurological")
                    elif condition == "Otro (especifique)":
                        medical_past.append("other_specify")

        if medical_past:
            data["medical_conditions_past"] = ";".join(medical_past)

        # Process psychological diagnosis
        psych_diagnosis = []
        psych_cols = [
            c
            for c in df.columns
            if c.startswith(
                "Tenés algún diagnóstico (por un profesional) en la actualidad de"
            )
        ]
        for col in psych_cols:
            if pd.notna(row[col]) and row[col] == 1:
                diagnosis = col.replace(
                    "Tenés algún diagnóstico (por un profesional) en la actualidad de",
                    "",
                ).strip()
                if diagnosis != "Ninguna de las anteriores":
                    if (
                        diagnosis
                        == "Ansiedad u otros trastornos de ansiedad (fobias específicas, trastorno de pánico, etc)"
                    ):
                        psych_diagnosis.append("anxiety_disorders")
                    elif (
                        diagnosis == "Depresión u otros trastornos del estado de ánimo"
                    ):
                        psych_diagnosis.append("mood_disorders")
                    elif diagnosis == "Bipolaridad":
                        psych_diagnosis.append("bipolar_disorder")
                    elif diagnosis == "Esquizofrenia":
                        psych_diagnosis.append("schizophrenia")
                    elif diagnosis == "Trastornos de personalidad":
                        psych_diagnosis.append("personality_disorders")
                    elif (
                        diagnosis
                        == "Especificar (en caso de haber marcado una de las opciones anteriores) o agregar otra opción"
                    ):
                        psych_diagnosis.append("other_specify")

        if psych_diagnosis:
            data["psychological_diagnosis"] = ";".join(psych_diagnosis)

        # Process current treatment
        treatments = []
        treatment_cols = [
            c
            for c in df.columns
            if c.startswith(
                "¿Estás actualmente en tratamiento psicológico/psiquiátrico/neurológico?"
            )
        ]
        for col in treatment_cols:
            if pd.notna(row[col]) and row[col] == 1:
                treatment = col.replace(
                    "¿Estás actualmente en tratamiento psicológico/psiquiátrico/neurológico?",
                    "",
                ).strip()
                if treatment != "No":
                    if treatment == "Si, psicológico":
                        treatments.append("psychological")
                    elif treatment == "Sí, psiquiátrico":
                        treatments.append("psychiatric")
                    elif treatment == "Si, neurológico":
                        treatments.append("neurological")
                    elif (
                        treatment
                        == "Sí, médico (diferente de psiquiatrico y neurológico)"
                    ):
                        treatments.append("other_medical")
                    elif (
                        treatment
                        == "Especificar (en caso de haber marcado una de las opciones anteriores) o agregar otra opción"
                    ):
                        treatments.append("other_specify")

        if treatments:
            data["current_treatment"] = ";".join(treatments)
        elif any(row[col] == 1 for col in treatment_cols if "No" in col):
            data["current_treatment"] = "none"

        # Store participant data
        participants_data[subject_id] = data

    return participants_data


def batch_update_participants(bids_root: str | Path, csv_path: str | Path):
    """
    Updates multiple participants' information from a CSV file.

    Parameters
    ----------
    bids_root : str | Path
        Root folder of the BIDS dataset
    csv_path : str | Path
        Path to the CSV file containing participant data
    """
    # Load participant data from CSV
    participants_data = load_participant_data(csv_path)

    # Update each participant's information
    for subject_id, data in participants_data.items():
        print(f"Updating participant: sub-{subject_id}")
        update_participant_info(bids_root, subject_id, **data)

    print(f"Updated information for {len(participants_data)} participants")


def update_participant_tsv(root_folder, metadata):
    """
    Updates the participants.tsv file with new data.

    Parameters
    ----------
    root_folder : str | Path
        Root folder of the BIDS dataset.
    metadata : dict
        Dictionary containing the participant metadata.
        Keys should be BIDS-compliant field names.
        Missing values should be set as 'n/a'.
    """
    import pandas as pd
    from pathlib import Path

    root_folder = Path(root_folder)

    # Path to participants.tsv file
    participants_tsv = root_folder / "participants.tsv"

    if not participants_tsv.exists():
        # Create participants.tsv with default columns if it doesn't exist
        participants_df = pd.DataFrame(
            columns=["participant_id", "sex", "age", "handedness"]
        )
        participants_df.to_csv(participants_tsv, sep="\t", index=False)

    # Read existing participants.tsv
    participants_df = pd.read_csv(participants_tsv, sep="\t")

    # Check if participant already exists
    subject_id = metadata.get("participant_id", "")
    if not subject_id:
        raise ValueError("participant_id is required in metadata")

    # If participant exists, update their information
    if subject_id in participants_df["participant_id"].values:
        for key, value in metadata.items():
            if key in participants_df.columns:
                participants_df.loc[
                    participants_df["participant_id"] == subject_id, key
                ] = value
    else:
        # Add new participant
        new_row = {col: "n/a" for col in participants_df.columns}
        new_row.update(metadata)
        participants_df = pd.concat(
            [participants_df, pd.DataFrame([new_row])], ignore_index=True
        )

    # Save updated participants.tsv
    participants_df.to_csv(participants_tsv, sep="\t", index=False)

    return participants_df


def read_participants_csv(csv_path):
    """
    Read participant data from a custom CSV format and convert to BIDS format.

    Parameters
    ----------
    csv_path : str | Path
        Path to the CSV file containing participant data.

    Returns
    -------
    dict
        Dictionary with participant IDs as keys and their BIDS metadata as values
    """
    import pandas as pd
    from pathlib import Path

    # Ensure path is a Path object
    csv_path = Path(csv_path)

    # Read CSV file
    df = pd.read_csv(csv_path, encoding="utf-8")

    # Initialize dictionary to store participant data
    participants_data = {}

    # Process each row (participant)
    for _, row in df.iterrows():
        # Extract participant ID
        participant_id = f"sub-{row['ID PARTICIPANTE']}"
        data = {
            "participant_id": participant_id,
        }

        # Extract demographic data if available
        try:
            birth_date = pd.to_datetime(row["Fecha de Nacimiento:"])
            data["birth_date"] = birth_date.strftime("%Y-%m-%d")
        except Exception:
            data["birth_date"] = None

        # Extract gender
        try:
            gender = row["Sexo:"].lower()
            if gender in ["hombre", "masculino", "male", "m"]:
                data["sex"] = "male"
            elif gender in ["mujer", "femenino", "female", "f"]:
                data["sex"] = "female"
            else:
                data["sex"] = gender
        except Exception:
            data["sex"] = None

        # Extract age
        try:
            age = int(row["Edad:"])
            data["age"] = age
        except Exception:
            data["age"] = None

        # Extract handedness
        try:
            handedness = row["Mano dominante:"].lower()
            if handedness in ["derecha", "right", "r", "diestro"]:
                data["handedness"] = "right"
            elif handedness in ["izquierda", "left", "l", "zurdo"]:
                data["handedness"] = "left"
            elif handedness in ["ambidiestro", "ambos", "both", "ambidextrous"]:
                data["handedness"] = "ambidextrous"
            else:
                data["handedness"] = handedness
        except Exception:
            data["handedness"] = None

        # Extract education
        try:
            data["education_level"] = row["Nivel de educación:"]
        except Exception:
            data["education_level"] = None

        # Store participant data
        participants_data[participant_id] = data

    return participants_data
