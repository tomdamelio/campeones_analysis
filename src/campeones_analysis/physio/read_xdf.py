# Standard library imports
import json
import sys
import warnings
from pathlib import Path
from typing import cast

# Add project root to Python path for imports when running as script
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))

# Third-party imports
import mne
import numpy as np
import pandas as pd
import pyxdf
from mne_bids import BIDSPath, write_raw_bids
from mnelab.io.xdf import read_raw_xdf

# Local imports
from src.campeones_analysis.utils.bids_compliance import (
    create_bids_metadata,
    update_participant_info,
)
from src.campeones_analysis.utils.preprocessing_helpers import (
    correct_channel_types,
    make_joystick_mapping,
    set_chs_montage,
)


def read_and_process_xdf(subject, session, task, run, acq="a"):
    """
    Read and process XDF files, converting them to BIDS format.

    Parameters
    ----------
    subject : str
        Subject identifier
    session : str
        Session identifier (e.g., "VR")
    task : str
        Task identifier
    run : str
        Run identifier
    acq : str, optional
        Acquisition identifier, by default "a"

    Returns
    -------
    mne.io.Raw
        Processed raw data object
    """
    # Get project root (two levels up from current file)
    project_root = Path(__file__).resolve().parents[3]
    print(f"Directorio raíz del proyecto: {project_root}")

    # Configuration of paths using pathlib
    data_folder = project_root / "data" / "sourcedata" / "xdf"
    bids_root = project_root / "data" / "raw"

    # Create or update BIDS metadata files
    create_bids_metadata(bids_root)

    # Update participant information (optional)
    update_participant_info(
        bids_root,
        subject,
        birth_date=None,
        age=None,
        gender=None,
        handedness=None,
        education_level=None,
        university_career=None,
        nationality=None,
        ethnicity=None,
        residence_location=None,
        vr_experience_count=None,
        medical_conditions_current=None,
        medical_conditions_past=None,
        psychological_diagnosis=None,
        current_treatment=None,
    )

    # Build path for original file (non-BIDS)
    xdf_path = (
        data_folder
        / f"sub-{subject}"
        / f"ses-{session}"
        / "physio"
        / f"sub-{subject}_ses-{session}_day-{acq}_task-{task}_run-{run}_eeg.xdf"
    )
    print("\nBuscando archivo en:")
    print(f"Path relativo: {xdf_path}")
    print(f"Path absoluto: {xdf_path.absolute()}")

    # Verify if file exists
    if not xdf_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {xdf_path}")

    print(f"\nProcesando archivo: {xdf_path}")
    streams, header = pyxdf.load_xdf(str(xdf_path))

    # Process and save events if they exist
    for stream in streams:
        if stream["info"]["type"][0] == "Markers":
            print("\nProcesando eventos...")
            # Convert marker values to numbers
            marker_values = {
                "test": 0,
                "self_report_post_start": 1,
                "self_report_post_end": 2,
                "instruction_start": 3,
                "instruction_end": 4,
                "video_start": 5,
                "video_end": 6,
                "baseline_start": 7,
                "baseline_end": 8,
                "luminance_start": 9,
                "luminance_end": 10,
                "audio_response_start": 11,
                "audio_response_end": 12,
                "calm_video_start": 13,
                "calm_video_end": 14,
                "SUPRABLOCK_START": 15,
                "SUPRABLOCK_END": 16,
                "BLOCK_START": 17,
                "BLOCK_END": 18,
                "self_report_pre_start": 19,
                "self_report_pre_end": 20,
            }

            events = []
            for timestamp, _, event_type in zip(
                stream["time_stamps"], stream["time_series"], stream["string_markers"]
            ):
                if event_type.strip() in marker_values:
                    events.append(
                        {
                            "onset": timestamp,
                            "duration": 0.0,
                            "trial_type": event_type.strip(),
                            "value": marker_values[event_type.strip()],
                        }
                    )

            # Write events to JSON file
            events_path = str(xdf_path).replace(".xdf", "_events.json")
            with open(events_path, "w") as f:
                json.dump(events, f, indent=4, ensure_ascii=True)
            print(f"Events saved to: {events_path}")

    # Initialize raw as None
    raw = None
    eeg_stream = None

    # First find EEG stream and show all streams
    print("\nStreams encontrados:")
    for i, stream in enumerate(streams):
        print(f"\nStream {i + 1}:")
        print(f"Tipo: {stream['info']['type'][0]}")
        print(f"Nombre: {stream['info']['name'][0]}")
        print(f"Stream ID: {stream['info']['stream_id']}")
        if stream["info"]["type"][0] == "EEG":
            eeg_stream = stream
            print("¡Este es el stream de EEG!")
        if stream["info"]["name"][0] == "T.16000MAxes":
            print("\nInformación de los ejes del joystick (X e Y):")
            print(f"Número total de muestras: {len(stream['time_series'])}")
            srate_joystick = float(stream["info"]["nominal_srate"][0])
            print(f"Frecuencia de muestreo: {srate_joystick} Hz")

            # Detailed joystick timing analysis
            print("\nAnálisis detallado de timing del joystick:")
            print(f"Tiempo de inicio: {stream['time_stamps'][0]:.2f} segundos")
            print(f"Tiempo final: {stream['time_stamps'][-1]:.2f} segundos")
            duration_joystick = stream["time_stamps"][-1] - stream["time_stamps"][0]
            print(f"Duración total: {duration_joystick:.2f} segundos")
            print(
                f"Intervalo entre muestras promedio: {duration_joystick / len(stream['time_stamps']):.6f} segundos"
            )

            # Calculate and show joystick duration in more readable format
            hours_joystick = int(duration_joystick // 3600)
            minutes_joystick = int((duration_joystick % 3600) // 60)
            seconds_joystick = duration_joystick % 60
            print(
                f"Duración total: {hours_joystick:02d}:{minutes_joystick:02d}:{seconds_joystick:05.2f} (HH:MM:SS)"
            )

            # Analyze joystick data
            np.array(stream["time_series"])
            time_stamps = np.array(stream["time_stamps"])
            _ = time_stamps - time_stamps[0]  # Calculate relative time

        if stream["info"]["type"][0] == "EEG":
            print("\nInformación del stream de EEG:")
            print(f"Número total de muestras: {len(stream['time_series'])}")
            srate_eeg = float(stream["info"]["nominal_srate"][0])
            print(f"Frecuencia de muestreo: {srate_eeg} Hz")
            print(f"Número de canales: {stream['info']['channel_count'][0]}")

            # Detailed EEG timing analysis
            print("\nAnálisis detallado de timing del EEG:")
            print(f"Tiempo de inicio: {stream['time_stamps'][0]:.2f} segundos")
            print(f"Tiempo final: {stream['time_stamps'][-1]:.2f} segundos")
            duration = stream["time_stamps"][-1] - stream["time_stamps"][0]
            print(f"Duración total: {duration:.2f} segundos")
            print(
                f"Intervalo entre muestras promedio: {duration / len(stream['time_stamps']):.6f} segundos"
            )

            # Calculate and show duration in more readable format
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = duration % 60
            print(
                f"Duración total: {hours:02d}:{minutes:02d}:{seconds:05.2f} (HH:MM:SS)"
            )

    # If we found EEG, process it
    if eeg_stream is None:
        raise ValueError("No se encontró stream de EEG en el archivo")

    print("\nProcesando stream EEG...")
    raw_result = read_raw_xdf(
        str(xdf_path), stream_ids=[eeg_stream["info"]["stream_id"]], preload=True
    )
    # Cast raw to the correct type to avoid type errors
    raw = cast(mne.io.Raw, raw_result)
    print("Info del raw EEG:")
    print(raw.info)

    # Correct channel types
    raw = correct_channel_types(raw)

    # Resample EEG to 250 Hz if necessary
    target_sfreq = 250
    if raw.info["sfreq"] != target_sfreq:
        print(f"\nResampleando EEG de {raw.info['sfreq']} Hz a {target_sfreq} Hz...")
        raw = raw.resample(target_sfreq, npad="auto")

    # Process joystick stream if it exists
    joystick_raw = None
    for stream in streams:
        if stream["info"]["name"][0] == "T.16000MAxes":
            print("\nProcesando joystick...")
            joystick_data = np.array(stream["time_series"])
            joystick_timestamps = stream["time_stamps"]

            joystick_raw_result = make_joystick_mapping(
                joystick_data, joystick_timestamps, raw.info["sfreq"]
            )
            # Cast joystick_raw to the correct type
            joystick_raw = cast(mne.io.Raw, joystick_raw_result)

    # If joystick data was found, add it to the main Raw object
    if joystick_raw is not None:
        print("\nAgregando canales de joystick al objeto Raw...")
        raw.add_channels([joystick_raw])

    # Apply EEG channel montage
    print("\nAplicando montaje de canales EEG...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        raw = set_chs_montage(raw)

    # Document channel mapping and units
    print("\nDocumentando información de canales...")
    channels_info = pd.DataFrame(
        {
            "name": raw.ch_names,
            "type": [raw.info["chs"][i]["kind"] for i in range(len(raw.ch_names))],
            "units": [
                raw.info["chs"][i].get("unit", "n/a") for i in range(len(raw.ch_names))
            ],
            "sampling_frequency": [raw.info["sfreq"]] * len(raw.ch_names),
            "low_cutoff": [raw.info["highpass"]] * len(raw.ch_names),
            "high_cutoff": [raw.info["lowpass"]] * len(raw.ch_names),
        }
    )

    # Create BIDSPath for EEG
    bids_path = BIDSPath(
        subject=subject,
        session=session.lower(),  # Use lowercase for BIDS
        task=task,
        run=run,
        acquisition=acq,
        root=bids_root,
        datatype="eeg",
        extension=".vhdr",
        check=True,
    )

    # Create necessary directories
    Path(bids_path.directory).mkdir(parents=True, exist_ok=True)

    # Save channels.tsv
    channels_path = bids_path.copy().update(suffix="channels", extension=".tsv")
    channels_info.to_csv(channels_path, sep="\t", index=False)
    print(f"Información de canales guardada en: {channels_path}")

    # Save EEG data in BIDS format
    print("\nGuardando datos EEG en formato BIDS...")
    # Convert events to BrainVision format before saving
    events_array = None
    event_id = None
    if raw.annotations:
        events, event_id = mne.events_from_annotations(raw)
        if len(events) > 0:
            events_array = events

    write_raw_bids(
        raw=raw,
        bids_path=bids_path,
        format="BrainVision",
        allow_preload=True,
        overwrite=True,
        events=events_array,
        event_id=event_id,
    )

    # Process and save events if they exist
    for stream in streams:
        if stream["info"]["type"][0] == "Markers":
            print("\nProcesando eventos...")
            # Convert marker values to numbers
            marker_values = {
                "test": 0,
                "self_report_post_start": 1,
                "self_report_post_end": 2,
                "instruction_start": 3,
                "instruction_end": 4,
                "video_start": 5,
                "video_end": 6,
                "baseline_start": 7,
                "baseline_end": 8,
                "luminance_start": 9,
                "luminance_end": 10,
                "audio_response_start": 11,
                "audio_response_end": 12,
                "calm_video_start": 13,
                "calm_video_end": 14,
                "SUPRABLOCK_START": 15,
                "SUPRABLOCK_END": 16,
                "BLOCK_START": 17,
                "BLOCK_END": 18,
                "self_report_pre_start": 19,
                "self_report_pre_end": 20,
            }

            # Convert marker values to numbers
            event_values = np.array(
                [marker_values.get(val[0], 0) for val in stream["time_series"]]
            )

            # Convert timestamps to samples relative to data start
            event_times = np.array(stream["time_stamps"])

            # Verify if there are events
            if len(event_times) == 0:
                print("Warning: No events found in the stream")
                continue

            event_samples = np.round(
                (event_times - event_times[0]) * raw.info["sfreq"]
            ).astype(int)

            # Filter events that are within data range
            valid_events = (event_samples >= 0) & (event_samples < len(raw.times))
            event_samples = event_samples[valid_events]
            event_values = event_values[valid_events]

            if len(event_samples) == 0:
                print("Warning: No valid events found within the data range")
                continue

            # Create stimulation channel if it doesn't exist
            if "STI 014" not in raw.ch_names:
                stim_data = np.zeros((1, len(raw.times)))
                info = mne.create_info(
                    ch_names=["STI 014"], sfreq=raw.info["sfreq"], ch_types="misc"
                )
                raw.add_channels([mne.io.RawArray(stim_data, info)])
                raw.set_channel_types({"STI 014": "stim"})

            # Create events
            events = np.column_stack(
                (event_samples, np.zeros(len(event_samples)), event_values)
            )

            # Add events to raw object
            raw.add_events(events, stim_channel="STI 014")

            # Save events in BIDS format
            events_df = pd.DataFrame(
                {
                    "onset": events[:, 0],
                    "duration": events[:, 1],
                    "description": events[:, 2],
                }
            )
            events_path = bids_path.copy().update(suffix="events", extension=".tsv")
            events_df.to_csv(events_path, sep="\t", index=False)

            # Save event metadata
            events_metadata = {
                "onset": {"Description": "Event onset", "Units": "seconds"},
                "duration": {"Description": "Event duration", "Units": "seconds"},
                "description": {
                    "Description": "Event description",
                    "event_id": marker_values,
                },
            }
            events_metadata_path = events_path.copy().update(extension=".json")
            with open(events_metadata_path.fpath, "w") as f:
                json.dump(events_metadata, f, indent=4, ensure_ascii=True)

    print("\nProcesamiento BIDS completado exitosamente.")
    return raw


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process XDF files and convert to BIDS format"
    )
    parser.add_argument(
        "--subject", type=str, help="Process specific subject (e.g., '01')"
    )
    parser.add_argument(
        "--session", type=str, help="Process specific session (e.g., 'a')"
    )
    parser.add_argument("--task", type=str, help="Process specific task")
    parser.add_argument("--run", type=str, help="Process specific run")
    parser.add_argument("--acq", type=str, default="a", help="Acquisition parameter")
    args = parser.parse_args()

    # Get project root
    project_root = Path(__file__).resolve().parents[3]
    data_folder = project_root / "data" / "sourcedata" / "xdf"

    # Function to find all available subjects
    def find_subjects():
        subjects = []
        for path in data_folder.glob("sub-*"):
            if path.is_dir():
                subject = path.name.replace("sub-", "")
                subjects.append(subject)
        return sorted(subjects)

    # Function to find sessions for a subject
    def find_sessions(subject):
        sessions = []
        subject_dir = data_folder / f"sub-{subject}"
        for path in subject_dir.glob("ses-*"):
            if path.is_dir():
                session = path.name.replace("ses-", "")
                sessions.append(session)
        return sorted(sessions)

    # Function to find tasks and runs for a subject and session
    def find_tasks_runs(subject, session):
        tasks_runs = []
        session_dir = data_folder / f"sub-{subject}" / f"ses-{session}" / "physio"
        if not session_dir.exists():
            return []

        pattern = f"sub-{subject}_ses-{session}_day-*_task-*_run-*_eeg.xdf"
        for path in session_dir.glob(pattern):
            filename = path.name
            parts = filename.split("_")

            task = None
            run = None
            acq = None

            for part in parts:
                if part.startswith("task-"):
                    task = part.replace("task-", "")
                elif part.startswith("run-"):
                    run = part.replace("run-", "")
                elif part.startswith("day-"):
                    acq = part.replace("day-", "")

            if task and run:
                tasks_runs.append((task, run, acq or "a"))

        return tasks_runs

    # Process all subjects or a specific one
    if args.subject:
        subjects = [args.subject]
    else:
        subjects = find_subjects()

    if not subjects:
        print("No subjects found in the sourcedata directory.")
        sys.exit(1)

    print(f"Found {len(subjects)} subjects: {', '.join(subjects)}")

    # Process each subject
    for subject in subjects:
        print(f"\n{'=' * 80}")
        print(f"Processing subject: {subject}")
        print(f"{'=' * 80}")

        if args.session:
            sessions = [args.session]
        else:
            sessions = find_sessions(subject)

        if not sessions:
            print(f"No sessions found for subject {subject}.")
            continue

        print(f"Found {len(sessions)} sessions: {', '.join(sessions)}")

        # Process each session
        for session in sessions:
            print(f"\n{'-' * 60}")
            print(f"Processing session: {session}")
            print(f"{'-' * 60}")

            if args.task and args.run:
                tasks_runs = [(args.task, args.run, args.acq)]
            else:
                tasks_runs = find_tasks_runs(subject, session)

            if not tasks_runs:
                print(f"No tasks/runs found for subject {subject}, session {session}.")
                continue

            print(f"Found {len(tasks_runs)} task/run combinations.")

            # Process each task/run combination
            for task, run, acq in tasks_runs:
                print(f"\nProcessing task: {task}, run: {run}, acq: {acq}")
                try:
                    raw = read_and_process_xdf(subject, session, task, run, acq)
                    print(
                        f"Successfully processed sub-{subject}_ses-{session}_task-{task}_run-{run}"
                    )
                except Exception as e:
                    print(
                        f"Error processing sub-{subject}_ses-{session}_task-{task}_run-{run}: {e}"
                    )
