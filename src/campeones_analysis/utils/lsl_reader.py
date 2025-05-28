import glob
import os
import re
import sys

import pyxdf
from mnelab.io.xdf import read_raw_xdf

from campeones_analysis.utils import bids_compliance, preprocessing_helpers
from campeones_analysis.utils.exceptions import (
    BehNotFound,
    InconsistentAnnotationsWithBeh,
)


def read_lsl(results_folder, subject, session, should_correct_markers=True):
    full_path = os.path.join(
        results_folder, f"sub-{subject}", f"ses-{session}", "physio"
    )

    if os.path.exists(full_path):
        xdf_files = glob.glob(os.path.join(full_path, "*.xdf"))
        print([file for file in xdf_files])
    else:
        print(f"The folder {full_path} does not exist.")
        return

    files_with_errors = []

    for file in xdf_files:
        match = re.search(r"task-(.*?)_physio", file)
        if match:
            task = match.group(1)
            print(f"Task name: {task}")
            streams, header = pyxdf.load_xdf(file)
            stream_eeg = next(
                (stream for stream in streams if stream["info"]["type"][0] == "EEG"),
                None,
            )

            if stream_eeg is None:
                files_with_errors.append(
                    {
                        "subject": subject,
                        "session": session,
                        "task": task,
                        "message": "No se encontró el stream de EEG",
                    }
                )
                continue

            stream_eeg_id = stream_eeg["info"]["stream_id"]
            stream_gaze = next(
                (stream for stream in streams if stream["info"]["type"][0] == "gaze"),
                None,
            )

            if stream_gaze is None:
                files_with_errors.append(
                    {
                        "subject": subject,
                        "session": session,
                        "task": task,
                        "message": "No se encontró el stream de gaze",
                    }
                )
                continue

            stream_gaze_id = stream_gaze["info"]["stream_id"]
            stream_markers = next(
                (
                    stream
                    for stream in streams
                    if stream["info"]["type"][0] == "Markers"
                ),
                None,
            )

            if stream_markers is None:
                files_with_errors.append(
                    {
                        "subject": subject,
                        "session": session,
                        "task": task,
                        "message": "No se encontró el stream de markers",
                    }
                )
                continue

            print("Shape Markers:", stream_markers["time_stamps"].shape)
            raw_eeg = read_raw_xdf(file, stream_ids=[stream_eeg_id], preload=True)
            raw_gaze = read_raw_xdf(file, stream_ids=[stream_gaze_id], preload=True)

            try:
                process_streams(
                    raw_eeg,
                    raw_gaze,
                    results_folder,
                    subject,
                    session,
                    task,
                    should_correct_markers=should_correct_markers,
                )
            except BehNotFound as e:
                files_with_errors.append(
                    {
                        "subject": subject,
                        "session": session,
                        "task": task,
                        "message": e.message,
                    }
                )
            except InconsistentAnnotationsWithBeh as e:
                files_with_errors.append(
                    {
                        "subject": subject,
                        "session": session,
                        "task": task,
                        "message": e.message,
                    }
                )
            except Exception as e:
                files_with_errors.append(
                    {
                        "subject": subject,
                        "session": session,
                        "task": task,
                        "message": f"Error desconocido: {e}",
                    }
                )

    if len(files_with_errors) > 0:
        print("Se han encontrado errores en:")
        print("\n")
        for file in files_with_errors:
            print(f"Sujeto: {file['subject']}")
            print(f"Sesión: {file['session']}")
            print(f"Tarea: {file['task']}")
            print(f"Mensaje: {file['message']}")
            print("\n")


def process_streams(
    raw_eeg, raw_gaze, results_folder, subject, session, task, should_correct_markers
):
    # Add necessary info to the raw object to be BIDS compliant
    raw_eeg.info["subject_info"] = {
        "id": subject,
        # TODO: complete infos from Beh metadata
        # 'his_id': '001',
        # 'birthday': (1990, 1, 1),
        # 'sex': 1,  # 0: unknown, 1: male, 2: female
        # 'hand': 1  # 0: unknown, 1: right, 2: left, 3: ambidextrous
    }

    raw_gaze.info["subject_info"] = {
        "id": subject,
        # TODO: complete infos from Beh metadata
        # 'his_id': '001',
        # 'birthday': (1990, 1, 1),
        # 'sex': 1,  # 0: unknown, 1: male, 2: female
        # 'hand': 1  # 0: unknown, 1: right, 2: left, 3: ambidextrous
    }

    raw_eeg.info["line_freq"] = 50
    raw_eeg = preprocessing_helpers.set_chs_montage(raw_eeg)
    raw_gaze = preprocessing_helpers.make_eyetracking_mapping(raw_gaze)
    # TODO: Implement the processing of gaze data
    task_bids = task

    for data_type, raw in zip(["eeg", "gaze"], [raw_eeg, raw_gaze]):
        results_path = os.path.join(
            "..", results_folder, f"sub-{subject}/ses-{session}/{data_type}/"
        )
        os.makedirs(results_path, exist_ok=True)
        bids_compliance.save_raw_bids_compliant(
            subject, session, task_bids, data_type, raw, results_folder
        )


if __name__ == "__main__":
    results_folder = sys.argv[1]
    subject = sys.argv[2]
    session = sys.argv[3]
    should_correct_markers = True if sys.argv[4] is None else sys.argv[4]

    read_lsl(
        results_folder, subject, session, should_correct_markers=should_correct_markers
    )
