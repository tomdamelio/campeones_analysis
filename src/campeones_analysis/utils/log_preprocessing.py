import json
import os

import numpy as np


class LogPreprocessingDetails:
    def __init__(self, json_path, subject, session, task):
        self.json_path = json_path
        self.subject = subject
        self.session = session
        self.task = task
        self.logs = self.load_preprocessing_details()

    def load_preprocessing_details(self):
        if os.path.exists(self.json_path):
            with open(self.json_path) as f:
                return json.load(f)
        else:
            return {}

    def save_preprocessing_details(self):
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.int64):
                return int(obj)
            if isinstance(obj, np.float64):
                return float(obj)
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj

        serializable_logs = convert_to_serializable(self.logs)

        with open(self.json_path, "w") as f:
            json.dump(serializable_logs, f, indent=4)

    def initialize_log_structure(self):
        if self.subject not in self.logs:
            self.logs[self.subject] = {}
        if self.session not in self.logs[self.subject]:
            self.logs[self.subject][self.session] = {}
        if self.task not in self.logs[self.subject][self.session]:
            self.logs[self.subject][self.session][self.task] = {}

    def log_detail(self, key, value):
        self.initialize_log_structure()
        if isinstance(value, np.ndarray):
            value = value.tolist()  # Convert numpy arrays to lists
        self.logs[self.subject][self.session][self.task][key] = value

    def get_log(self):
        self.initialize_log_structure()
        return self.logs[self.subject][self.session][self.task]

    def import_bad_channels_another_task(self):
        self.initialize_log_structure()
        for other_task, details in self.logs[self.subject][self.session].items():
            if other_task != self.task and "interpolated_channels" in details:
                return details["interpolated_channels"]
            else:
                return []
