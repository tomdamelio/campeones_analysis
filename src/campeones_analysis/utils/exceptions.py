class BehNotFoundError(Exception):
    def __init__(self, message="Beheavioral data not found"):
        self.message = message
        super().__init__(self.message)


class InconsistentAnnotationsWithBehError(Exception):
    def __init__(self, message="Annotations are inconsistent with behavioral data"):
        self.message = message
        super().__init__(self.message)


class StimChannelNotFoundError(Exception):
    def __init__(self, message="Stim channel not found"):
        self.message = message
        super().__init__(self.message)
