# pypore_compat.py
# Minimal compatibility layer for loading pre-segmented JSON data
# Bypasses broken PyPore.cparsers

import json


class Segment(object):
    """Simple segment container matching PyPore.DataTypes.Segment interface."""

    def __init__(self, data):
        self.mean = data.get("mean", 0)
        self.std = data.get("std", 0)
        self.duration = data.get("duration", 0)
        self.start = data.get("start", 0)
        self.end = data.get("end", 0)
        self.min = data.get("min", 0)
        self.max = data.get("max", 0)
        self.name = data.get("name", "Segment")

    def __repr__(self):
        return json.dumps(
            {
                "mean": self.mean,
                "std": self.std,
                "duration": self.duration,
                "start": self.start,
                "end": self.end,
                "name": self.name,
            },
            indent=4,
        )


class Event(object):
    """Simple event container matching PyPore.DataTypes.Event interface."""

    def __init__(self, data):
        self.segments = [Segment(s) for s in data.get("segments", [])]
        self.n = len(self.segments)
        self.mean = data.get("mean", 0)
        self.std = data.get("std", 0)
        self.duration = data.get("duration", 0)
        self.start = data.get("start", 0)
        self.end = data.get("end", 0)
        self.min = data.get("min", 0)
        self.max = data.get("max", 0)
        self.name = data.get("name", "Event")


class File(object):
    """Simple file container matching PyPore.DataTypes.File interface."""

    def __init__(self, data):
        self.events = [Event(e) for e in data.get("events", [])]
        self.n = len(self.events)
        self.filename = data.get("filename", "")
        self.mean = data.get("mean", 0)
        self.std = data.get("std", 0)
        self.name = data.get("name", "File")

    @classmethod
    def from_json(cls, filename):
        """Load file from JSON, matching PyPore interface."""
        with open(filename, "r") as f:
            data = json.load(f)
        return cls(data)

    def close(self):
        """No-op for compatibility."""
        pass


# Re-export HMMBoard and distributions from PyPore.hmm (these work)
# from PyPore.hmm import (
#     HMMBoard,
#     ModularProfileModel,
#     Phi29GlobalAlignmentModule,
#     NormalDistribution,
#     UniformDistribution,
#     GaussianKernelDensity,
# )

# Re-export yahmm classes
# from yahmm import Model, State
