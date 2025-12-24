import time


class Timer:
    """Simple timing utility for performance measurement."""

    def __init__(self):
        self._start_time = None
        self._end_time = None

    def start(self):
        """Start the timer."""
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer and return elapsed time in milliseconds."""
        if self._start_time is None:
            raise RuntimeError("Timer not started")
        self._end_time = time.perf_counter()
        return (self._end_time - self._start_time) * 1000

    def reset(self):
        """Reset the timer."""
        self._start_time = None
        self._end_time = None
