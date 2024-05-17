from time import time


class Timer:
    def __init__(self, trigger_time=None):
        self.trigger_time = trigger_time
        self._time = time()

    @property
    def triggered(self) -> bool:
        """
        A property method that checks if the timer has been triggered based on the trigger_time value.
        Returns a true indicating whether the timer has been triggered.
        """
        if self.trigger_time is None:
            return False
        return self.elapsed_time >= self.trigger_time

    def reset(self):
        self._time = time()

    @property
    def elapsed_time(self):
        return time() - self._time
