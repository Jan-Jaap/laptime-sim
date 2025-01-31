import datetime


class Timer:
    def __init__(self, trigger_time=None):
        """
        Initializes a Timer object.

        Args:
            trigger_time (int, optional): Trigger time in seconds. Defaults to None.

        Attributes:
            trigger_time (datetime.timedelta): The time to trigger the timer.
        """
        if trigger_time is not None:
            self.trigger_time = datetime.timedelta(seconds=trigger_time)
        self._time = datetime.datetime.now()

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
        """
        Resets the timer to the current time.

        This method sets the timer back to 0 by setting the internal _time attribute
        to the current time.
        """
        self._time = datetime.datetime.now()

    @property
    def elapsed_time(self):
        """
        A property method that returns the elapsed time in seconds since the timer was last reset.

        Returns a datetime.timedelta object which is the difference between the current time and the time the timer was last reset.
        """

        return datetime.datetime.now() - self._time
