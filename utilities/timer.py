import time

class Timer:
    def __init__(self):
        self.time = time.time()
    def reset(self):
        self.time = time.time()
    @property
    def elapsed_time(self):
        return time.time() - self.time