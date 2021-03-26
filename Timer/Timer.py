import time
import time


class Timer:
    def __init__(self):
        """"""
        self.start_time = None

    def start(self):
        """Method to start a new timer"""
        if self.start_time is not None:
            raise Exception("The timer is currently running")

        self.start_time = time.perf_counter()

    def stop(self):
        """Method to stop the timer"""
        if self.start_time is None:
            raise Exception("The timer is not running")

        elapsed_time = time.perf_counter() - self.start_time
        print(f"Elapsed time : {elapsed_time} seconds")
