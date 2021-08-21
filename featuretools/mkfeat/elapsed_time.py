import time


class ElapsedTime:
    def __init__(self):
        self._elapsed_secs: list[int] = []
        self._sec_prev: int = 0

    def mark(self):
        cur_sec = time.time()
        if self._sec_prev > 0:
            self._elapsed_secs.append(cur_sec - self._sec_prev)
        self._sec_prev = cur_sec

    def get_elapsed_secs(self):
        return self._elapsed_secs
