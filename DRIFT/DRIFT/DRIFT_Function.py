import time


class DRIFT_Function:
    def __init__(self, *args):
        self.time_in_here = 0

    def computes_function(self, f):
        t = time.time()
        res = f()
        self.time_in_here += time.time() - t
        return res