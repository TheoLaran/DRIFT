import time

from utils.utils import SAROS
from DRIFT.DRIFT_Function import DRIFT_Function


class COS(DRIFT_Function):
    def __init__(self, latent_dim, num_users_all, num_items):
        self.time_in_here = 0
        self._table = {}
        self.model = SAROS(latent_dim, u=num_users_all, i=num_items, alpha_reg=0.01)
        self.computation_time = 0

    def get_model(self):
        return self.model


    def add_DO(self, do):
        def _add_DO():
            t = time.time()
            items = do.get_item()
            for item in items:
                if item not in self._table:
                    self._table[item] = []
                self._table[item].append(do)
            self.time_in_here += time.time() - t
        return self.computes_function(_add_DO)


    def get_data_owner_from_item(self, item):
        def _get_data_owner_from_item():
            if item not in self._table:
                raise Exception(f"item {item} not found in the table")
            return self._table[item]
        return self.computes_function(_get_data_owner_from_item)





