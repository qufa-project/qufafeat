from pandas import DataFrame

from .rowset import RowSet


class RowSetManager:
    def __init__(self, df: DataFrame):
        self._df = df
        self._rowsets = {}

    def get(self, cols: frozenset):
        if cols not in self._rowsets:
            self._rowsets[cols] = RowSet(self._df[list(cols)])
        return self._rowsets[cols]
