import pandas as pd
import csv

from .columnspec import ColumnSpec
from .error import Error


# CSV데이터가 header를 포함하는지 여부. 데이터 연동 서비스측에 따라 결정됨. 현재 구현은 2가지 경우를 모두 감안하기로 함
csv_has_header = True


class QufaCsv:
    def __init__(self, path: str, colspec: ColumnSpec):
        self._path = path
        self._colspec = colspec
        self._skiprows = 1 if csv_has_header else None

    def load(self, callback, label_only: bool = False, exclude_label: bool = False, numeric_only: bool = False):
        usecols = None
        colnames = self._colspec.get_colnames()
        if len(colnames) != self._guess_n_columns():
            return Error.ERR_COLUMN_COUNT_MISMATCH
        usecols = self._colspec.get_usecols(label_only=label_only, exclude_label=exclude_label,
                                            numeric_only=numeric_only)

        with open(self._path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            data = list(reader)
            row_count = len(data)

        try:
            chunk_size = row_count // 10
            prog = 0
            callback(0, prog, 0, True)
            for data in pd.read_csv(self._path, header=None, names=colnames, converters=self._colspec.get_converters(),
                                skiprows=self._skiprows, usecols=usecols, dtype=self._colspec.get_dtypes(),
                                true_values=['Y', 'true', 'T'], false_values=['N', 'false', 'F'],
                                chunksize=chunk_size):
                prog += 1
                callback(0, prog, 0, True)

        except ValueError:
            return Error.ERR_COLUMN_TYPE

        return data

    def _guess_n_columns(self):
        data = pd.read_csv(self._path, header=0, skiprows=self._skiprows, nrows=1)
        return len(data.columns)
