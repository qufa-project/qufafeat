import pandas as pd

from .columnspec import ColumnSpec
from .error import Error


# CSV데이터가 header를 포함하는지 여부. 데이터 연동 서비스측에 따라 결정됨. 현재 구현은 2가지 경우를 모두 감안하기로 함
csv_has_header = True


class QufaCsv:
    def __init__(self, path: str, colspec: ColumnSpec):
        self._path = path
        self._colspec = colspec
        self._skiprows = 1 if csv_has_header else None

    def get_row(self, path):
        with open(path, "r") as f:
            lines = f.readlines()
            return len(lines)

    def load(self, callback, label_only: bool = False, exclude_label: bool = False, numeric_only: bool = False):
        usecols = None
        colnames = self._colspec.get_colnames()
        if len(colnames) != self._guess_n_columns():
            return Error.ERR_COLUMN_COUNT_MISMATCH
        usecols = self._colspec.get_usecols(label_only=label_only, exclude_label=exclude_label,
                                            numeric_only=numeric_only)

        row_count = self.get_row(self._path)

        try:
            chunk_size = 10000
            chunk_prog = chunk_size / row_count * 100
            prog = 0
            data_arr = []
            for data in pd.read_csv(self._path, header=None, names=colnames, converters=self._colspec.get_converters(),
                                skiprows=self._skiprows, usecols=usecols, dtype=self._colspec.get_dtypes(),
                                true_values=['Y', 'true', 'T'], false_values=['N', 'false', 'F'],
                                chunksize=chunk_size):
                data_arr.append(data)
                prog += chunk_prog
                callback(0, prog, 0, True)

            data_concat = pd.concat([data for data in data_arr])

        except ValueError:
            return Error.ERR_COLUMN_TYPE

        return data_concat

    def _guess_n_columns(self):
        data = pd.read_csv(self._path, header=0, skiprows=self._skiprows, nrows=1)
        return len(data.columns)
