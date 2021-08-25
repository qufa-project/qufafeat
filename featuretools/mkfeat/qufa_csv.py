import pandas as pd

from .columnspec import ColumnSpec
from .extract_phase import ExtractPhase
from .error import Error


# CSV데이터가 header를 포함하는지 여부. 데이터 연동 서비스측에 따라 결정됨. 현재 구현은 2가지 경우를 모두 감안하기로 함
csv_has_header = True


class QufaCsv:
    def __init__(self, path: str, colspec: ColumnSpec):
        self._path = path
        self._colspec = colspec
        self._skiprows = 1 if csv_has_header else None

    def get_n_rows(self) -> int:
        with open(self._path, "r") as f:
            n_rows = 0
            while f.readline():
                n_rows += 1
            return n_rows

    def load(self, callback, label_only: bool = False, exclude_label: bool = False, numeric_only: bool = False):
        usecols = None
        colnames = self._colspec.get_colnames()
        if len(colnames) != self._guess_n_columns():
            return Error.ERR_COLUMN_COUNT_MISMATCH
        usecols = self._colspec.get_usecols(label_only=label_only, exclude_label=exclude_label,
                                            numeric_only=numeric_only)

        n_total_rows = self.get_n_rows()

        try:
            chunk_size = 10000
            n_rows = 0
            chunks = []
            for chunk in pd.read_csv(self._path, header=None, names=colnames, converters=self._colspec.get_converters(),
                                    skiprows=self._skiprows, usecols=usecols, dtype=self._colspec.get_dtypes(),
                                    true_values=['Y', 'true', 'T'], false_values=['N', 'false', 'F'],
                                    chunksize=chunk_size):
                chunks.append(chunk)
                n_rows += chunk_size
                prog = n_rows / n_total_rows * 100
                callback(prog, ExtractPhase.READ_CSV)

            return pd.concat(chunks)
        except ValueError:
            return Error.ERR_COLUMN_TYPE

    def _guess_n_columns(self):
        data = pd.read_csv(self._path, header=0, skiprows=self._skiprows, nrows=1)
        return len(data.columns)
