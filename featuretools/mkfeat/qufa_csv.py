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

    def load(self):
        colnames = self._colspec.get_colnames()
        if len(colnames) != self._guess_n_columns():
            return Error.ERR_COLUMN_COUNT_MISMATCH
        data = pd.read_csv(self._path, header=None, names=colnames, converters=self._colspec.get_converters(),
                           skiprows=self._skiprows,
                           true_values=['Y', 'true', 'T'], false_values=['N', 'false', 'F'])
        return data

    def _guess_n_columns(self):
        data = pd.read_csv(self._path, header=0, nrows=self._skiprows)
        return len(data.columns)
