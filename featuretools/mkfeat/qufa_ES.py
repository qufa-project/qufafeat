from featuretools.entityset import EntitySet
import autonormalize as an
import pandas as pd

from columnspec import ColumnSpec
from error import Error


# CSV데이터가 header를 포함하는지 여부. 데이터 연동 서비스측에 따라 결정됨. 현재 구현은 2가지 경우를 모두 감안하기로 함
csv_has_header = True


class QufaES(EntitySet):
    def __init__(self):
        super().__init__()
        self.target_entity_name = None

    def load_from_csv(self, path, colspec: ColumnSpec) -> Error:
        colnames = colspec.get_colnames()
        if len(colnames) != self._guess_n_columns(path):
            return Error.ERR_COLUMN_COUNT_MISMATCH
        skiprows = 1 if csv_has_header else None
        data = pd.read_csv(path, header=None, names=colnames, skiprows=skiprows)
        es = an.auto_entityset(data, index=colspec.get_key_colname(), accuracy=1, name="main")
        self.entity_dict = es.entity_dict
        self.relationships = es.relationships
        self.reset_data_description()
        self.target_entity_name = colnames[0]

        return Error.OK

    @staticmethod
    def _guess_n_columns(path):
        data = pd.read_csv(path, header=0, nrows=1)
        return len(data.columns)


