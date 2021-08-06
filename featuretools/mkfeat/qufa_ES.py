from featuretools.entityset import EntitySet
import autonormalize as an
import pandas as pd

from .columnspec import ColumnSpec
from .error import Error
from .normalize import normalize

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
        data = pd.read_csv(path, header=None, names=colnames, converters=colspec.get_converters(), skiprows=skiprows,
                           true_values=['Y', 'true', 'T'], false_values=['N', 'false', 'F'])
        key_colname = colspec.get_key_colname()
        norminfos = normalize(data, key_colname)

        self.entity_from_dataframe("main", data, index=key_colname)
        for norminfo in norminfos:
            keyname = norminfo[0]
            vars = norminfo[1:]
            etname = self._search_owner_entity(keyname)
            self.normalize_entity(etname, "tbl_{}".format(keyname), norminfo[0], additional_variables=vars)
        self.target_entity_name = "main"

        return Error.OK

    def _search_owner_entity(self, varname):
        for et in self.entities:
            for var in et.variables:
                if var.name == varname:
                    return et.id
        # Never happen
        return None

    @staticmethod
    def _guess_n_columns(path):
        data = pd.read_csv(path, header=0, nrows=1)
        return len(data.columns)


