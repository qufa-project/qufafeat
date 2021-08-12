from featuretools.entityset import EntitySet
import autonormalize as an
import pandas as pd

from .columnspec import ColumnSpec
from .error import Error
from .qufa_csv import QufaCsv
from .normalize import normalize


class QufaES(EntitySet):
    def __init__(self):
        super().__init__()
        self.target_entity_name = None
        self._df_label = None

    def load_from_csv(self, path, colspec: ColumnSpec) -> Error:
        csv = QufaCsv(path, colspec)
        data = csv.load()

        colname_key = colspec.get_key_colname()
        colname_label = colspec.get_label_colname()
        if colname_label:
            self._df_label = data[[colname_key, colname_label]]
            data = data.drop(columns=colname_label)

        norminfos = normalize(data, colname_key)

        self.entity_from_dataframe("main", data, index=colname_key)
        for norminfo in norminfos:
            keyname = norminfo[0]
            vars = norminfo[1:]
            etname = self._search_owner_entity(keyname)
            self.normalize_entity(etname, "tbl_{}".format(keyname), norminfo[0], additional_variables=vars)

        self.target_entity_name = "main"

        return Error.OK

    def get_df_label(self):
        return self._df_label

    def _search_owner_entity(self, varname):
        for et in self.entities:
            for var in et.variables:
                if var.name == varname:
                    return et.id
        # Never happen
        return None

