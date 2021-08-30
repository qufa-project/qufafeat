from featuretools.entityset import EntitySet
import numpy as np

from .columnspec import ColumnSpec
from .error import Error
from .qufa_csv import QufaCsv
from .normalize import normalize


class QufaES(EntitySet):
    def __init__(self):
        super().__init__()
        self.target_entity_name = None
        self._is_auto_key = False
        self._df_label = None
        self._df_train = None
        self._df_bypass = None

    def load_from_csv(self, path, callback, colspec: ColumnSpec) -> Error:
        csv = QufaCsv(path, colspec)
        data = csv.load(callback)
        if isinstance(data, Error):
            return data

        colname_key = colspec.get_key_colname()
        if colspec.is_auto_keyname():
            data[colname_key] = np.arange(len(data))
            self._is_auto_key = True
        colname_label = colspec.get_label_colname()
        if colname_label:
            self._df_label = data[[colname_key, colname_label]]
            data.drop(columns=colname_label, inplace=True)
            self._df_label.set_index(colname_key, inplace=True)

        colname_train = colspec.get_train_colname()
        if colname_train:
            self._df_train = data[[colname_key, colname_train]]
            data.drop(columns=colname_train, inplace=True)
            self._df_train.set_index(colname_key, inplace=True)

        colnames_bypass = colspec.get_bypass_colnames()
        if colnames_bypass:
            colnames_bypass.insert(0, colname_key)
            self._df_bypass = data[colnames_bypass]
            self._df_bypass.set_index(colname_key, inplace=True)
            colnames_bypass.remove(colname_key)
            data = data.drop(columns=colnames_bypass)

        try:
            norminfos = normalize(data, colname_key)
        except AssertionError:
            # There are many cases. One observed case is that key index is not unique.
            return Error.ERR_COLUMN_BAD

        self.entity_from_dataframe("main", data, index=colname_key)
        if norminfos:
            for norminfo in norminfos:
                keyname = norminfo[0]
                vars = norminfo[1:]
                etname = self._search_owner_entity(keyname)
                if etname:
                    self.normalize_entity(etname, "tbl_{}".format(keyname), norminfo[0], additional_variables=vars)

        self.target_entity_name = "main"

        return Error.OK

    def is_auto_key(self):
        return self._is_auto_key

    def get_df_label(self):
        return self._df_label

    def get_df_train(self):
        return self._df_train

    def get_df_bypass(self):
        return self._df_bypass

    def _search_owner_entity(self, varname):
        for et in self.entities:
            for var in et.variables:
                if var.name == varname:
                    return et.id
        # Never happen
        return None

