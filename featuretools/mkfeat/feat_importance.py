import os.path
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from .error import Error
from .columnspec import ColumnSpec
from .qufa_csv import QufaCsv


class TrainCallback(xgb.callback.TrainingCallback):
    def __init__(self, proghandler: callable, n_epochs):
        self.proghandler = proghandler
        self.n_epochs = n_epochs

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        prog = int(100.0 * epoch / self.n_epochs)
        if prog >= 100:
            prog = 99
        if self.proghandler:
            return self.proghandler(prog)
        return False


class FeatureImportance:
    def __init__(self):
        self.data = None
        self.label = None
        self.model = None
        self.n_epochs = 300
        self._colspec_data: ColumnSpec = None

    def load(self, path_data: str, columns_data: dict, path_label: str, columns_label: dict) -> Error:
        if path_data is None or columns_data is None:
            return Error.ERR_INVALID_ARG
        if not os.path.isfile(path_data):
            return Error.ERR_DATA_NOT_FOUND
        if path_label is not None:
            if columns_label is None:
                return Error.ERR_INVALID_ARG
            if not os.path.isfile(path_label):
                return Error.ERR_LABEL_NOT_FOUND

        self._colspec_data = colspec_data = ColumnSpec(columns_data)
        if path_label is None:
            if colspec_data.get_label_colname() is None:
                return Error.ERR_LABEL_NOT_FOUND

        csv_data = QufaCsv(path_data, colspec_data)
        exclude_label = True if path_label is None else False
        data = csv_data.load(exclude_label=exclude_label, numeric_only=True)
        if isinstance(data, Error):
            return data
        self.data = data

        if path_label is None:
            label = csv_data.load(label_only=True)
        else:
            colspec_label = ColumnSpec(columns_label)
            csv_label = QufaCsv(path_label, colspec_label)
            label = csv_label.load()
        if isinstance(label, Error):
            return label
        self.label = label
        return Error.OK

    def analyze(self, proghandler: callable = None):
        xtr, xv, ytr, yv = train_test_split(self.data.values, self.label, test_size=0.2, random_state=0)
        dtrain = xgb.DMatrix(xtr, label=ytr)
        dvalid = xgb.DMatrix(xv, label=yv)

        evals = [(dtrain, 'train'), (dvalid, 'valid')]

        params = {
            'min_child_weight': 1, 'eta': 0.166,
            'colsample_bytree': 0.4, 'max_depth': 9,
            'subsample': 1.0, 'lambda': 57.93,
            'booster': 'gbtree', 'gamma': 0.5,
            'silent': 1, 'eval_metric': 'rmse',
            'objective': 'reg:linear',
        }

        callback = TrainCallback(proghandler, self.n_epochs)
        self.model = xgb.train(params=params, dtrain=dtrain, num_boost_round=self.n_epochs,
                               callbacks=[callback],
                               evals=evals, early_stopping_rounds=60, maximize=False, verbose_eval=10)
        if proghandler is not None:
            proghandler(100)

    def get_importance(self):
        fscores = self.model.get_fscore()
        fscore_sum = 0
        for i in range(len(self.data.columns)):
            colname = 'f' + str(i)
            if colname in fscores:
                fscore_sum += fscores[colname]
        importances = []
        for i in range(len(self.data.columns)):
            colname = 'f' + str(i)
            if colname in fscores:
                importances.append(fscores[colname] / fscore_sum)
            else:
                importances.append(0.0)
        idx = 0
        for is_numeric in self._colspec_data.get_is_numerics():
            if not is_numeric:
                importances.insert(idx, 0)
            idx += 1
        return importances
