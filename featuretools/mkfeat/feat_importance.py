import os.path
import xgboost as xgb
from sklearn.model_selection import train_test_split

from .error import Error
from .progress_phase import ProgressPhase
from .columnspec import ColumnSpec
from .qufa_csv import QufaCsv


class TrainCallback(xgb.callback.TrainingCallback):
    def __init__(self, proghandler: callable, n_epochs):
        self.proghandler = proghandler
        self.n_epochs = n_epochs

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        prog = int(100.0 * epoch / self.n_epochs)
        if self.proghandler:
            return self.proghandler(prog, ProgressPhase.IMPORTANCE)
        return False


class FeatureImportance:
    def __init__(self, path_data: str, columns_data: dict, path_label: str, columns_label: dict, proghandler: callable):
        self.data = None
        self.label = None
        self.model = None
        self.n_epochs = 300
        self._path_data = path_data
        self._columns_data = columns_data
        self._path_label = path_label
        self._columns_label = columns_label
        self._proghandler = proghandler
        self._colspec_data: ColumnSpec = None

    def _load(self) -> Error:
        if self._path_data is None or self._columns_data is None:
            return Error.ERR_INVALID_ARG
        if not os.path.isfile(self._path_data):
            return Error.ERR_DATA_NOT_FOUND
        if self._path_label is not None:
            if self._columns_label is None:
                return Error.ERR_INVALID_ARG
            if not os.path.isfile(self._path_label):
                return Error.ERR_LABEL_NOT_FOUND

        self._colspec_data = colspec_data = ColumnSpec(self._columns_data)
        if self._path_label is None:
            if colspec_data.get_label_colname() is None:
                return Error.ERR_LABEL_NOT_FOUND

        csv_data = QufaCsv(self._path_data, colspec_data)
        data = csv_data.load(self._progress_report, exclude_skip=True, numeric_only=True)
        if isinstance(data, Error):
            return data
        self.data = data

        if self._path_label is None:
            label = csv_data.load(None, label_only=True)
        else:
            colspec_label = ColumnSpec(self._columns_label)
            csv_label = QufaCsv(self._path_label, colspec_label)
            label = csv_label.load(None, numeric_only=True)
        if isinstance(label, Error):
            return label
        self.label = label
        return Error.OK

    def _progress_report(self, prog, phase: ProgressPhase):
        if phase == ProgressPhase.READ_CSV:
            prog = int(prog * 0.2)
        else:
            prog = int(20 + prog * 0.8)

        if prog >= 100:
            prog = 99
        if self._proghandler is not None:
            self._proghandler(prog)
        self._prog = prog

    def analyze(self) -> Error:
        err = self._load()
        if err != Error.OK:
            return err

        xtr, xv, ytr, yv = train_test_split(self.data.values, self.label.values, test_size=0.2, random_state=0)
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

        callback = TrainCallback(self._progress_report, self.n_epochs)
        self.model = xgb.train(params=params, dtrain=dtrain, num_boost_round=self.n_epochs,
                               callbacks=[callback],
                               evals=evals, early_stopping_rounds=60, maximize=False, verbose_eval=10)
        if self._proghandler is not None:
            self._proghandler(100)

        return Error.OK

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
