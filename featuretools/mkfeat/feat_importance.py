import os.path
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from .error import Error


class TrainCallback(xgb.callback.TrainingCallback):
    def __init__(self, proghandler: callable, n_epochs):
        self.proghandler = proghandler
        self.n_epochs = n_epochs

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        prog = int(100.0 * epoch / self.n_epochs)
        if prog >= 100:
            prog = 99
        if self.proghandler:
            self.proghandler(prog)
        return False


class FeatureImportance:
    def __init__(self):
        self.data = None
        self.label = None
        self.model = None
        self.n_epochs = 300

    def load(self, path_data: str, path_label: str):
        if not os.path.isfile(path_data):
            return Error.ERR_DATA_NOT_FOUND
        if not os.path.isfile(path_label):
            return Error.ERR_LABEL_NOT_FOUND

        self.data = pd.read_csv(path_data)
        self.label = pd.read_csv(path_label)
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
        return self.model.get_fscore()
