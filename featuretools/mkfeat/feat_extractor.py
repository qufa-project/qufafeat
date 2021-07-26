import featuretools as ft
from featuretools.mkfeat.qufa_ES import QufaES

import ipc


class FeatureExtractor:
    def __init__(self):
        self.es = None

        self.feature_matrix = None
        self.features = None
        self.ipc = None

    def load(self, path):
        self.es = QufaES()
        self.es.load_from_csv(path)

    def _progress_report(self, update, progress_percent, time_elapsed):
        self.ipc.set_prog(progress_percent)

    def extract_features(self, ipc_name):
        self.ipc = ipc.IPC(ipc_name)
        self.ipc.create()

        self.feature_matrix, self.features = ft.dfs(entityset=self.es, target_entity="main", progress_callback=self._progress_report)
        self.ipc.set_complete()

    def save(self, path):
        self.feature_matrix.to_csv(path)
