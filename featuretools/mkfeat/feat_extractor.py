import featuretools as ft
from featuretools.mkfeat.qufa_ES import QufaES

import ipc


class FeatureExtractor:
    """
    특징 추출 클래스
    """

    def __init__(self):
        self.es = None

        self.feature_matrix = None
        self.features = None
        self.ipc = None

    def load(self, path):
        """
        CSV 데이터를 로딩함

        Args:
            path: CSV 파일 경로

        Returns:

        """
        self.es = QufaES()
        self.es.load_from_csv(path)

    def _progress_report(self, update, progress_percent, time_elapsed):
        self.ipc.set_prog(progress_percent)

    def extract_features(self, ipc_name):
        """
        특징 추출 작업 시작. 데이터 크기 및 operator 개수에 따라 수십분 이상의 시간이 소요될 수 있음

        Args:
            ipc_name: 진행율을 기록할 파일명

        Returns:
            특정한 반환값 없음. 오류 발생시 exception이 발생할 수 있음
        """
        self.ipc = ipc.IPC(ipc_name)
        self.ipc.create()

        self.feature_matrix, self.features = ft.dfs(entityset=self.es, target_entity="main", progress_callback=self._progress_report)
        self.ipc.set_complete()

    def save(self, path):
        """
        추출된 특징 데이터로 결과를 저장함

        Args:
            path: 저장하고자 하는 CSV 경로

        Returns:

        """
        self.feature_matrix.to_csv(path)
