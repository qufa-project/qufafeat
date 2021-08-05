import featuretools as ft

from .qufa_ES import QufaES
from .columnspec import ColumnSpec
from .opmgr import OperatorManager
from .feathelper import FeatureHelper
from .error import Error


class FeatureExtractor:
    """
    특징 추출 클래스
    """

    def __init__(self):
        self.es = None

        self.feature_matrix = None
        self.feature_helper = None
        self.proghandler = None
        self._prog = None

    def load(self, path: str, columns) -> Error:
        """
        CSV형식의 테이블 데이터를 로딩

        Args:
            columns: 데이터 테이블의 컬럼들에 대한 정보. 서비스에 맞게 추후 개발 필요. 현재는 활용되고 있지 않으나, 추후 활용 필요
            path: CSV 형식의 데이터 테이블 입력 파일 경로. CSV는 헤더가 있거나 없을 수 있음. (추후 고려 필요)

        Returns:
            현재는 특별한 값을 반환하지 않으나, 보강 필요
        """
        self.es = QufaES()
        colspec = ColumnSpec(columns)
        return self.es.load_from_csv(path, colspec)

    def _progress_report(self, update, progress_percent, time_elapsed):
        prog = int(progress_percent)
        if prog >= 100:
            prog = 99
        if self.proghandler is not None:
            self.proghandler(prog)
        self._prog = prog

    def extract_features(self, operators, proghandler: callable = None):
        """
        특징 추출 작업 시작. 데이터 크기 및 operator 개수에 따라 수십분 이상의 시간이 소요될 수 있음

        Args:
            operators: 특징 추출시 적용하고자 하는 특징 연산자
            proghandler: 특징 추출 작업시 진행율을 전달 받는 callback 함수. proghandler(prog: int) 형태. 0에서 100사이의 값으로
                작업이 진행될 때 마다 증가. 100의 경우 작업 완료를 의미함

        Returns:
            현재로서는 특정한 반환값 없음. 추후 필요시 반환값 혹은 exception을 발생시킬 수 있음
        """

        opmgr = OperatorManager(operators)
        self.proghandler = proghandler
        self.feature_matrix, features = ft.dfs(entityset=self.es, target_entity=self.es.target_entity_name,
                                               trans_primitives=opmgr.get_transform_operators(),
                                               agg_primitives=opmgr.get_aggregation_operators(),
                                               progress_callback=self._progress_report, max_depth=3)
        self.feature_helper = FeatureHelper(features)

        if proghandler is None:
            proghandler(100)
        self._prog = 100

    def get_progress(self):
        """
            progress handler를 등록하지 않고, Polling 방식으로 진행율을 얻는 경우 활용 가능(웹서비스에서는 이 방식이 편리할 듯)

        Returns:
            int: 진행율 정보
        """
        return self._prog

    def save(self, path):
        """
        추출된 특징 데이터로 결과를 저장함

        Args:
            path: 저장하고자 하는 CSV 경로

        Returns:

        """
        self.feature_matrix.to_csv(path)

    def get_feature_info(self):
        """
        추출 완료된 특징들에 대한 정보 반환. 현재는 각 특징에 대한 특징명과 타입을 문자열로 반환함. 파생특징에 대한 이름으로 부터 어떠한 특징과 연산자에 기반하여
        생성되었는지 확인이 가능함.

        Returns:
            list[list]:
                성공적으로 수행된 경우 반환되는 값. 각 배열 항목은 추출된 특징의 컬럼명과 형식에 대한 문자열임. 다음과 같은 형식.
                [ [ "colname1", "number" ], [ "colname2", "string" ] .. ]
            :class:`.Error`:
                오류가 발생한 경우 반환됨. 반환 가능한 오류값은
                - ERR_GENERAL: 알수 없는 오류. 특징 추출이 진행되지 않는 경우도 해당함
                - ERR_ONGOING: 특징 추출 작업이 진행중인 경우
        """
        if self.proghandler is None:
            return Error.ERR_GENERAL
        if self.feature_matrix is None:
            return Error.ERR_ONGOING
        return self.feature_helper.to_array()
