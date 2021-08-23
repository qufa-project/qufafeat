import featuretools as ft
import os.path

from .qufa_ES import QufaES
from .columnspec import ColumnSpec
from .opmgr import OperatorManager
from .feathelper import FeatureHelper
from .error import Error
from .extract_phase import ExtractPhase
from .elapsed_time import ElapsedTime
import featsel


class FeatureExtractor:
    """
    특징 추출 클래스
    """

    def __init__(self, path_input: str, columns: dict, proghandler: callable):
        """

        Args:
            path_input: CSV 형식의 데이터 테이블 입력 파일 경로. CSV는 헤더가 있거나 없을 수 있음
            columns: 데이터 테이블의 컬럼들에 대한 정보. 서비스에 맞게 추후 개발 필요. 현재는 활용되고 있지 않으나, 추후 활용 필요
            proghandler: 특징 추출 작업시 진행율을 전달 받는 callback 함수. proghandler(prog: int) 형태. 0에서 100사이의 값으로
                작업이 진행될 때 마다 증가. 100의 경우 작업 완료를 의미함
        """
        self.es = None

        self.feature_matrix = None
        self.feature_helper = None
        self._path_input = path_input
        self._columns = columns
        self._proghandler = proghandler
        self._prog = None
        self._elapsed_time = ElapsedTime()

    def _load(self) -> Error:
        """
        CSV형식의 테이블 데이터를 로딩. 내부 호출 방식으로 변경

        Returns:
            Error.OK : 성공적으로 로딩한 경우
            Error.ERR_DATA_NOT_FOUND: 데이터 경로가 존재하지 않음
        """
        if not os.path.isfile(self._path_input):
            return Error.ERR_DATA_NOT_FOUND
        colspec = ColumnSpec(self._columns)
        err = colspec.validate()
        if err != Error.OK:
            return err
        self.es = QufaES()
        return self.es.load_from_csv(self._path_input, self._progress_report, colspec)

    def _progress_report(self, prog, phase: ExtractPhase):
        if phase == ExtractPhase.READ_CSV:
            prog = int(prog * 0.1)
        else:
            prog = int(10 + prog * 0.9)

        if prog >= 100:
            prog = 99
        if self._proghandler is not None:
            self._proghandler(prog)
        self._prog = prog

    def _progress_report_dfs(self, update, progress_percent, time_elapsed):
        self._progress_report(int(progress_percent), ExtractPhase.DFS)

    def extract_features(self, operators: list) -> Error:
        """
        특징 추출 작업 시작. 데이터 크기 및 operator 개수에 따라 수십분 이상의 시간이 소요될 수 있음.
        진행율 처리를 위해 로딩 작업과 통합됨

        Args:
            operators: 특징 추출시 적용하고자 하는 특징 연산자

        Returns:
            Error.OK : 성공적으로 특징 추출
            Error.ERR_DATA_NOT_FOUND: 데이터 경로가 존재하지 않음
        """

        self._elapsed_time.mark()

        err = self._load()
        if err != Error.OK:
            return err

        self._elapsed_time.mark()

        opmgr = OperatorManager(operators)
        feature_matrix, features = ft.dfs(entityset=self.es, target_entity=self.es.target_entity_name,
                                          trans_primitives=opmgr.get_transform_operators(),
                                          agg_primitives=opmgr.get_aggregation_operators(),
                                          progress_callback=self._progress_report_dfs, max_depth=3)
        self._elapsed_time.mark()
        self.feature_matrix, features = featsel.select_features(feature_matrix, features, int(len(self._columns) * 1.5),
                                                                self._elapsed_time)
        self._elapsed_time.mark()

        self.feature_helper = FeatureHelper(features)
        df_skip = self.es.get_df_skip()
        if df_skip is not None:
            self.feature_matrix = self.feature_matrix.join(df_skip)

        if self._proghandler is not None:
            self._proghandler(100)
        self._prog = 100

        return Error.OK

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
        if self._prog is None:
            return Error.ERR_GENERAL
        if self.feature_matrix is None:
            return Error.ERR_ONGOING
        return self.feature_helper.to_array()

    def get_elapsed_secs(self):
        return self._elapsed_time.get_elapsed_secs()
