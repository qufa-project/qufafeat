from enum import Enum


class Error(Enum):
    """
    mkfeat과 관련된 일체의 오류를 정의. Exception보다는 반환값으로 API를 정의하고자 함
    """
    OK = 0
    ERR_GENERAL = -1
    ERR_COLUMN_COUNT_MISMATCH = -2