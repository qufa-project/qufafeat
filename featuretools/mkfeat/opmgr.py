import featuretools as ft
from featuretools.primitives import utils


class OperatorManager:
    """
    1.
    현재 설계 상으로는 특징 추출시 연산자를 사용자가 지정하도록 되어 있으나, 사용자가 연산자를 지정하지 않는 방식으로 특징 추출이 구현될 수 있음.
    이러한 경우, 전체 operator를 반환하여 특징 추출을 지원하는 용도로 이 클래스를 활용함.
    2.
    지금 구현에서는 사용자가 단순히 operator 이름만 지정하는데, featuretool의 dfs 입력은 transform이나 aggregation 타입을 구분하여 operator를 지정해야 함.
    이를 위해 사용자가 지정한 operator에 해당하는 유형을 반환함.
    """
    def __init__(self, operators: list):
        if not operators:
            self.operators = None
        else:
            self.operators = set(str.lower(op) for op in operators)
        self.transforms = set(utils.get_transform_primitives().keys())
        self.aggregations = set(utils.get_aggregation_primitives().keys())

    def get_transform_operators(self):
        if self.operators is None:
            return self.transforms
        return set(self.operators & self.transforms)

    def get_aggregation_operators(self):
        if self.operators is None:
            return self.aggregations
        return set(self.operators & self.aggregations)
