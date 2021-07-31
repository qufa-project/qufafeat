class OperatorManager:
    """
    현재 설계상으로는 특징 추출시 연산자를 사용자가 지정하도록 되어 있으나, 사용자가 연산자를 지정하지 않는 방식으로 특징 추출이 구현될
    수 있음. 이러한 경우를 지원하는 용도로 이 클래스가 향후 활용될 수 있음
    지금 구현에서는 사용자는 단순히 operator 이름만 지정하는데, featuretool의 dfs 입력은 transform이나 aggregation 타입을 구분하여
    operator를 지정해야 함. 이를 위해 사용자가 지정한 operator에 대해 해당 유형을 적절히 반환할 수 있도록 해야 함
    """
    def __init__(self, operators):
        self.operators = set(operators)
        self.transforms = {
            "Absolute"
        }
        self.aggregations = {
            "Sum"
        }

    def get_transform_operators(self):
        return set(self.operators & self.transforms)

    def get_aggregation_operators(self):
        return set(self.operators & self.aggregations)
