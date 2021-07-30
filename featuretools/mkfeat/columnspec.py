class ColumnSpec:
    """
    컬럼명이나 유형과 관련된 정보를 처리하는 목적의 클래스. 현재는 컬럼명에 대한 정보를 처리하는 기능만 구현됨
    """
    def __init__(self, columns):
        self.columns = columns

    def get_colnames(self):
        """
        컬럼명 배열을 반환. pandas의 read_csv() 함수 전달 인자를 쉽게 생성하기 위함

        Returns:
            컬럼명으로 구성된 배열
        """
        colnames = []
        for colinfo in self.columns:
            colnames.append(colinfo['name'])
        return colnames

    def get_key_colname(self):
        """
        특징 추출시 id로 지정가능한 컬럼명 반환. key로 지정된 column명이 없는 경우 첫번째 컬럼명 반환

        Returns:
            id로 지정 가능한 column name.
        """
        for colinfo in self.columns:
            if 'key' in colinfo and colinfo['key']:
                return colinfo['name']
        return self.columns[0]['name']
