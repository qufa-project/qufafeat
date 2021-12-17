from pandas import DataFrame


class RowSet:
    def __init__(self, df: DataFrame):
        self.cnset: frozenset = frozenset(df.columns)

        rowset_map = {}
        for row in df.iterrows():
            idx = row[0]
            tpl = tuple(row[1].values)
            if tpl not in rowset_map:
                rowset_map[tpl] = set()

            rowset_map[tpl].add(idx)

        self._rowset = set()
        for item in rowset_map.values():
            self._rowset.add(frozenset(item))

    def get_desc(self):
        return "(" + ",".join(self.cnset) + ")"

    def __repr__(self):
        return self.get_desc() + ': ' + str(self._rowset)

    def issubset(self, rg):
        for rg_in_rs in self._rowset:
            if rg.issubset(rg_in_rs):
                return True
        return False

    def has_dep(self, rs):
        for rg in rs._rowset:
            if not self.issubset(rg):
                return False
        return True
