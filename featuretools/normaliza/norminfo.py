class NormInfo:
    def __init__(self, cnset: frozenset, idx_parent: frozenset, cnset_key: frozenset):
        self.cnset = cnset
        self.idx_parent = idx_parent
        self.cnset_key = cnset_key

    def __repr__(self):
        return str(self.idx_parent) + ":" + ",".join(self.cnset_key) + ":" + ",".join(self.cnset)

    def get_key(self):
        for cn in self.cnset_key:
            return cn
        return None

    def get_additional_vars(self):
        return list(self.cnset)
