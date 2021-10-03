import featuretools.normaliza.coldepnode as coldepnode


class ColDepLink:
    def __init__(self, lhs: coldepnode.ColDepNode, rhs: coldepnode.ColDepNode,
                 cnset_lhs: frozenset, cnset_rhs: frozenset):
        self.lhs = lhs
        self.rhs = rhs
        self.cnset_lhs = cnset_lhs
        self.cnset_rhs = cnset_rhs

    def __repr__(self):
        return "(" + ",".join(self.cnset_lhs) + ")->" + "(" + ",".join(self.cnset_rhs) + ")"

