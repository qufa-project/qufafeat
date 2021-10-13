import featuretools.normaliza.coldepnode as coldepnode


class ColDepLink:
    def __init__(self, lhs: coldepnode.ColDepNode, rhs: coldepnode.ColDepNode,
                 cnset_lhs: frozenset, cnset_rhs: frozenset):
        self.lhs = lhs
        self.rhs = rhs
        self.cnset_lhs = cnset_lhs
        self.cnset_rhs = cnset_rhs

    def is_invalid(self):
        if self.lhs.is_invalid() or self.rhs.is_invalid():
            return True
        if self.lhs == self.rhs:
            return True
        if self.cnset_lhs and self.cnset_lhs == self.cnset_rhs:
            return True
        return False

    def __repr__(self):
        return "(" + ",".join(self.cnset_lhs) + ")->" + "(" + ",".join(self.cnset_rhs) + ")"

    def __eq__(self, other):
        if self.cnset_lhs == other.cnset_lhs and self.cnset_rhs == other.cnset_rhs:
            return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.cnset_lhs, self.cnset_rhs))
