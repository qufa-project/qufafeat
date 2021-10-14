from typing import Set


class IteratorChild:
    def __init__(self, link_childs):
        self._iter = link_childs.__iter__()

    def __next__(self):
        link_child = self._iter.__next__()
        return link_child.rhs
