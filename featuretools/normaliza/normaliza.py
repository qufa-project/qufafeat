from pandas import DataFrame
from typing import List

from .coldeptree import ColDepTree
from .norminfo import NormInfo


def get_norminfos_for_ES(df: DataFrame) -> List[NormInfo]:
    tree = ColDepTree(df, True)
    tree.collapse_roots()
    tree.make_single_parent()
    tree.subsumes_children()
    return tree.get_norminfos()
