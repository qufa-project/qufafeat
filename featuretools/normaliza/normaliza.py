from pandas import DataFrame
from typing import List

from .coldeptree import ColDepTree
from .coldepset import ColDepSet
from .norminfo import NormInfo


n_samples = 1000


def get_norminfos_for_es(df: DataFrame) -> List[NormInfo]:
    coldeps = _get_coldeps_by_iterative_sampling(df)
    tree = ColDepTree()
    tree.build(coldeps)
    tree.collapse_roots()
    tree.make_single_parent()
    tree.subsumes_children()
    return tree.get_norminfos()


def _get_coldeps_by_iterative_sampling(df: DataFrame):
    coldeps = ColDepSet(True)

    if len(df) < n_samples * 10:
        coldeps.analyze(df)
    else:
        n_stable_checked = 0
        while True:
            df_samp = df.sample(n=n_samples)
            if coldeps.analyze(df_samp):
                n_stable_checked += 1
                if n_stable_checked > 3:
                    break
            else:
                n_stable_checked = 0

    return coldeps
