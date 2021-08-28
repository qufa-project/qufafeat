from enum import Enum


class ProgressPhase(Enum):
    READ_CSV = 1
    DFS = 2
    REMOVE_SINGLE = 3
    REMOVE_CORREL = 4
    SELECT_BEST = 5
    IMPORTANCE = 6
