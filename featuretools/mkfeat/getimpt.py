#!/usr/bin/env python3

import sys
import os
from os.path import dirname
import logger


def usage():
    print("""\
Usage: mkimpt.py <data path> <label path>
""")


def _handle_progress(prog: int):
    print("\rprogress: {}%".format(prog), end='')


if __name__ == "__main__":
    logger.init("mkimpt")

    topdir = dirname(dirname(dirname(__file__)))
    sys.path.insert(0, topdir)
    from featuretools.mkfeat.feat_importance import FeatureImportance

    if len(sys.argv) < 3:
        usage()
        exit(1)

    if not os.path.isfile(sys.argv[1]):
        logger.error("data file not found: {}".format(sys.argv[1]))
        exit(1)
    if not os.path.exists(sys.argv[2]):
        logger.error("label file not found: {}".format(sys.argv[2]))
        exit(1)

    impt = FeatureImportance()
    impt.load(sys.argv[1], sys.argv[2])
    impt.analyze(_handle_progress)
    print(impt.get_importance())

    exit(0)
