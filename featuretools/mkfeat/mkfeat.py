#!/usr/bin/env python3

import sys
import os
from os.path import dirname
import logger


def usage():
    print("""\
Usage: mkfeat.py <jobid> <csv path> <result path>
""")


if __name__ == "__main__":
    logger.init("mkfeat")

    topdir = dirname(dirname(dirname(__file__)))
    sys.path.insert(0, topdir)
    from featuretools.mkfeat.feat_extractor import FeatureExtractor

    if len(sys.argv) < 4:
        usage()
        exit(1)

    if not os.path.isfile(sys.argv[2]):
        logger.error("file not found: {}".format(sys.argv[2]))
        exit(1)
    if os.path.exists(sys.argv[3]):
        logger.error("result path already exist: {}".format(sys.argv[3]))
        exit(1)

    extractor = FeatureExtractor()
    extractor.load(sys.argv[2])
    extractor.extract_features(sys.argv[1])
    extractor.save(sys.argv[3])

    exit(0)
