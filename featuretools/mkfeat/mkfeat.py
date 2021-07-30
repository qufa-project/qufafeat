#!/usr/bin/env python3

import sys
import os
from os.path import dirname
import logger
import json
from error import Error


def usage():
    print("""\
Usage: mkfeat.py <configuration in json>
""")


def load_conf(path_conf: str):
    with open(sys.argv[1]) as f:
        try:
            conf = json.load(f)
        except json.decoder.JSONDecodeError:
            logger.error("configuration has wrong format")
            exit(2)
        if 'path_input' not in conf or 'path_output' not in conf:
            logger.error("configuration does not have path_input or path_output")
        if 'columns' not in conf:
            logger.error("configuration has no column information")
            exit(2)
        return conf


def handle_progress(prog: int):
    print("\rprogress: {}%".format(prog), end='')


if __name__ == "__main__":
    logger.init("mkfeat")

    topdir = dirname(dirname(dirname(__file__)))
    sys.path.insert(0, topdir)
    from featuretools.mkfeat.feat_extractor import FeatureExtractor

    if len(sys.argv) < 2:
        usage()
        exit(1)

    if not os.path.isfile(sys.argv[1]):
        logger.error("configuration file not found: {}".format(sys.argv[1]))
        exit(1)

    conf = load_conf(sys.argv[1])

    if not os.path.exists(conf['path_input']):
        logger.error("input path does not exist: {}".format(conf['path_input']))
        exit(1)

    if os.path.exists(conf['path_output']):
        logger.error("output path already exist: {}".format(conf['path_output']))
        exit(1)

    extractor = FeatureExtractor()
    err = extractor.load(conf['path_input'], conf['columns'])
    if err != Error.OK:
        logger.error("load error: {}".format(err))
        exit(2)
    extractor.extract_features(handle_progress)
    extractor.save(conf['path_output'])
    print()
