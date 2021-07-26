#!/usr/bin/env python3

import sys
import os
from os.path import dirname
import logger


def usage():
    print("""\
Usage: getprog.py <jobid>
""")


if __name__ == "__main__":
    logger.init("getprog")

    topdir = dirname(dirname(dirname(__file__)))
    sys.path.insert(0, topdir)

    if len(sys.argv) < 2:
        usage()
        exit(1)

    from featuretools.mkfeat.ipc import IPC
    myipc = IPC(sys.argv[1])
    print(myipc.get_prog())
