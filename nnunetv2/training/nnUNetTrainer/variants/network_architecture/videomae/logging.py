# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


"""Logging."""

import atexit
import builtins
import decimal
import functools
import logging
import os
import sys

# import simplejson
import torch
import torch.distributed as dist
# from iopath.common.file_io import g_pathmgr as pathmgr


def is_master_proc(multinode=False):
    """
    Determines if the current process is the master process.
    """
    if dist.is_initialized():
        if multinode:
            return dist.get_rank() % dist.get_world_size() == 0
        else:
            return dist.get_rank() % torch.cuda.device_count() == 0
    else:
        return True


def _suppress_print():
    """
    Suppresses printing from the current process.
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


# @functools.lru_cache(maxsize=None)
# def _cached_log_stream(filename):
#     # Use 1K buffer if writing to cloud storage.
#     io = pathmgr.open(filename, "a", buffering=1024 if "://" in filename else -1)
#     atexit.register(io.close)
#     return io


def setup_logging(output_dir=None):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """
    # Set up logging format.
    if is_master_proc():
        # Enable logging for the master process.
        logging.root.handlers = []
    else:
        # Suppress logging for non-master processes.
        _suppress_print()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(filename)s: %(lineno)3d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )

    if is_master_proc():
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(plain_formatter)
        logger.addHandler(ch)

    if output_dir is not None and is_master_proc(multinode=True):
        filename = os.path.join(output_dir, "stdout.log")
        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)


# def log_json_stats(stats):
#     """
#     Logs json stats.
#     Args:
#         stats (dict): a dictionary of statistical information to log.
#     """
#     stats = {
#         k: decimal.Decimal("{:.5f}".format(v)) if isinstance(v, float) else v
#         for k, v in stats.items()
#     }
#     json_stats = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
#     get_logger(__name__)
#     print("json_stats: {:s}".format(json_stats))


def master_print(*args, **kwargs):
    if is_master_proc():
        print(*args, **kwargs)
    else:
        pass
