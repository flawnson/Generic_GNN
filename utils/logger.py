""" This file contains the generic logger code that can be called from within any pipeline to log data about the run
    Custom logger to be used in training and data preprocessing pipelines. Tuning pipeline uses ray tune's logger """

import datetime
import logging
import errno
import time
import os

import os.path as osp
from functools import wraps

log_format = '%(levelname)-8s %(asctime)s %(filename)s:%(lineno)d] %(message)s '
date_format = '%Y-%m-%d %I%M%S %p'


# Callable function to set logger for any module in the repo
def set_file_logger(config: dict, name: str = '', filename: str = "run.log", level=logging.DEBUG):
    if config["logging"]:
        logger = logging.getLogger(name)
        logging.basicConfig(format=log_format,
                            datefmt=date_format,
                            level=level)
        time = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        try:
            os.makedirs(osp.join(osp.dirname(__file__), f"../logs"))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise e
        log_file = osp.join(osp.dirname(__file__), f"../logs", f"{time}-{filename}")
        print(f"Writing logs to {log_file}")
        handler = logging.FileHandler(osp.join(osp.dirname(__file__), f"../logs", f"{time}-{filename}"), mode="a")
        log_formatter = logging.Formatter(fmt=log_format,
                                          datefmt=date_format)
        handler.setFormatter(log_formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        return logger
    else:
        print("Logging has been disabled (WARNING: Note that this also disables logging for Ray Tune's pipeline)")
        logging.disable(level=logging.CRITICAL)


def get_logger(name: str):
    logger = logging.getLogger(name)
    return logger


# Callable function to time the execution time of a function when called
def timed(func):
    """This decorator prints the execution time for the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        get_logger(__name__).info("{} ran in {}s".format(func.__name__, round(end - start, 2)))
        return result

    return wrapper


console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter(log_format, datefmt=date_format)
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

log = get_logger(__name__)
