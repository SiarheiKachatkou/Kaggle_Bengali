import logging
import os
from .kaggle_bengali.consts import LOG_FILENAME, LOG_LEVEL

def get_logger(name):
    fh=logging.FileHandler(LOG_FILENAME)
    fh.setLevel(LOG_LEVEL)

    sh=logging.StreamHandler()
    sh.setLevel(LOG_LEVEL)

    logger=logging.getLogger(name)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.setLevel(LOG_LEVEL)

    with open(os.path.join(os.path.dirname(__file__),'kaggle_bengali/consts.py'),'rt') as file:
        consts=file.read()
    logger.info(consts)

    return logger
