import logging
from .kaggle_bengali.consts import LOG_FILENAME, LOG_LEVEL

def get_logger(name):
    fh=logging.FileHandler(LOG_FILENAME)
    fh.setLevel(LOG_LEVEL)
    '''
    sh=logging.StreamHandler()
    sh.setLevel(LOG_LEVEL)
    '''
    logger=logging.getLogger(name)
    logger.addHandler(fh)
    #logger.addHandler(sh)
    return logger
