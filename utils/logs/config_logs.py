import os
from enum import Enum, IntEnum

USER_PATH = os.path.expanduser("~")
PROJECT_NAME = "trading"
PROJECT_BASE = os.path.join(USER_PATH, "projects")
DATA_PATH = os.path.join(PROJECT_BASE, "data")

class LogLabel(IntEnum):
    debug       = 1
    info        = 2
    warning     = 3
    error       = 4
    critical    = 5
LOG_LABEL = 'debug'
IS_WRITE = lambda x: LogLabel[x] >= LogLabel[LOG_LABEL]
LOG_PATH = os.path.join(DATA_PATH, 'logs', PROJECT_NAME)
LOG_FILE_NAME_PREFIX = f'{PROJECT_NAME}'
LOG_FILE_NAME = f'{LOG_FILE_NAME_PREFIX}.log'
LOG_FILE_PATH = os.path.join(LOG_PATH, LOG_FILE_NAME)
LOG_FILE_SIZE = '100M'
LOG_FILE_COUNT = 10
class ByteSize(Enum):
    K = 1024
    M = 1024 * 1024
    G = 1024 * 1024 * 1024
    T = 1024 * 1024 * 1024 * 1024
LOG_COLOR = {
    'CRITICAL': '\033[91m',
    '   ERROR': '\033[31m',
    ' WARNING': '\033[95m',
    '    INFO': '\033[93m',
    '   DEBUG': '\033[37m',
}
LOG_COLOR_END = '\033[0m'
class LOG_LEVEL(IntEnum):
    debug               = 1
    info                = 2
    warning             = 3
    error               = 4
    critical            = 5