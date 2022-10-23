import os
import platform
from enum import Enum, IntEnum

# System Info
USER_PATH = os.path.expanduser("~")
PACKAGE_NAME = "freeman"
PROJECT_NAME = "trading"
PROJECT_BASE = os.path.join(USER_PATH, "projects")
PROJECT_PATH = os.path.join(PROJECT_BASE, PACKAGE_NAME, "task", PROJECT_NAME)
PROJECT_SYSFILE_PATH = PROJECT_PATH
DATA_PATH = os.path.join(PROJECT_BASE, "data")

if 'Windows' in platform.platform():
    WEB_DRIVER_PATH = os.path.join(USER_PATH, '.local', 'bin', 'chromedriver.exe')
else:
    WEB_DRIVER_PATH = os.path.join(USER_PATH, '.local', 'bin', 'chromedriver')
KRX_CRAWLING_URL = 'http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201020101'

# Crawling 
CRAWLING_PATH = os.path.join(DATA_PATH, "crawling")
## KRX Directory Info
KRX_CRAWLING_PATH = os.path.join(CRAWLING_PATH, "krx")
KRX_DOWNLOAD_PATH = os.path.join(USER_PATH, "download")
KRX_ORIGINAL_PATH = os.path.join(KRX_CRAWLING_PATH, "original")
KRX_BACKUP_PATH = os.path.join(KRX_CRAWLING_PATH, "backup")
KRX_MERGE_PATH = os.path.join(KRX_CRAWLING_PATH, "merge")
KRX_MERGE_FILE = "krx-merge.csv"
KRX_MERGE_FILE_INFO = os.path.join(KRX_MERGE_PATH, KRX_MERGE_FILE)
KRX_MERGE_1012_INFO = os.path.join(KRX_MERGE_PATH, "krx-merge-20221012.csv")
KRX_COMPANY_FILE = "company_list.csv"
KRX_COMPANY_FILE_INFO = os.path.join(KRX_MERGE_PATH, KRX_COMPANY_FILE)
# KRX Trading Info
FIRST_TRADING_DATE = "20000529"
BASE_TIME_INCLUDING_TODAY = "17:00:00"
DOWNLOAD_WATING_TIME_BEFORE_BASE_TIME = 10  # 장중
DOWNLOAD_WATING_TIME_AFTER_BASE_TIME = 3   # 폐장 후
# KRX Trading Data Info
KRX_COLUMN_NAMES = [ 'com_code', 'com_name', 'm_type', 'm_dept',
                     'close', 'diff', 'ratio', 'open', 'high', 'low', 'volume',
                     'value', 't_value', 't_volume' ]
KRX_COMPANY_COLUMNS = [
    'code', 'short_code', 'name_kr', 'short_name_kr', 'name_en', 'listing_date',
    'type_market', 'type_stock', 'affiliated', 'category_stock', 'par_value', 'num_listed'
]

# AI
SPECIAL_COMPANY = '삼성전자'