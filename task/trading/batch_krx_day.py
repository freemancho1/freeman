import os
import re
import shutil
import pandas as pd

from tqdm import tqdm
from datetime import datetime, timedelta
from dateutil.parser import parse

from freeman.task.trading.config import *
from freeman.task.trading.krx_crawler import KrxDataCrawler
from freeman.utils.logs.log import Logger as log


def krx_crawling():
    last_crawling_date = parse(max(os.listdir(KRX_BACKUP_PATH))[:8]).date()
    s_date = last_crawling_date + timedelta(days=1)
    crawler = KrxDataCrawler(s_date=s_date, e_date=None)
    crawler.start_crawling(is_display_processing_step=True)
    
def merge_crawling_data():
    if os.path.isfile(KRX_MERGE_FILE_INFO):
        merge_df = pd.read_csv(KRX_MERGE_FILE_INFO, sep=",")
    else:
        merge_df = pd.DataFrame()
    log.debug(f"작업 전 통합파일 크기: {merge_df.shape}")
    
    def file_processing(csv_file_name):
        add_df = pd.read_csv(os.path.join(KRX_ORIGINAL_PATH, csv_file_name),
                             delimiter=",", encoding="CP949", skiprows=[0],
                             names=KRX_COLUMN_NAMES)
        add_df["date"] = parse(str(re.findall("\d{8}", csv_file_name)[0])).date()
        add_df = add_df.drop(["m_dept"], axis=1)
        shutil.move(os.path.join(KRX_ORIGINAL_PATH, csv_file_name),
                    os.path.join(KRX_BACKUP_PATH, csv_file_name))
        return add_df
    
    for file_name in tqdm(sorted(os.listdir(KRX_ORIGINAL_PATH))):
        add_df = file_processing(file_name)
        merge_df = pd.concat([merge_df, add_df])
        
    merge_df.to_csv(KRX_MERGE_FILE_INFO, index=False)
    log.debug(f"작업 후 통합파일 크기: {merge_df.shape}")
    
    
if __name__ == "__main__":
    # krx_crawling()
    merge_crawling_data()
    