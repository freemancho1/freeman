import os
import sys
import time
import shutil
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from datetime import timedelta, datetime
from dateutil.parser import parse
from tqdm import tqdm

from config import *
from freeman.utils.data import *
from freeman.utils.logs.log import Logger as log
from freeman.utils.logs.log import StartEndLogging


is_including_today = lambda: datetime.now().time() > parse(BASE_TIME_INCLUDING_TODAY).time()

def get_crawling_period(s_date=None, e_date=None):
    today = datetime.now().date()
    first_trading_date = parse(FIRST_TRADING_DATE).date()
    
    try:
        p_first = parse(s_date if type(s_date)=="str" else str(s_date)).date()
    except:
        p_first = first_trading_date
        
    try:
        p_second = parse(e_date if type(e_date)=="str" else str(e_date)).date()
    except:
        p_second = today
        
    (p_first, p_second) = minmax([p_first, p_second])
        
    if p_second > today:
        p_second = today
    if p_second == today and not is_including_today():
        p_second -= timedelta(days=1)
    log.debug(f"시작날자({p_first}), 종료날자({p_second})")
    
    return p_first, p_second  


class KrxDataCrawler:
    
    def __init__(self, s_date=None, e_date=None):
        self.web_driver = webdriver.Chrome()  # ver 4.x
        # self.web_driver = webdriver.Chrome(WEB_DRIVER_PATH)  # ver 3.x
        self.web_driver.get(KRX_CRAWLING_URL)
        time.sleep(2)
        self.s_date, self.e_date = get_crawling_period(s_date, e_date)
        self.ws_date = None
        self.we_date = None
        self.num_work_days = 0
        self.is_start = True
        self.is_stop = False
        if not os.path.exists(KRX_ORIGINAL_PATH):
            os.makedirs(KRX_ORIGINAL_PATH, exist_ok=True)
        
    def start_crawling(self, is_display_processing_step=False):
        se_check = StartEndLogging()
        about_period_months = (self.e_date - self.s_date).days // 30 + 2
        log.info(f"Input Days: {self.s_date} ~ {self.e_date}"
                 f"({(self.e_date-self.s_date).days+1} days)")
        
        if is_display_processing_step:
            period_months = tqdm(range(about_period_months))
        else:
            period_months = range(about_period_months)
            
        for _ in period_months:
            num_work_days, num_skip_days = \
                self.go_end_trading_day(not self.is_start) if self.is_start else self.change_month()
            if num_work_days == 0: 
                break
            prev_work_date = None
            for day_idx in reversed(range(num_work_days)):
                curr_work_day = self.get_day_data(day_idx)
                self.num_work_days += 1
                if self.we_date is None:
                    self.we_date = parse(curr_work_day).date()
                self.ws_date = parse(curr_work_day).date()
                if self.ws_date == self.s_date:
                    self.is_stop = True
                    break
                if self.ws_date < self.s_date:
                    self.is_stop = True
                    os.remove(os.path.join(KRX_ORIGINAL_PATH, f"{curr_work_day}.csv"))
                    self.ws_date = prev_work_date
                    self.num_work_days -= 1
                    break
                prev_work_date = self.ws_date
            if str(self.ws_date)[:7] == str(self.s_date)[:7]:
                self.is_stop = True
            if self.is_stop:
                break
        
        log.info(f"Working Days: "
                 f"{self.ws_date} ~ {self.we_date}({self.num_work_days}) days")
        self.web_driver.quit()
        se_check.end()
                    
    def get_day_data(self, idx):
        self.web_driver.find_element(By.CLASS_NAME, "cal-btn-open").click()
        work_weeks = self.web_driver.find_element(By.CLASS_NAME, "cal-monthly-table").find_elements(By.CSS_SELECTOR, "tbody > tr")
        work_days = []
        for week in work_weeks:
            for day in week.find_elements(By.CSS_SELECTOR, "td > a"):
                work_days.append(day)
            
        work_days[idx].click()
        self.web_driver.find_element(By.ID, "jsSearchButton").click()
        wating_time = DOWNLOAD_WATING_TIME_AFTER_BASE_TIME if is_including_today() else DOWNLOAD_WATING_TIME_BEFORE_BASE_TIME
        time.sleep(wating_time)
        
        self.web_driver.find_element(By.XPATH, '//*[@id="MDCSTAT015_FORM"]/div[2]/div/p[2]/button[2]').click()
        self.web_driver.find_elements(By.CSS_SELECTOR, 'span.ico_filedown')[1].click()
        time.sleep(5)
        
        curr_work_day = self.web_driver.find_element(By.ID, "trdDd").get_attribute("value")
        source_file = os.path.join(KRX_DOWNLOAD_PATH, os.listdir(KRX_DOWNLOAD_PATH)[0])
        target_file = os.path.join(KRX_ORIGINAL_PATH, f"{curr_work_day}.csv")
        shutil.move(source_file, target_file)
        
        return curr_work_day                
                
    def go_end_trading_day(self, is_start=True):
        self.web_driver.find_element(By.CLASS_NAME, "cal-btn-open").click()
        if is_start:
            self.web_driver.find_element(By.CLASS_NAME, "cal-btn-prevM").click()
            
        first_day = self.get_start_trading_day_in_curr_month()
        cnt_diff_days = (self.e_date - first_day).days
        if cnt_diff_days < 0:
            if str(self.e_date)[:7] == str(first_day)[:7]:
                self.e_date = self.e_date.replace(day=1) - timedelta(days=1)
            num_work_days, num_skip_days = self.go_end_trading_day()
        else:
            self.web_driver.find_element(By.CLASS_NAME, "cal-btn-open").click()
            work_weeks = self.web_driver.find_element(By.CLASS_NAME, "cal-monthly-table") \
                                        .find_elements(By.CSS_SELECTOR, "tbody > tr")
            num_work_days, num_skip_days = 0, 0
            for work_week in work_weeks:
                for day in work_week.find_elements(By.CSS_SELECTOR, "td > a"):
                    work_day = int(day.text)
                    if work_day <= self.e_date.day:
                        num_work_days += 1
                        if str(self.s_date)[:7] == str(first_day)[:7] and work_day < self.s_date.day:
                            num_skip_days += 1
                    else:
                        break
            self.web_driver.find_element(By.CLASS_NAME, "cal-btn-open").click()
            
        self.is_start = False
        return num_work_days, num_skip_days
    
    def change_month(self):
        self.web_driver.find_element(By.CLASS_NAME, "cal-btn-open").click()
        self.web_driver.find_element(By.CLASS_NAME, "cal-btn-prevM").click()
        work_weeks = self.web_driver.find_element(By.CLASS_NAME, "cal-monthly-table") \
                                    .find_elements(By.CSS_SELECTOR, "tbody > tr")
        work_days = []
        for week in work_weeks:
            for day in week.find_elements(By.CSS_SELECTOR, "td > a"):
                work_days.append(day)
        work_days[0].click()
        return len(work_days), 0
    
    def get_start_trading_day_in_curr_month(self):
        try:
            self.web_driver.find_element(By.CLASS_NAME, "cal-monthly-table") \
                           .find_elements(By.CSS_SELECTOR, "tbody > tr")[0] \
                           .find_elements(By.CSS_SELECTOR, "td > a")[0].click()
        except:
            self.web_driver.find_element(By.CLASS_NAME, "cal-monthly-table") \
                           .find_elements(By.CSS_SELECTOR, "tbody > tr")[1] \
                           .find_elements(By.CSS_SELECTOR, "td > a")[0].click()
        first_day = parse(self.web_driver.find_element(By.ID, "trdDd").get_attribute("value")).date()
        return first_day
        
        
if __name__ == "__main__":
    _s_date = sys.argv[1] if len(sys.argv) > 1 else None
    _e_date = sys.argv[2] if len(sys.argv) > 2 else None
    krx_crawling = KrxDataCrawler(s_date=_s_date, e_date=_e_date)
    krx_crawling.start_crawling(is_display_processing_step=True)