import undetected_chromedriver.v2 as uc
from selenium.webdriver.support.ui import WebDriverWait
from time import sleep
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time

email = 'traindata.irblab1@gmail.com' # replace email
password = 'dowaa001' # replace password

driver = uc.Chrome()
wait = WebDriverWait(driver, 5)

driver.delete_all_cookies()

driver.get('https://drive.google.com/drive/my-drive')
sleep(2)


WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.NAME, 'identifier'))).send_keys(f'{email}\n')
WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.NAME, 'Passwd'))).send_keys(f'{password}\n')
time.sleep(15)

driver.get('https://gmail.com')
sleep(2)