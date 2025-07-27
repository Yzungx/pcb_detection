import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import Select
import time

#config----------------------------------------------------------------------------
email = 'traindata.irblab1@gmail.com' # replace email
password = 'dowaa001' # replace password

driver = uc.Chrome(use_subprocess=True)
wait = WebDriverWait(driver, 5)
#url-------------------------------------------------------------------------------
drive = 'https://drive.google.com/drive/my-drive'
colab = 'https://colab.research.google.com/drive/1ihOpdlBVE2Yt88gO2-EIDPrBzdkrqdzA?hl=en#scrollTo=4RQeksYqvufl'
driver.get(drive)
# func------------------------------------------------------------------------------
# ActionChains(driver).double_click(WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "//*[@id=':5']/div/c-wiz/div[2]/c-wiz/div[1]/c-wiz/div/c-wiz/div[1]/c-wiz/c-wiz/div/c-wiz[1]/div/div/div/div")))).perform()
def handlePopUp():

	WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.NAME, 'identifier'))).send_keys(f'{email}\n')
	WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.NAME, 'Passwd'))).send_keys(f'{password}\n')
	
	driver.maximize_window()
	driver.get(drive)
	time.sleep(10)
	driver.get(colab)
	time.sleep(5)
	# driver.find_element(By.TAG_NAME,"colab-toolbar-button").click()
	# time.sleep(10)
	time.sleep(1000000)
	
handlePopUp()