#import func------------------------------------------------
from upload import upload
from createFolder import createFolder
#-----------------------------------------------------------
from googleapiclient.http import MediaFileUpload
import os
import undetected_chromedriver.v2 as uc2
from selenium.webdriver.common.keys import Keys
# import undetected_chromedriver as ucp
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import Select
import time
import easygui
#config
upload = upload()
createFolder = createFolder()
#config load image name
path = "./chrome_drive/data_test"
files = os.listdir(path)
#config account-----------------------------------------------
email = 'traindata.irblab1@gmail.com' # replace email
password = 'dowaa001' # replace password
#url-------------------------------------------------------------------------------
drive = 'https://drive.google.com/drive/my-drive'
colab = 'https://colab.research.google.com/drive/1ihOpdlBVE2Yt88gO2-EIDPrBzdkrqdzA?hl=en#scrollTo=4RQeksYqvufl'

#func-------------------------------------------------------
#create folder and get id folder 
def createFolderData():
	file_metadata = {
	'name': 'yolov7_train',
	'mimeType': 'application/vnd.google-apps.folder'
	}
        # pylint: disable=maybe-no-member
	file = createFolder.files().create(body=file_metadata, fields='id').execute()
	print(F'Folder ID: "{file.get("id")}".')
	return file.get('id')
#-----------------------------------------------------------
#load image name
def loadImgName():
	path = []
	for file in files:
		# a = ['.jpg', '.png', 'jpeg', '.txt']
		if file.endswith(('.zip')):
			path.append(file)
			print(path)
	return path
#upload
def updata(idFolder):
	easygui.msgbox("Đang upload Data\nAn OK va doi", title="alert")
	folder_id = [idFolder]
	file_names = loadImgName()  
	print(file_names)
	for file_name in file_names:
		file_metadata = {
		"name": file_name,
		"parents": folder_id
		}
		#path img
		media = MediaFileUpload("./chrome_drive/data_test/{0}".format(file_name), resumable=True)
		send = upload.files().create(body = file_metadata,media_body = media,fields = "id").execute()
		print('file id: %s' %send.get('id'))

def handlePopUp():
	driver = uc2.Chrome()#use_subprocess=True
	wait = WebDriverWait(driver, 5)

	driver.maximize_window()
	driver.get(drive)
	time.sleep(10)

	WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.NAME, 'identifier'))).send_keys(f'{email}\n')
	WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.NAME, 'Passwd'))).send_keys(f'{password}\n')
	
	time.sleep(10)
	driver.get(colab)
	time.sleep(5)
	# Find the connection button and click it
	parentWrapper = driver.find_element(By.CSS_SELECTOR, "colab-connect-button").click()
	# buttonShadow = driver.find_element(By.TAG_NAME, "colab-toolbar-button").text
	time.sleep(20)
	print(parentWrapper)
	# press runtime all
	# runTime = driver.find_element(By.XPATH, "(//div[@class='goog-inline-block goog-menu-button-inner-box'])[5]").click()
	# # //div[@id='runtime-menu-button']//div[@class='goog-inline-block goog-menu-button-inner-box']
	ActionChains(driver)\
		.key_down(Keys.CONTROL)\
		.send_keys(Keys.F9)\
		.key_up(Keys.CONTROL)\
		.perform()
	time.sleep(40)
# -----------------------------------------------------------------------
# khong auto connect vi vs moi acc sẽ generate random div
	# # accept login
	# runTime = driver.find_element(By.XPATH, "(//mwc-button[@slot='primaryAction'])[1]").click()
	# time.sleep(10)
	# # switch window an handle
	# driver.switch_to.window(driver.window_handles[1])
	# time.sleep(10)
	# # 
# -----------------------------------------------------------------------
	Note()
	time.sleep(1000000)
def Note():
	easygui.msgbox("Để thực hiện train hãy làm theo hướng dẫn\n1. Nhấn kết nôi\n2. Nhấn run từng block\n3. Đợi khi hoàn thành train thì quay lại giao diện ấn tải Weight", title="How to use")
def dowloadWeight():
	linkFolderTrain = 'https://drive.google.com/drive'
	driver = uc2.Chrome()#use_subprocess=True
	wait = WebDriverWait(driver, 5)

	driver.maximize_window()
	driver.get(drive)
	WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.NAME, 'identifier'))).send_keys(f'{email}\n')
	WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.NAME, 'Passwd'))).send_keys(f'{password}\n')
	time.sleep(15)
	# easygui.msgbox("1. Tìm thư mục yolov7_train\n2. Tìm thư mục yolov7\n3. Tìm thư mục Run và vào mục 'weight' dowload weight phù hợp", title="How to use")
	# search box click
	searchBox = driver.find_element(By.XPATH, "//input[@placeholder='Tìm trong Drive']").click()
	# press weight in keyboard
	ActionChains(driver)\
		.send_keys("weight")\
		.send_keys(Keys.ENTER)\
		.perform()
	time.sleep(5)
	easygui.msgbox("Chọn weight cần tải", title="alert")
	# wait
	time.sleep(1000000)

#-----------------------------------------------------------
#run
out = easygui.buttonbox('Lựa chọn hành động', 'Select', ('Train', 'Dowload Weight'))

if(out == 'Train'):
	idFolder = createFolderData()
	updata(idFolder)
	handlePopUp()
elif(out == 'Dowload Weight'):
	dowloadWeight()
