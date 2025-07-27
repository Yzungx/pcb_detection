from selenium import webdriver

options = webdriver.ChromeOptions() 
options.add_argument("start-maximized")
options.add_argument("--auto-open-devtools-for-tabs")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
driver = webdriver.Chrome(options=options, executable_path=r'./chromedriver.exe')
driver.get("https://selenium.dev/documentation/en/")
print(driver.title)