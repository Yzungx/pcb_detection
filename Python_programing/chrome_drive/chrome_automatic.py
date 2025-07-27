from selenium import webdriver
import time
import easygui


# search = input('Enter the search word: ')
# print(search)
# vi thg py theo sync nen phai viet mess trc def

web = webdriver.Chrome('./chromedriver')

def drive():
	#popup
	easygui.msgbox("Để thực hiện train hãy làm theo hướng dẫn", title="How to use")
	easygui.msgbox("Đăng nhập vào Drive và tạo folder `Data`", title="How to use")
	easygui.msgbox("Upload file Data vao thu muc Data", title="How to use")
	#getlink
	# L = web.get("https://www.google.com/search?q=" + search + "&start" + str(i))
	driveLink = web.get("https://drive.google.com/drive/my-drive")
	out = easygui.buttonbox('Sau khi thuc hien xong cac buoc tren thi an Done', 'Check Drive', ('Done', ))
	#setimeout to connect collab
	if(out == 'Done'):
		n = 1
	else:
		n = 1000000
	time.sleep(n) #max time

def collab():
	easygui.msgbox("Để thực hiện train hãy làm theo hướng dẫn", title="How to use")
	easygui.msgbox("Đăng nhập vào Collab", title="How to use")
	
	collabLink = web.get("https://colab.research.google.com/drive/1Oc5g8BAqa95lOyHGxX6meLPXftZhnoWV#scrollTo=2-DFqWAGftXx")
	easygui.msgbox("F12> chuyen tab console.log", title="How to use")
	easygui.msgbox("Paste: //3 minute default interval:intervalInMilliSeconds =  180000//include good ol' jQueryvar script = document.createElement('script');script.src = 'https://code.jquery.com/jquery-3.4.1.min.js';document.getElementsByTagName('head')[0].appendChild(script);//init and first click. Replace :nth(0) with whichever button you want to click. 0 is the first play button.setTimeout(function(){$('colab-run-button:nth(0)').click()}, 1000)//intervalsetInterval(function(){ $('colab-run-button:nth(0)').click(); }, intervalInMilliSeconds)", title="How to use")
	easygui.msgbox("An Enter", title="How to use")
	out = easygui.buttonbox('xong thi an done', 'Check Collab', ('Done', ))
	if(out == 'Done'):
		n = 1
	else:
		n = 1000000
	time.sleep(n) #max time

#run
drive()

collab()
