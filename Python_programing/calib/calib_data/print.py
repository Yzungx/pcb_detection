from numpy import load
import cv2

data = load('Calib_data.npz')
print('a')
lst = data.files

camMatrix = data['camMatrix']
distCoef = data['distCoef']
rVector = data['rVector']
tVector =  data['tVector']
cam_id = data['cam']

print(camMatrix)
print('---------------------------------------------')
print(distCoef)
print('---------------------------------------------')
print(rVector)
print('---------------------------------------------')
print(tVector)
print('---------------------------------------------')
print(cam_id)
print('---------------------------------------------')
