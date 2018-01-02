from PIL import Image
import os, sys
import cv2

def resize_all(path,h,w,dest_path):
	if(path[len(path)-1]!='/'):
		path = path+'/'
	dirs = os.listdir( path )
	count = 1
	if(dest_path[len(dest_path)-1]!='/'):
		dest_path = dest_path+'/'
	if not os.path.exists(dest_path):
		os.makedirs(dest_path)
	for file in dirs:
		try:
			img=cv2.imread(path+file)
			resized_image = cv2.resize(img,(h,w),90)
			cv2.imwrite(dest_path+str(count)+".jpg",resized_image)
			count=count+1
		except Exception as e:
			print(str(e))