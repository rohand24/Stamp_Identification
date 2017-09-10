import cv2, cv
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pdb
import glob2

# img_name = 'BD_466_081.JPG'
# img_name = 'AA_466_081.jpg'
img_name = 'AA_498_080.jpg'
path_to_data = '/shared/kgcoe-research/mil/stamp_stamp/data/QueenVictorian1864PennyRed/'
def cropout_stamp_section(img_name):
	# img_name = 'AF_575_171-001.jpg'
	img = cv2.imread(img_name)
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	gaussian_1 = cv2.GaussianBlur(img, (9,9), 10.0)
	img = cv2.addWeighted(img, 1.5, gaussian_1, -0.5, 0, img)

	lower_blue = np.array([110,100,150])
	upper_blue = np.array([130,255,255])
	kernel = np.ones((5,5), np.uint8)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, lower_blue, upper_blue)
	res = cv2.bitwise_and(img, img, mask= mask)
	res = cv2.dilate(res, kernel, iterations=20)

	###Crop img
	gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
	_,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
	contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	area = None
	for cont in contours:
		ar = cv2.contourArea(cont)
		if area == None:
			area = ar
			x,y,w,h = cv2.boundingRect(cont)
		else:
			if ar > area:
				area = ar
				x,y,w,h = cv2.boundingRect(cont)
	crop = img[y:y+h,x:x+w]

# jpg_imgs = glob2.glob(path_to_files+'/**/*.jpg')
# capital_jpg_imgs = glob2.glob(path_to_files+'/**/*.JPG')
# all_imgs = jpg_imgs + capital_jpg_imgs


# cv2.imwrite('res.jpg',cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
# cv2.imwrite('gray.jpg',gray)
	cv2.imwrite('crop_'+img_name,cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

cropout_stamp_section(img_name)









