import cv2, cv
import numpy as np
import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt
import pdb, os
import glob2
import pickle as pkl

# img_name = 'BD_466_081.JPG'
# img_name = 'AA_466_081.jpg'
path_to_data = '/shared/kgcoe-research/mil/stamp_stamp/data/QueenVictorian1864PennyRed'
path_to_crop_images = '/shared/kgcoe-research/mil/stamp_stamp/data/cropout_stamp'
def cropout_stamp_section(path_img, img_name, path_to_save):
	# img_name = 'AF_575_171-001.jpg'
	img = cv2.imread(path_img)
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	gaussian_1 = cv2.GaussianBlur(img, (9,9), 10.0)
	img = cv2.addWeighted(img, 1.5, gaussian_1, -0.5, 0, img)

	lower_blue = np.array([110,100,150])
	upper_blue = np.array([130,255,255])
	kernel = np.ones((5,5), np.uint8)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, lower_blue, upper_blue)
	res = cv2.bitwise_and(img, img, mask= mask)
	res = cv2.dilate(res, kernel, iterations=10)

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
	cv2.imwrite(path_to_save + img_name,cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
# jpg_imgs = glob2.glob(path_to_data+'/**/*.jpg')
# capital_jpg_imgs = glob2.glob(path_to_data+'/**/*.JPG')
# all_imgs = jpg_imgs + capital_jpg_imgs
# with open('path_all_stamps.pkl','w') as f:
# 	pkl.dump(all_imgs, f)
with open('path_all_stamps.pkl','r') as f:
	all_imgs = pkl.load(f)
count = 0
for i in xrange(len(all_imgs)):
	extract_plate = all_imgs[i].split('/')[-3].replace('_',' ').split(' ')[0]
	
	img_name = all_imgs[i].split('/')[-1].split(' ')[0] + '_'+str(count)+'.jpg'
	path_to_new_crop = path_to_crop_images + '/' + extract_plate + '/'
	if not os.path.exists(path_to_new_crop):
	    os.makedirs(path_to_new_crop)
	path_to_img = all_imgs[i]
	cropout_stamp_section(path_to_img, img_name, path_to_new_crop)
	print "Process image:", count 
	count +=1

# pdb.set_trace()
# cv2.imwrite('res.jpg',cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
# cv2.imwrite('gray.jpg',gray)
# cv2.imwrite('crop_'+img_name,cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))








