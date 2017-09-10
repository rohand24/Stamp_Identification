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

img_name = '224_PL_2772.jpg'



img = cv2.imread(img_name)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# edges = cv2.Canny(img,50,200)
# gaussian_1 = cv2.GaussianBlur(img, (9,9), 10.0)
# img = cv2.addWeighted(img, 1.5, gaussian_1, -0.5, 0, img)
# cv2.imwrite('edge_' + img_name, edges)    
    
# lower_blue = np.array([110,100,150])
# upper_blue = np.array([130,255,255])

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
hist_scale = 10
hist = np.clip(hist*0.005*hist_scale, 0, 1)
vis = hsv_map*h[:,:,np.newaxis] / 255.0

cv2.imwrite('hist' + img_name,vis)    
# lower mask (0-10)
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(hsv, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(hsv, lower_red, upper_red)

pdb.set_trace()

kernel = np.ones((2,2), np.uint8)
mask = cv2.inRange(hsv, lower_red, upper_red)
res = cv2.bitwise_and(img, img, mask= mask)

cv2.imwrite('mask' + img_name,cv2.cvtColor(res, cv2.COLOR_BGR2RGB))    

# res = cv2.erode(res, kernel, iterations=1) 
# res = cv2.dilate(res, kernel, iterations=1) 
res = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


cv2.imwrite('sharpen_' + img_name,cv2.cvtColor(res, cv2.COLOR_BGR2RGB))    