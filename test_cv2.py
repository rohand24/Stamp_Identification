import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pdb


# img_name = 'BD_466_081.JPG'
img_name = 'AF_575_171-001.jpg'
img = cv2.imread(img_name)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)






lower_blue = np.array([110,100,150])
upper_blue = np.array([130,255,255])
# # lower_blue = np.array([110,70,60])
# # upper_blue = np.array([130,100,70])


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# hue = hsv[:, :, 0]
mask = cv2.inRange(hsv, lower_blue, upper_blue)
	
# pdb.set_trace()
res = cv2.bitwise_and(img, img, mask= mask)
###Crop img
# gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
# _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
# contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# cnt = contours[0]
# x,y,w,h = cv2.boundingRect(cnt)
# crop = res[y:y+h,x:x+w]

cv2.imwrite('output_process_stamp.jpg',cv2.cvtColor(res, cv2.COLOR_BGR2RGB))





# #get rid of very bright and very dark regions
# delta=30
# lower_gray = np.array([delta, delta,delta])
# upper_gray = np.array([255-delta,255-delta,255-delta])
# # pdb.set_trace()
# # Threshold the image to get only selected
# mask = cv2.inRange(img, lower_gray, upper_gray)
# # Bitwise-AND mask and original image
# res = cv2.bitwise_and(img,img, mask= mask)

# #Convert to HSV space
# HSV_img = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
# hue = HSV_img[:, :, 0]

# #select maximum value of H component from histogram
# hist = cv2.calcHist([hue],[0],None,[256],[0,256])
# hist= hist[1:, :] #suppress black value
# elem = np.argmax(hist)
# print np.max(hist), np.argmax(hist)

# tolerance=10
# lower_gray = np.array([elem-tolerance, 0,0])
# upper_gray = np.array([elem+tolerance,255,255])

# # lower_blue = np.array([110,50,50])
# # upper_blue = np.array([130,255,255])
# # Threshold the image to get only selected
# # mask = cv2.inRange(HSV_img, lower_gray, upper_gray)
# mask = cv2.inRange(HSV_img, lower_blue, upper_blue)
# # Bitwise-AND mask and original image
# res2 = cv2.bitwise_and(img,img, mask= mask)


# titles = ['Original Image', 'Selected Gray Values', 'Hue', 'Result']
# images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.cvtColor(res, cv2.COLOR_BGR2RGB), hue, cv2.cvtColor(res2, cv2.COLOR_BGR2RGB)]
# for i in xrange(4):
#     plt.subplot(2,2,i+1),plt.imshow(images[i])
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])

# plt.savefig('test_cv2.jpg')
# cv2.imwrite('cv2_'+img_name,res2)