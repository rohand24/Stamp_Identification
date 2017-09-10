import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import PIL
from PIL import Image
import glob2
import pdb
import pickle as pkl


# path_to_data = '/shared/kgcoe-research/mil/stamp_stamp/data/cropout_stamp'
# jpg_imgs = glob2.glob(path_to_data+'/**/*.jpg')
# with open('path_all_crop_stamps.pkl','w') as f:
	# pkl.dump(jpg_imgs, f)
    
path_to_row_column = '/shared/kgcoe-research/mil/stamp_stamp/data/row_column'
path_to_plate = '/shared/kgcoe-research/mil/stamp_stamp/data/plate'
with open('path_all_crop_stamps.pkl','r') as f:
	all_crop_stamps = pkl.load(f)

def check_path(path):
    if not os.path.exists(path):
	    os.makedirs(path)

def extract_rc_plate(img_name, iteration):
    img = cv2.imread(img_name)
    name = img_name.split('/')[-2:]
    plate = str(int(name[0]))
    try:
        row_name = name[1].split('_')[0][1].upper()
        column_name = name[1].split('_')[0][0].upper()
    except:
        return
    if ord(row_name)>90 or ord(row_name)<65 or ord(column_name)>90 or ord(column_name)<65:
        return
    # if ord(column_name)>90 or ord(column_name)<65:
        # return
    # pdb.set_trace()
    #Convert BGR image to RGB
    # img= images[i]
    b,g,r = cv2.split(img)
    img2 = cv2.merge([r,g,b])

    # # Crop Corners----->PLATE Row and Column of the Stamp
    # m = 0.16*img2.shape[1]      #Left part
    # n=0.84*img2.shape[1]        #Right part
    # p=0.13*img2.shape[0]        #Top part
    # q=0.87*img2.shape[0]        #Bottom part

    # Crop Corners----->PLATE Row and Column of the Stamp
    m = 0.16*img2.shape[1]      #Left part
    n=0.84*img2.shape[1]        #Right part
    p=0.26*img2.shape[0]        #Top part
    q=0.74*img2.shape[0]        #Bottom part

    img_crop1 = img2[0:p,0:m ,:]
    img_crop2 = img2[0:p,n:,:]
    img_crop3 = img2[q:, 0:m,:]
    img_crop4 = img2[q:, n:, :]

    #Resize Corners
    dim = (112, 224)
    img_crop1 = cv2.resize(img_crop1, dim, interpolation=cv2.INTER_AREA)
    img_crop2 = cv2.resize(img_crop2, dim, interpolation=cv2.INTER_AREA)
    img_crop3 = cv2.resize(img_crop3, dim, interpolation=cv2.INTER_AREA)
    img_crop4 = cv2.resize(img_crop4, dim, interpolation=cv2.INTER_AREA)

    #Combine to get a block of 4 Corners    [1,2];[3,4]
    column1 = np.concatenate((img_crop1,img_crop3),axis=0)
    column2 = np.concatenate((img_crop2, img_crop4), axis=0)
    block = np.concatenate((column1,column2),axis=1)

    #Combine to get Same element block
    Row = np.concatenate((img_crop1,img_crop4),axis=1)
    Column= np.concatenate((img_crop3, img_crop2), axis=1)

    # Crop Side-Middle portion----->PLATE number of the Stamp
    width_plate_left = 0.24*img2.shape[1] 
    width_plate_right = 0.86*img2.shape[1] 
    x= 0.35*img2.shape[0]       #Middle Top
    y= 0.65*img2.shape[0]       #Middle Bottom

    img_crop5 = img2[x:y,0:width_plate_left,:]
    img_crop6 = img2[x:y,width_plate_right:,:]

    #Resize
    dim = (135, 224)
    img_crop5= cv2.resize(img_crop5,dim,interpolation=cv2.INTER_AREA)
    dim = (89, 224)
    
    img_crop6 = cv2.resize(img_crop6, dim, interpolation=cv2.INTER_AREA)

    #Combine to get Final ouptut--->PLATE number
    visual = np.concatenate((img_crop5,img_crop6),axis=1)
    # path_to_save_plate = path_to_plate + 
    check_path(path_to_row_column)
    check_path(path_to_plate)
    # try:
    cv2.imwrite(path_to_row_column+'/' + row_name + '_r_' + plate + '_' + name[1].split('.')[0] + '.jpg',cv2.cvtColor(Row, cv2.COLOR_BGR2RGB))
    cv2.imwrite(path_to_row_column+'/' + column_name + '_r_' + plate + '_' + name[1].split('.')[0] + '.jpg',cv2.cvtColor(Column, cv2.COLOR_BGR2RGB))    
    cv2.imwrite(path_to_plate+'/' + plate + '_'+name[1].split('.')[0]+'.jpg',cv2.cvtColor(visual, cv2.COLOR_BGR2RGB))
    # except:
        # pdb.set_trace()
for i in xrange(len(all_crop_stamps)):
    extract_rc_plate(all_crop_stamps[i],i)
    print "Iteration: ", i
pdb.set_trace()    