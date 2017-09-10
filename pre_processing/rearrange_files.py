import pdb
import numpy as np 
import glob2
import re
import os,sys
path_to_files = '/shared/kgcoe-research/mil/stamp_stamp/data/QueenVictorian1864PennyRed/mixing'
rearrange = '/shared/kgcoe-research/mil/stamp_stamp/data/QueenVictorian1864PennyRed/rearrange'


jpg_imgs = glob2.glob(path_to_files+'/**/*.jpg')
capital_jpg_imgs = glob2.glob(path_to_files+'/**/*.JPG')
all_imgs = jpg_imgs + capital_jpg_imgs

# pdb.set_trace()
arrange = []
outliers = []
counter = 0
for i in xrange(len(all_imgs)):
	path_to_file = all_imgs[i]
	file_name = path_to_file.replace('.',' ').replace('-',' ').split('/')[-1]
	file_name_split = file_name.split(' ')
	add = 0
	len_plate = []
	for j in xrange(len(file_name_split)):
		try:
			t = int(file_name_split[j])
			if 69<t<80:
				len_plate.append(t)
		except:
			pass
	if len(len_plate):
		arrange.append(all_imgs[i])				
		new_path = rearrange + '/' + str(len_plate[-1]) + '/all/'
		if not os.path.exists(new_path):
		    os.makedirs(new_path)
		str_to_call = 'cp "' + all_imgs[i] + '" ' + new_path		
		os.system(str_to_call)
		add =1
		counter+=1
	if add == 0:
		outliers.append(all_imgs[i])

pdb.set_trace()



