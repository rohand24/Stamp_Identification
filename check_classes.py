from pre_processing import glob2
import pdb
import pickle as pkl
path = '/shared/kgcoe-research/mil/stamp_stamp/data/row_column'


all_classes = []
with open('pre_processing/path_all_crop_stamps.pkl','r') as f:
	all_crop_stamps = pkl.load(f)

for i in xrange(len(all_crop_stamps)):
	img_name = all_crop_stamps[i]
	name = img_name.split('/')[-1]
	# pdb.set_trace()
	try:
		row_name = name.split('_')[0][1]
		column_name = name.split('_')[0][0]
		if ord(row_name) not in all_classes:
			all_classes.append(ord(row_name))
		if ord(column_name) not in all_classes:
			all_classes.append(ord(column_name))
		if ord(row_name)>97 or ord(column_name)>97:
			pdb.set_trace() 
	except:
		pass

str_classes = [str(unichr(i)) for i in sorted(all_classes)]
pdb.set_trace()