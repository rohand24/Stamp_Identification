from PIL import Image
import pdb
import numpy as np

# path ='/shared/kgcoe-research/mil/stamp_stamp/QueenVictorian1864PennyRed/plate_81/AA\ to\ CL/'
img_name = 'BD_466_081.JPG'
#img_name = 'test.jpg'
im = Image.open(img_name)
# im = im.convert('RGBA')
data = np.array(im)[:,:,0]
# with open('red.out','w') as f:
# 	for d in data:
# 		np.savetxt(f,d, delimiter = ' ')	
# 		f.write('\n')
white = [255,255,255]

# color = [0, 144, 223]
r1, g1, b1 = 15, 112, 172
r2, g2, b2 = 70, 150, 255
# rgb = data[:,:,:]
red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
mask = (red>=r1)&(green>=g1)&(blue>=b1)&(red<=r2)&(green<=g2)&(blue<=b2)
pdb.set_trace()

data[:,:,:3][mask] = white

new_im = Image.fromarray(data)
new_im.save('test.jpg')

pdb.set_trace()