from PIL import Image
import pdb
import numpy as np

# path ='/shared/kgcoe-research/mil/stamp_stamp/QueenVictorian1864PennyRed/plate_81/AA\ to\ CL/'
img_name = 'BD_466_081.JPG'
#img_name = 'test.jpg'
im = Image.open(img_name,)
# im = im.convert('RGBA')
# pdb.set_trace()
data = np.array(im)
white = [255,255,255]

# color = [0, 144, 223]
# r1, g1, b1 = 10, 112, 172
# r2, g2, b2 = 70, 150, 255
r1, g1, b1 = 40, 200, 255

# rgb = data[:,:,:]
red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
# mask = (red>=r1)&(green>=g1)&(blue>=b1)&(red<=r2)&(green<=g2)&(blue<=b2)
mask = (red<=r1)&(green<=g1)&(blue<=b1)&(green>=40)&(blue>=112)




data[:,:,:3][mask] = white

new_im = Image.fromarray(data)
new_im.save('test.jpg')

# pdb.set_trace()




