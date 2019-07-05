# python3 code

import sys
import math
import cv2
import numpy as np
import glob
import time

line_thickness = 25
angle_max = 100
template = np.zeros((angle_max, 100,100), dtype=np.float32)
mask = np.zeros((100,100), dtype=np.float32)
mask = cv2.circle(mask, (50, 50), 50, (1), cv2.FILLED)

def generate_template():
    
    for angle in range(angle_max):
        temp = np.zeros((100,100), dtype=np.float32)
        temp = cv2.circle(temp, (50, 50), 50, (1), cv2.FILLED)
        
        temp2 = np.zeros((100,100),dtype=np.float32)
        temp2 = cv2.line(temp2, (0, 0), (100,100), 1, thickness=line_thickness)
        
        rows, cols = temp2.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        temp2 = cv2.warpAffine(temp2,M,(cols, rows))
        
        temp3 = np.zeros((100,100),dtype=np.float32)
        temp3 = cv2.circle(temp3, (50, 50), 50, (1), cv2.FILLED)
        temp2 = temp2 * temp3
        #cv2.imshow('%d'%angle,temp2)
        template[angle,:,:] = temp2

def the_function(source):
    img = cv2.imread(source, 2)
    img = img.astype(np.float32)/256.0

    img_disp = cv2.imread(source, 2)
    img_disp = img_disp.astype(np.float32)/256.0

    result2 = np.zeros((600))

    peak_diff = 0
    peak_rot = -1
    peak_pos = -1

    for m in range(0, 600, 2):

        angle = 0
        result = np.zeros((angle_max))
        for angle in range(angle_max):
            focus = img[m:m+100, 300:400]
            val = (focus * mask * template[angle,:,:]).sum()
            result[angle] = np.sqrt(val)
       
        min_a = np.argmin(result)
        diff = result.max()-result.min()
        if diff > peak_diff:
            peak_diff = diff
            peak_rot = min_a
            peak_pos = m
        
    print('peak_diff ', peak_diff, '  estimated angle', peak_rot)
    if peak_diff > 10.0:    # this value have to be learn by evaluating all training image
    
        tmp = np.zeros((100,100),dtype=np.float32)
        tmp = cv2.line(tmp, (0, 0), (100,100), 1, thickness=23)
        rows, cols = tmp.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), peak_rot, 1)
        tmp = cv2.warpAffine(tmp,M,(cols, rows))
    
        tmp3 = np.zeros((100,100),dtype=np.float32)
        tmp3 = cv2.circle(tmp3, (50, 50), 50, (1), cv2.FILLED)
        tmp = tmp * tmp3
    
        img_disp[peak_pos:peak_pos+100, 300:400] = img_disp[peak_pos:peak_pos+100, 300:400] + tmp
        #cv2.imshow('', img_disp)
        #cv2.waitKey(0)
        cv2.imwrite(source[:-4]+'%03d'%(peak_rot-45)+'.JPG', img_disp*128)

print('generate template')
generate_template()
print('detecting')

files = glob.glob('/home/ins/line_image/*.jpg')
t0=time.time()
for fn in files:
    print(fn)
    the_function(fn)
    
print(time.time()-t0)
