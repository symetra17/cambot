# python3 code

import sys
import math
import cv2
import numpy as np
import glob
import time

fter_size = 100
line_thickness = 25
angle_max = 92
template = np.zeros((angle_max, fter_size,fter_size), dtype=np.float32)
mask = np.zeros((fter_size,fter_size), dtype=np.float32)
mask = cv2.circle(mask, (int(fter_size/2), int(fter_size/2)), int(fter_size/2), (1), cv2.FILLED)
diff_thd = 7.0    # the lower this value, the more sensitive

def init():
    
    for angle in range(angle_max):
        temp = np.zeros((fter_size,fter_size), dtype=np.float32)
        temp = cv2.circle(temp, (int(fter_size/2), int(fter_size/2)), 
                int(fter_size/2), 1, cv2.FILLED)
        
        temp2 = np.zeros((fter_size,fter_size),dtype=np.float32)
        temp2 = cv2.line(temp2, (0, 0), (fter_size,fter_size), 1, 
                thickness=line_thickness)
        
        rows, cols = temp2.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        temp2 = cv2.warpAffine(temp2,M,(cols, rows))
        
        temp3 = np.zeros((fter_size,fter_size),dtype=np.float32)
        temp3 = cv2.circle(temp3, (int(fter_size/2), int(fter_size/2)), 
                int(fter_size/2), 1, cv2.FILLED)
        temp2 = temp2 * temp3
        template[angle,:,:] = temp2

def estimate_angle(source):
    img = cv2.imread(source, 2)
    img = img.astype(np.float32)/256.0

    img_disp = cv2.imread(source, 2)
    img_disp = img_disp.astype(np.float32)/256.0

    peak_diff = 0
    peak_rot = -1
    peak_pos = -1

    t0=time.time()

    for m in range(0, img.shape[0]-fter_size, 4):

        focus = img[m:m+fter_size, 310:310+fter_size]
        result_tensor = np.multiply(focus, template)
        result_sum = result_tensor.sum(axis=(1,2))
        result = np.sqrt(result_sum)
        min_a = np.argmin(result)
        diff = result.max()-result.min()
        if diff > peak_diff:
            peak_diff = diff
            peak_rot = min_a
            peak_pos = m
    print('----', time.time()-t0)

    print('peak_diff %.1f'%peak_diff, '  estimated angle', peak_rot)
    if peak_diff > diff_thd:    # this value have to be learn by evaluating all training image
    
        tmp = np.zeros((fter_size,fter_size),dtype=np.float32)
        tmp = cv2.line(tmp, (0, 0), (fter_size,fter_size), 1, thickness=23)
        rows, cols = tmp.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), peak_rot, 1)
        tmp = cv2.warpAffine(tmp,M,(cols, rows))
    
        tmp3 = np.zeros((fter_size,fter_size),dtype=np.float32)
        tmp3 = cv2.circle(tmp3, (int(fter_size/2), int(fter_size/2)), 
                int(fter_size/2), (1), cv2.FILLED)
        tmp = tmp * tmp3
    
        img_disp[peak_pos:peak_pos+fter_size, 310:310+fter_size] = \
                img_disp[peak_pos:peak_pos+fter_size, 310:310+fter_size] + tmp
        cv2.imwrite(source[:-4] + '.JPG', img_disp*128)
    return peak_rot-45

if '__name__'=='__main__':
    init()
    files = glob.glob('*.bmp')
    t0=time.time()
    for fn in files:
        print(fn)
        estimate_angle(fn)    
    print(time.time()-t0)
