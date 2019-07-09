# this test created a 1 layer conv2d model with 90 feature filters, 
# each 100x100 in size

import os
import glob
import cv2
use_cpu = False
if use_cpu:   # it would use up all cores in a cpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import time

model = Sequential()

def init():
    fter_size = 100
    line_thickness = 25
    angle_max = 92
    template = np.zeros((fter_size,fter_size,1,angle_max), dtype=np.float32)
    
    mask = np.zeros((fter_size,fter_size), dtype=np.float32)
    mask = cv2.circle(mask, (int(fter_size/2), int(fter_size/2)), \
        int(fter_size/2), (1), cv2.FILLED)
    diff_thd = 7.0    # the lower this value, the more sensitive

    for angle in range(angle_max):
        
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
        template[:,:,0,angle] = temp2    
        
        #cv2.imwrite('template/%d.jpg'%angle, 256*template[:,:,0,angle])
        
    t0=time.time()
    
    
    model.add(Conv2D(angle_max, (fter_size, fter_size), \
        activation='linear', input_shape=(720, 100, 1)))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    
    biases = np.zeros((angle_max))
    model.layers[0].set_weights([template, biases])
    
    print(int(time.time()-t0),'sec')

def detect(inp_file):
    t0=time.time()
    x_test = cv2.imread(inp_file, 2)[:, 360-50:360+50]
    x_test = x_test.astype(np.float32)/256
    x_test = x_test.reshape((1,720,100,1))
    
    res = model.predict(x_test)   # output shape (1, 621, 1, 92)
    
    result = res.reshape((res.shape[1], res.shape[3]))
    result = np.sqrt(result)    # shape (621,92)
    #print(result)
    #print('result shape ', result.shape)
    
    #print('max heat ', np.amax(result, axis=1))
    #print('min heat ', np.amin(result, axis=1))
    
    diff = np.amax(result, axis=1) - np.amin(result, axis=1)
    
    #print('max diff ', diff.max())
    est_pos = np.argmax(diff)
    print('detected pos ', est_pos)
    
    print('est angle ', np.argmin(result[est_pos][:]) -45 )
    print(int(1000*(time.time()-t0)),'ms')

    img = cv2.imread(inp_file)
    img[est_pos+50, 360-50:360+50] = 255
    img[est_pos:est_pos+100, 360] = 255
    out_file = inp_file[:-3] + 'JPG'
    
    cv2.imwrite(out_file, img)
    
if __name__=='__main__':
    init()
    for fn in glob.glob('*.bmp'):
        detect(fn)