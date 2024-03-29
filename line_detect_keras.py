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

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

model = Sequential()
template = None
diff_thd = 7.0    # the lower this value, the more sensitive

def init():

    global template

    fter_size = 100
    line_thickness = 23
    angle_max = 92
    template = np.zeros((fter_size,fter_size,1,angle_max), dtype=np.float32)
    
    mask = np.zeros((fter_size,fter_size), dtype=np.float32)
    mask = cv2.circle(mask, (int(fter_size/2), int(fter_size/2)), \
        int(fter_size/2), (1), cv2.FILLED)

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
    #model.save('mynet')
    print(int(time.time()-t0),'sec')

def detect(inp_file):

    t0 = time.time()
    
    if type(inp_file)==str:
        x_test = cv2.imread(inp_file, 2)[:, 360-50:360+50]
        x_test = x_test.astype(np.float32)/256
    elif type(inp_file)==np.ndarray:
        x_test = inp_file
    else:
        print('unsupported data type')

    x_test = x_test.reshape((1,720,100,1))

    res = model.predict(x_test)   # output shape (1, 621, 1, 92)    
    result = res.reshape((res.shape[1], res.shape[3]))
    result = np.sqrt(result)    # shape (621,92)    
    diff = np.amax(result, axis=1) - np.amin(result, axis=1)
    print(diff.max())
    if diff.max() < diff_thd:
        return []

    est_pos = np.argmax(diff)
    est_angle = np.argmin(result[est_pos][:])
    
    print('pos', est_pos, ' angle',  est_angle-45)
    print(int(1000*(time.time()-t0)),'ms')

    if type(inp_file)==str:
        img = cv2.imread(inp_file, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img = img/512    
        img[est_pos:est_pos+100, 360-50:360+50] += template[:,:,0,est_angle]
        out_file = inp_file[:-3] + 'JPG'    
        cv2.imwrite(out_file, img*256)

    return [(est_pos, est_angle)]

if __name__=='__main__':
    init()
    #img = np.zeros((720,100))
    #detect(img)
    for fn in glob.glob('samples/*.bmp'):
        detect(fn)
        input('press enter')