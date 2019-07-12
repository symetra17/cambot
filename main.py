# This is a python3 code

import numpy as np
import cv2
import time
import os
import my_cam
import motor

from ctypes import *
import math
import random
import pprint
import time
import glob
#import line_detect_keras

#line_detect_keras.init()

base_dir = '/home/nvidia/my_yolo/'

  
def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),      # channel
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    
lib = CDLL("/home/nvidia/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

p0 = glob.glob(base_dir + 'testimg/yellow/*.jpg')[0]
im_yolo = load_image(p0.encode('utf-8') , 0, 0)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

import ctypes

def yolo_image2npy(im_yo):
    npy = np.ctypeslib.as_array(im_yo, shape=(800*800*3,))
    npy = np.reshape(npy, (3,800,800))
    npy = np.transpose(npy, (1,2,0))
    return npy

def detect(net, meta, image, thresh=.4, hier_thresh=.5, nms=.45):

    t0 = time.time()

    xxx = np.transpose(image, (2,0,1))
    xxx = xxx.flatten()
    numpy_ptr =  xxx.ctypes.data_as(POINTER(c_float))
    
    im_yolo.data=numpy_ptr
    
    #print ' load...', int(1000.0*(time.time()-t0)), 'ms'
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im_yolo)

    dets = get_network_boxes(net, im_yolo.w, im_yolo.h, thresh, 
            hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);
    
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    #free_image(im_yolo)
    
    #im_yolo.data = POINTER(c_float)()
    
    free_detections(dets, num)
    print("detect ", int(1000.0*(time.time()-t0)), 'ms')
    
    binary_output = False
    x = 0
    y_nearest = 0
    
    print("num of item: ", len(res))
    
    if len(res) > 0:
      for item in res:
        label = item[0]
        confi = item[1]
        print(label, '  confi %.2f'%confi)
        loc   = item[2]
        x = int(loc[0])
        y = int(loc[1])
        w = int(loc[2])
        h = int(loc[3])
        if label == b'bottle':
            y_nearest = y + int(h/2)
            binary_output = True
            cv2.rectangle(image, 
                    (x-int(w/2), y-int(h/2)), (x+int(w/2), y+int(h/2)), 
                    color=(1, 1, 0), thickness=2)

    cv2.imshow("", image)

    return binary_output, x, y_nearest

def motor_op(r,x,y):
    post_step = 300
    
    if r and y > 500:
        if 0 < x <= 200:
            print('small cw')
            motor.turn(40)
            motor.forward(200)
            motor.turn(-40)
            motor.forward(post_step)
        
        elif 200 < x <= 400:
            print('cw')
            motor.turn(45)
            motor.forward(350)
            motor.turn(-45)
            motor.forward(post_step)
            
        elif 400 < x <= 600:
            print('ccw')
            motor.turn(45)
            motor.forward(350)
            motor.turn(-45)
            motor.forward(post_step)
        else:    # >600
            print('small ccw')
            motor.turn(-40)
            motor.forward(200)
            motor.turn(40)
            motor.forward(post_step)
            
    elif r and y > 200:
        print("small forward")
        motor.forward(200)
    else:
        print("forward")
        motor.forward(300)
    
if __name__ == "__main__":

    cam = my_cam.cam_init()
    exceed = int((1280-720)/2)

    #motor.reset_device()
    #motor.forward(20)
    
    net = None
    net = load_net((base_dir + "my_yolov3-tiny.cfg").encode('utf-8'), 
                   (base_dir + "weights/my_yolov3-tiny_10000.weights").encode('utf-8'), 0)
    meta = load_meta((base_dir + "darknet.data").encode('utf-8'))

    for n in range(1000):
        for h in range(6):
            ret, img = cam.read()
        ret, img = cam.read()
        img = img[:, 0+exceed:720+exceed]
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.resize(img, (800,800))
        img = img.astype(np.float32)/256.0
        r, x, y = detect(net, meta, img)
        print('x ', x, '   y ', y)
        
        
        key = cv2.waitKey(100)
        if key&0xff==ord('q'):
            quit()
            
        # y is 220 for 400mm from robot tip
        
        #motor_op(r,x,y)
        
        time.sleep(1)
        
        key = cv2.waitKey(0)
        if key&0xff==ord('q'):
            quit()
            
    quit()
    
    
    for n in range(500):
        ret, im = cam.read()
        im = im[:, 0+exceed:720+exceed]        
        im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)        
        cv2.imshow("", im)
        cv2.waitKey(50)
    quit()


#for n in range(1):
#
#    motor.forward(200)
#
#    # Capture frame-by-frame
#    ret, frame = cap.read()
#    frame = frame[:, 0+exceed:720+exceed]
#    print frame.shape
#    # Display the resulting frame
#    #cv2.imshow('frame', frame)
#    #if cv2.waitKey(1) & 0xFF == ord('q'):
#    #    break
#    cv2.imwrite("test.jpg", frame)
#    motor.forward(-200)
#
#    time.sleep(3)
#
## When everything done, release the capture
#cap.release()
#cv2.destroyAllWindows()

