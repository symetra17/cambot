import cv2
import os

def cam_init():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720);
    ret, frame = cap.read()
    os.system('./my_init_webcam.sh')
    for n in range(10):
        ret, frame = cap.read()
    return cap
