import cv2
import os
import time

def cam_init():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720);
    ret, frame = cap.read()
    os.system('./my_init_webcam.sh')
    for n in range(10):
        ret, frame = cap.read()
    return cap

if __name__== '__main__':
    cap=cam_init()
    n=0
    while True:
      for m in range(6):
          cap.read()
      ret, im=cap.read()
      im=cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
      im=im[280:-280,:,:]
      cv2.imshow('opencv imshow', im)
      kee = cv2.waitKey(10)
      if kee==ord('q'):
          quit()
      elif kee==ord(' '):
          print('write %d'%n)
          cv2.imwrite('im_%04d.bmp'%n, im)
          n=n+1
      time.sleep(0.5)

