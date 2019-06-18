import numpy as np
import cv2
import motor
import time

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720);

for n in range(1):

    motor.forward(200)

    # Capture frame-by-frame
    ret, frame = cap.read()
    print frame.shape
    # Display the resulting frame
    #cv2.imshow('frame', frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
    cv2.imwrite("test.jpg", frame)
    motor.forward(-200)

    time.sleep(3)
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

