import sys
import math
import cv2
import numpy as np

img = cv2.imread('/home/ins/im_0018.jpg', 2)
img = img.astype(np.float32)/256.0

angle = 0
result = np.zeros((180))
for angle in range(180):
    focus = img[500-5:600-5, 300:400]
    
    temp = np.zeros((100,100), dtype=np.float32)
    temp = cv2.circle(temp, (50, 50), 50, (1), cv2.FILLED)
    
    temp2 = np.zeros((100,100),dtype=np.float32)
    temp2 = cv2.line(temp2, (0, 0), (100,100), 1, thickness=23)
    
    rows, cols = temp2.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    temp2 = cv2.warpAffine(temp2,M,(cols, rows))
    
    temp3 = np.zeros((100,100),dtype=np.float32)
    temp3 = cv2.circle(temp3, (50, 50), 50, (1), cv2.FILLED)
    temp2 = temp2 * temp3
    
    val = (focus*temp*temp2).mean()
    result[angle] = np.sqrt(val)
    
    #cv2.imshow('temp2', temp2)
    #cv2.imshow('wtf', (focus*temp2))
    #cv2.waitKey(0)

min_a = np.argmin(result)
print(min_a, 'degree is minimum')
print("min",result.min(),'  max',result.max(), '  diff', result.max()-result.min())

angle = min_a

temp = np.zeros((100,100),dtype=np.float32)
temp = cv2.circle(temp, (50, 50), 50, (1), cv2.FILLED)
focus = focus * temp

temp2 = np.zeros((100,100),dtype=np.float32)
temp2 = cv2.line(temp2, (0, 0), (100,100), 1, thickness=23)

rows, cols = temp2.shape
M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
temp2 = cv2.warpAffine(temp2,M,(cols, rows))

temp3 = np.zeros((100,100),dtype=np.float32)
temp3 = cv2.circle(temp3, (50, 50), 50, (1), cv2.FILLED)
temp2 = temp2 * temp3

cv2.imshow('', focus*temp2)
cv2.waitKey(0)
