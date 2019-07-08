# this test created a 1 layer conv2d model with 90 feature filters, 
# each 100x100 in size

import os
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
model.add(Conv2D(90, (100,100), activation='linear', \
    input_shape=(720, 720, 1)))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

www = 3*np.ones((100, 100, 1, 90))
bbb = np.zeros((90))
model.layers[0].set_weights([www, bbb])


for n in range(100):
    t0=time.time()
    x_test = 2*np.ones((1, 720, 720, 1))
    score = model.predict(x_test)
    print(score.shape)
    print(time.time()-t0)
