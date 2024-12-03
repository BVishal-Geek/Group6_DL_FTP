import numpy as np
from tensorflow.keras.models import load_model
from keras.utils import to_categorical

import sys
sys.path.append("../../")

from components.utils import *
#%%
# ----- LOAD TEST IMAGES AND RESHAPE FOR MODELING -----
X_test = np.load('test_images.npy')
y_test = np.load('y_test.npy')
y_test = to_categorical(y_test, num_classes=2)
print('----------TEST IMAGES LOADED----------')
#%%
# ----- LOAD TRAINED MODEL AND GENERATE PREDICTIONS -----

model = load_model('finetuned_vgg16.h5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

loss, accuracy = model.evaluate(X_test, y_test)
print(loss) # 0.16
print(accuracy) # 0.94