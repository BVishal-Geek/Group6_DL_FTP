#%%
import pandas as pd
import numpy as np
from keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D
import sys
sys.path.append("../")

from components.utils import *
#%%


y1 = pd.read_excel('../../data/Celeb-real-sample.xlsx')
y1 = y1['label']
y0 = pd.read_excel('../../data/Celeb-synthesis-sample.xlsx')
y0 = y0['label']

y = list(y1) + list(y0)

#%%
real, y_real = image_to_array(mapping='../../data/Celeb-real-sample.xlsx',
               image_fp='../../data/Celeb-real-sample/Frames',
               image_size=224,
                      split='train')

fake, y_fake = image_to_array(mapping='../../data/Celeb-synthesis-sample.xlsx',
               image_fp='../../data/Celeb-synthesis-sample/Frames',
               image_size=224,
                      split='train')

# Combine the arrays
frames = np.concatenate((real, fake), axis=0)
y = y_real + y_fake

#%%
# Create Train-Test Splits and Extract Features from Pretrained Model

X_train, X_test, y_train, y_test = process_input(X=frames, y=y, model='ResNet50', num_classes=2, input_type='train')
input_shape = int(X_train.shape[1])
#
#%%
# Define Classification Layers

model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(input_shape,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

#%%
# defining a function to save the weights of best model

from tensorflow.keras.callbacks import ModelCheckpoint
filepath=".weights.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=True,save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
# training the model
model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test), batch_size=16, callbacks=callbacks_list)
