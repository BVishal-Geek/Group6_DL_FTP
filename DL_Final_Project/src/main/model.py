#%%
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

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

real = image_to_array(mapping='../../data/Celeb-real-sample.xlsx',
               image_fp='../../data/Celeb-real-sample/Frames',
               image_size=224)
print(f'Shape of real: {real.shape}')
fake = image_to_array(mapping='../../data/Celeb-synthesis-sample.xlsx',
               image_fp='../../data/Celeb-synthesis-sample/Frames',
               image_size=224)
print(f'Shape of fake: {fake.shape}')
# Combine the arrays
frames = np.concatenate((real, fake), axis=0)
# creating the training and validation set
print(f'Shape of frames: {frames.shape}')

X_train, X_test, y_train, y_test = train_test_split(frames, y, random_state=6303, test_size=0.2, stratify = y)
# creating the base model of pre-trained VGG16 model

base_model = VGG16(weights='imagenet', include_top=False)

# extracting features for training frames

X_train = base_model.predict(X_train)
print(f'X Train shape: {X_train.shape}')

# extracting features for validation frames

X_test = base_model.predict(X_test)
print(f'X Test shape: {X_test.shape}')

#%%
X_train = X_train.reshape(80, 7*7*512)
X_test = X_test.reshape(20, 7*7*512)

#%%
model = Sequential()

model.add(Dense(1024, activation='relu', input_shape=(25088,)))

model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(101, activation='softmax'))

#%%
# defining a function to save the weights of best model

from keras.callbacks import ModelCheckpoint

mcp_save = ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')
# compiling the model

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

# training the model
model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test), callbacks=[mcp_save], batch_size=128)
