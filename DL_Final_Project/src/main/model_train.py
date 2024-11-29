#%%
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint

import sys
sys.path.append("../")

from components.utils import *
#%%
# ----- LOAD TRAIN IMAGES AND RESHAPE FOR MODELING -----

real, y_real = image_to_array(mapping='../../data/Celeb-real.xlsx',
               image_fp='../../data/Celeb-real/Frames',
               image_size=224,
                      split='train')

fake, y_fake = image_to_array(mapping='../../data/Celeb-synthesis.xlsx',
               image_fp='../../data/Celeb-synthesis/Frames',
               image_size=224,
                      split='train')

frames = np.concatenate((real, fake), axis=0)
y = y_real + y_fake

#%%
# ----- EXTRACT FEATURES FROM BASE MODEL -----

X_train, X_test, y_train, y_test = process_input(X=frames, y=y, model='ResNet50', num_classes=2, input_type='train')
input_shape = int(X_train.shape[1])


#%%
# ----- FINETUNE BASE MODEL AND SAVE WEIGHTS -----

model = CustomModel(input_shape=input_shape)
model.compile_model()

filepath=".weights.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=True,save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.train_model(X_train, y_train, epochs=1, validation_data=(X_test, y_test), batch_size=16, callbacks=callbacks_list)


