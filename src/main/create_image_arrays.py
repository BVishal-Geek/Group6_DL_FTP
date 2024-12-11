#%%
import numpy as np
import sys
sys.path.append("../")

from components.utils import *

#%%
# ----- LOAD TRAIN IMAGES AND RESHAPE FOR MODELING -----
print('----------LOADING TRAIN IMAGES----------')

real, y_real = image_to_array(mapping='../../data/Celeb-real.xlsx',
               image_fp='../../data/Celeb-real/Frames',
               image_size=224,
                      split='train')

fake, y_fake = image_to_array(mapping='../../data/Celeb-synthesis.xlsx',
               image_fp='../../data/Celeb-synthesis/Frames',
               image_size=224,
                      split='train')

X_train = np.concatenate((real, fake), axis=0)
y_train = y_real + y_fake

np.save('train_images.npy', X_train)
np.save('y_train.npy', y_train)
print('----------TRAIN IMAGES AND LABELS SAVED----------')

#%%
# ----- LOAD TEST IMAGES AND RESHAPE FOR MODELING -----
print('----------LOADING TEST IMAGES----------')

X_real, y_real = image_to_array(mapping='../../data/Celeb-real.xlsx',
               image_fp='../../data/Celeb-real/Frames',
               image_size=224,
                      split='test')

X_fake, y_fake = image_to_array(mapping='../../data/Celeb-synthesis.xlsx',
               image_fp='../../data/Celeb-synthesis/Frames',
               image_size=224,
                      split='test')

X_test = np.concatenate((X_real, X_fake), axis=0)
y_test = y_real + y_fake

np.save('test_images.npy', X_test)
np.save('y_test.npy', y_test)

print('----------TEST IMAGES AND LABELS SAVED----------')
