import numpy as np
import sys
sys.path.append("../")

from components.utils import *
#%%
# ----- LOAD TEST IMAGES AND RESHAPE FOR MODELING -----

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

X_test, y_test = process_input(X=X_test,
              y=y_test,
              model='ResNet50',
              num_classes=2,
              input_type='test')
#%%
# ----- LOAD TRAINED MODEL AND GENERATE PREDICTIONS -----

input_shape = int(X_test.shape[1])
model = CustomModel(input_shape=input_shape)
model.load_model(".weights.h5")
model.compile_model()
scores = model.test_model(X_test, y_test)