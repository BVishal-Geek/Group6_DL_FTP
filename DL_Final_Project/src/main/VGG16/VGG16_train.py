#%%
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import optimizers

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import sys
sys.path.append("../../")

from components.utils import *
#%%
# ----- LOAD TRAIN IMAGES AND RESHAPE FOR MODELING -----
print('----------LOADING TRAIN IMAGES----------')

X = np.load('train_images.npy')
y = np.load('y_train.npy')

print('----------TRAIN IMAGES LOADED----------')
y = to_categorical(y, num_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6303, test_size=0.2)

#%%
base_model = FineTuneModel(model_name='VGG16', input_shape=(224, 224, 3), num_classes=2)

# Build and train the model
base_model.add_custom_layers()
base_model.freeze_base_layers()
base_model.compile_model(learning_rate=0.0001)

base_history = base_model.train_model((X_train, y_train), (X_test, y_test), epochs=15, batch_size=16)
plot_loss(history=base_history, model_name='VGG16', layers_info='Pretrained Layers Frozen', image_name='VGG16_pretrained')

#%%
base_model.unfreeze_layers(num_layers=30)
base_model.compile_model(optimizer='adamw',learning_rate=0.00001)
base_history_ft = base_model.train_model((X_train, y_train), (X_test, y_test), epochs=25, batch_size=16)
plot_loss(history=base_history_ft, model_name='VGG16', layers_info='Last 30 Layers Unfrozen', image_name='VGG16_finetuned')

# Save the fine-tuned model
base_model.save_model('finetuned_vgg16.h5')
