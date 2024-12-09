#%%
import os

import numpy as np
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
sys.path.append("../../")

from components.image_classification import *
#%%
print('\n\n----------INSTANTIATING IMAGE GENERATORS----------\n\n')

train_directory = '../../../data/Frames_train'
valid_directory = '../../../data/Frames_valid'
BATCH_SIZE = 32
generator = ImageDataGenerator()
train_generator = generator.flow_from_directory(
    directory=train_directory,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=6303
)

valid_generator = generator.flow_from_directory(
    directory=valid_directory,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=6303)
#%%
base_model = FineTuneModel(model_name='ResNet50', input_shape=(224, 224, 3), num_classes=2)

# Build and train the model
base_model.add_custom_layers()
base_model.freeze_base_layers()
base_model.compile_model(learning_rate=0.0001)

steps_per_epoch = train_generator.n // BATCH_SIZE
validation_steps = valid_generator.n // BATCH_SIZE
print(f'\n\n----------TRAIN STEPS PER EPOCH: {steps_per_epoch}----------\n\n')
print(f'\n\n----------VALIDATION STEP: {validation_steps}----------\n\n')
#%%
print(f'\n\n----------CALCULATING CLASS WEIGHTS----------\n\n')
class_weights = compute_class_weights_from_generator(train_generator, steps_per_epoch)

print(f'\n\n----------CLASS WEIGHTS {class_weights}----------\n\n')

#%%
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

base_history = base_model.train_model(train_generator, valid_generator, epochs=15, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,callbacks=[early_stopping], class_weight=class_weights)
plot_loss(history=base_history, model_name='ResNet50', layers_info='Pretrained Layers Frozen', image_name='ResNet50_pretrained')

#%%
base_model.unfreeze_layers(num_layers=30)
base_model.compile_model(optimizer='adamw',learning_rate=0.00001)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.000005,
                              patience=7, min_lr=0.00001)

base_history_ft = base_model.train_model(train_generator, valid_generator, epochs=25, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, callbacks=[early_stopping, reduce_lr], class_weight=class_weights)
plot_loss(history=base_history_ft, model_name='ResNet50', layers_info='Last 30 Layers Unfrozen', image_name='ResNet50_finetuned')

# Save the fine-tuned model
base_model.save_model('finetuned_resnet50.keras')
