#%%
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
sys.path.append("../../")

from components.image_classification import *
from components.image_preprocessing import *
#%%
print('\n\n----------INSTANTIATING IMAGE GENERATORS----------\n\n')

train_directory = '../../../data/Frames_train_ela'
valid_directory = '../../../data/Frames_valid_ela'
BATCH_SIZE = 64
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
base_model = FineTuneModel(model_name='VGG16', input_shape=(224, 224, 3), num_classes=2)

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
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode='min')

base_history = base_model.train_model(train_generator, valid_generator, epochs=15, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,callbacks=[early_stopping], class_weight=class_weights)
plot_loss(history=base_history, model_name='VGG16_ela', layers_info='Pretrained Layers Frozen', image_name='VGG16_pretrained_ela')

#%%
base_model.unfreeze_layers(num_layers=30)
base_model.compile_model(optimizer='adam',learning_rate=0.00001)

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25,
                              patience=5)
checkpoint = ModelCheckpoint('finetuned_vgg16.keras',save_best_only=True, monitor='val_loss', verbose=1, mode='min')

base_history_ft = base_model.train_model(train_generator, valid_generator, epochs=25, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, callbacks=[early_stopping, reduce_lr, checkpoint], class_weight=class_weights)
plot_loss(history=base_history_ft, model_name='VGG16_ela', layers_info='All Layers Unfrozen', image_name='VGG16_finetuned_ela')

# Save the fine-tuned model
name = 'finetuned_vgg16_ela'
base_model.save_model(f'{name}.keras')

#%%
base_model.model_summary(to_file=name)
print(base_model.model_summary())