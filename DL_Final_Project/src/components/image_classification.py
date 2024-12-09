import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, AdamW, SGD
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Input, Dense, RandomCrop, Resizing, RandomFlip, Conv2D, Multiply
)
import os
#%%
class FineTuneModel:
    def __init__(self, model_name='ResNet50', input_shape=(224, 224, 3), num_classes=2):
        """
        Initialize the FineTuneModel class.

        Parameters:
        - model_name: str, name of the pre-trained model to use ('ResNet50', 'VGG16', 'InceptionV3').
        - input_shape: tuple, input shape of the images (default: (224, 224, 3)).
        - num_classes: int, number of output classes.
        """
        self.model_name = model_name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_model = self._load_pretrained_model()
        self.model = None

    def _load_pretrained_model(self):
        """
        Load the specified pre-trained model without the top layer.
        """
        if self.model_name == 'ResNet50':
            return ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif self.model_name == 'VGG16':
            return VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif self.model_name == 'InceptionV3':
            return InceptionV3(weights='imagenet', include_top=False, input_shape=self.input_shape)
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}")

    def add_custom_layers(self):
        """
        Add custom top layers to the model.
        """
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)
        self.model = Model(inputs=self.base_model.input, outputs=predictions)
        # F1 Macro 44.73 0.17 0.73
        # gap = GlobalAveragePooling2D()(x)
        # attention = Dense(self.base_model.output_shape[-1], activation='sigmoid')(gap)
        # attention_output = Multiply()([x, attention])  # Element-wise multiplication
        # gap_output = GlobalAveragePooling2D()(attention_output)
        # output = Dense(1, activation='sigmoid')(gap_output)
        #
        # self.model = Model(inputs=self.base_model.input, outputs=output)


    def freeze_base_layers(self):
        """
        Freeze all layers in the base model to retain pre-trained features.
        """
        for layer in self.base_model.layers:
            layer.trainable = False

    def unfreeze_layers(self, num_layers):
        """
        Unfreeze the last `num_layers` layers of the model for fine-tuning.

        Parameters:
        - num_layers: int, number of layers to unfreeze from the end.
        """
        for layer in self.model.layers[-num_layers:]:
            layer.trainable = True

    def compile_model(self, learning_rate=0.0001, optimizer='adam'):
        """
        Compile the model with the specified optimizer and learning rate.

        Parameters:
        - learning_rate: float, learning rate for the optimizer.
        - optimizer: str, name of the optimizer ('adam' or 'adamw').
        """
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer == 'adamw':
            opt = AdamW(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = SGD(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        def f1_macro(y_true, y_pred):
            """
            Compute F1 macro score as a custom metric.
            """
            y_pred = tf.round(y_pred)  # Convert predictions to 0 or 1
            tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32), axis=0)
            fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32), axis=0)
            fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32), axis=0)

            precision = tp / (tp + fp + tf.keras.backend.epsilon())
            recall = tp / (tp + fn + tf.keras.backend.epsilon())
            f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())

            # Compute the mean F1 score across all classes
            f1_macro = tf.reduce_mean(f1)
            return f1_macro

        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', f1_macro])


    def train_model(self, train_gen, val_data, steps_per_epoch=None, validation_steps=None, epochs=10, callbacks=None, class_weight=None):
        """
        Trains the model using the provided training and validation data generators.

        Args:
            train_gen (tf.keras.preprocessing.image.DirectoryIterator or tf.data.Dataset):
                The training data generator or dataset, yielding batches of training data.
            val_data (tf.keras.preprocessing.image.DirectoryIterator or tf.data.Dataset):
                The validation data generator or dataset, yielding batches of validation data.
            steps_per_epoch (int, optional):
                Number of batches of data to process in each training epoch. If `None`, it is inferred
                from the size of `train_gen`.
            validation_steps (int, optional):
                Number of batches of data to process in each validation epoch. If `None`, it is inferred
                from the size of `val_data`.
            epochs (int, optional):
                Number of epochs to train the model (default: 10).
            callbacks (list, optional):
                List of Keras callback functions to apply during training (e.g., EarlyStopping, ModelCheckpoint).

        Returns:
            tf.keras.callbacks.History:
                A history object containing training and validation metrics for each epoch, accessible
                through the `history` attribute.
        """

        history = self.model.fit(
            train_gen,
            validation_data=val_data,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=2,
            callbacks=callbacks,
            class_weight=class_weight
        )
        return history


    def save_model(self, filepath):
        """
        Save the trained model to the specified filepath.

        Parameters:
        - filepath: str, path to save the model.
        """
        self.model.save(filepath)

    def model_summary(self, to_file=None):
        """
        Print or save the model summary.

        Parameters:
        - to_file: str, optional. Path to save the model summary to a file.
        """
        if self.model is None:
            raise ValueError("Model has not been created yet. Call 'add_custom_layers()' first.")

        if to_file:
            with open(to_file, 'w') as f:
                self.model.summary(print_fn=lambda x: f.write(x + '\n'))
            print(f"Model summary saved to {to_file}")
            print(self.model.summary())
        else:
            self.model.summary()

def compute_class_weights_from_generator(generator, steps):
    """
    Compute class weights from a generator.

    Args:
        generator: A generator yielding (x_batch, y_batch).
        steps: Number of steps to iterate (typically the size of the dataset divided by batch size).

    Returns:
        A dictionary with class weights.
    """
    label_counts = Counter()

    for _ in tqdm.tqdm(range(steps)):
        _, y_batch = next(generator)

        # Ensure y_batch is a flattened array of labels
        if len(y_batch.shape) > 1:  # For one-hot encoded labels
            y_batch = np.argmax(y_batch, axis=1)

        label_counts.update(y_batch)

    total_samples = sum(label_counts.values())
    num_classes = len(label_counts)
    class_weights = {
        cls: total_samples / (num_classes * count) for cls, count in label_counts.items()
    }

    return class_weights

def plot_accuracy(history, model_name, layers_info, image_name):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_name} Accuracy: {layers_info}')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'{image_name}.png')
    print(f'-----{image_name}.png SAVED-----')

def plot_loss(history, model_name, layers_info, image_name):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} Loss: {layers_info}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'{image_name}.png')
    print(f'-----{image_name}.png SAVED-----')

