import os
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, RandomCrop, Resizing, RandomFlip
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, AdamW, SGD
#%%



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

        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


    def train_model(self, train_gen, val_data, steps_per_epoch=None, validation_steps=None, epochs=10, callbacks=None):
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
            callbacks=callbacks
        )
        return history


    def save_model(self, filepath):
        """
        Save the trained model to the specified filepath.

        Parameters:
        - filepath: str, path to save the model.
        """
        self.model.save(filepath)

def capture_frames(filepath, output_dir_train, output_dir_test, output_dir_valid, num_frames, label):
    """
    Captures frames from video files in a specified directory, splits them into training, testing,
    and validation sets, and saves the frames as images. Metadata about the frames is returned
    in a pandas DataFrame.

    Args:
        filepath (str): Path to the directory containing video files.
        output_dir_train (str): Directory where captured training frames will be saved.
        output_dir_test (str): Directory where captured testing frames will be saved.
        output_dir_valid (str): Directory where captured validation frames will be saved.
        num_frames (int): Number of frames to capture from each video.
        label (str): Class label for the video frames (e.g., "real" or "fake").

    Returns:
        pd.DataFrame: A DataFrame containing metadata for the captured frames, with the following columns:
            - 'video_name': Name of the source video file.
            - 'frame': Frame index within the video.
            - 'video_name_frame': Unique identifier for each frame image.
            - 'label': Class label for the frame.
            - 'split': Data split (train, test, or valid).
            - 'directory': Directory where the frame image is saved."""

    data = {'video_name':[],
            'frame':[],
            'video_name_frame':[],
            'label':[],
            'split':[],
            'directory':[]}

    print(f'\n\n----------THERE ARE {len(os.listdir(filepath))} VIDEOS----------\n\n')

    video_ids = list(set(os.listdir(filepath)))
    train_ids, test_ids = train_test_split(video_ids, test_size=0.2, random_state=6303)
    print(f'\n\n----------TRAIN TEST SPLIT OF IDS COMPLETE----------\n\n')

    train_ids, valid_ids = train_test_split(train_ids, test_size=0.2, random_state=6303)
    print(f'\n\n----------TRAIN VALIDATION SPLIT OF IDS COMPLETE----------\n\n')


    for video_file in tqdm.tqdm(os.listdir(filepath)):
        if video_file in train_ids:
            video_path = os.path.join(filepath, video_file)
            cap = cv2.VideoCapture(video_path)  # capturing the video from the given path

            # Total number of frames in the video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                print(f"Skipping {video_file}: unable to determine frame count.")
                continue

            # Calculate the interval to capture frames
            interval = max(1, total_frames // num_frames)

            count = 0
            for i in range(0, total_frames, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # Set the video position to the i-th frame
                ret, frame = cap.read()

                if not ret:
                    print(f"Frame {i} not readable in {video_file}. Skipping.")
                    continue

                filename = os.path.join(output_dir_train, f"{video_file}-frm-{count}.jpg")
                data['video_name'].append(video_file)
                data['frame'].append(count)
                data['video_name_frame'].append(f"{video_file}-frm-{count}.jpg")
                data['label'].append(label)
                data['split'].append('train')
                data['directory'].append(output_dir_train)

                # Save the frame as an image
                cv2.imwrite(filename, frame)
                count += 1

                # Stop if the desired number of frames is captured
                if count >= num_frames:
                    break

            cap.release()
        elif video_file in test_ids:
            video_path = os.path.join(filepath, video_file)
            cap = cv2.VideoCapture(video_path)  # capturing the video from the given path

            # Total number of frames in the video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                print(f"Skipping {video_file}: unable to determine frame count.")
                continue

            # Calculate the interval to capture frames
            interval = max(1, total_frames // num_frames)

            count = 0
            for i in range(0, total_frames, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # Set the video position to the i-th frame
                ret, frame = cap.read()

                if not ret:
                    print(f"Frame {i} not readable in {video_file}. Skipping.")
                    continue

                filename = os.path.join(output_dir_test, f"{video_file}-frm-{count}.jpg")
                data['video_name'].append(video_file)
                data['frame'].append(count)
                data['video_name_frame'].append(f"{video_file}-frm-{count}.jpg")
                data['label'].append(label)
                data['split'].append('test')
                data['directory'].append(output_dir_test)

                # Save the frame as an image
                cv2.imwrite(filename, frame)
                count += 1

                # Stop if the desired number of frames is captured
                if count >= num_frames:
                    break

            cap.release()
        else:
            video_path = os.path.join(filepath, video_file)
            cap = cv2.VideoCapture(video_path)  # capturing the video from the given path

            # Total number of frames in the video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                print(f"Skipping {video_file}: unable to determine frame count.")
                continue

            # Calculate the interval to capture frames
            interval = max(1, total_frames // num_frames)

            count = 0
            for i in range(0, total_frames, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # Set the video position to the i-th frame
                ret, frame = cap.read()

                if not ret:
                    print(f"Frame {i} not readable in {video_file}. Skipping.")
                    continue

                filename = os.path.join(output_dir_valid, f"{video_file}-frm-{count}.jpg")
                data['video_name'].append(video_file)
                data['frame'].append(count)
                data['video_name_frame'].append(f"{video_file}-frm-{count}.jpg")
                data['label'].append(label)
                data['split'].append('valid')
                data['directory'].append(output_dir_valid)

                # Save the frame as an image
                cv2.imwrite(filename, frame)
                count += 1

                # Stop if the desired number of frames is captured
                if count >= num_frames:
                    break

            cap.release()
    # Convert dictionary to DataFrame
    df = pd.DataFrame(data)
    print(f'\n\n----------THERE ARE {len(df.video_name_frame)} IMAGES----------\n\n')
    print(f'\n\n----------SPLIT OF IMAGES: {df["split"].value_counts()}----------\n\n')

    return df

def generate_augmentations(images_fp, images_names, output_info_dir, output_arr_dir, target_size=(224, 224, 3), augmentations=None, seed=6303):
    if augmentations is None:
        random_crop = RandomCrop(height=112, width=112)  # Crop smaller region
        resize = Resizing(height=224, width=224)
        flip = RandomFlip(mode="horizontal_and_vertical", seed=seed)

        augmentations = tf.keras.Sequential([random_crop, resize, flip])

    image_array = []
    data = {

        'original_image_name': [],
        'original_image_path': [],
        'augmented_image_name': [],
        'augmented_image_path': []
    }


    for image_file in tqdm.tqdm(images_names):
        # Check if it's a valid file
        image_path = os.path.join(images_fp, image_file)
        if not os.path.isfile(image_path):
            print(f"Skipping {image_file}: Not a valid file")
            continue

        img = image.load_img(image_path, target_size=target_size)

        # Convert the image to a numpy array if it is not already
        if not isinstance(img, np.ndarray):
            img = img_to_array(img)

        # Ensure the image is scaled to [0, 1]
        img = img / 255.0 if img.max() > 1 else img

        # Add batch dimension and convert to a tensor
        image_tensor = tf.expand_dims(img, axis=0)

        # Apply the augmentation
        augmented_image_tensor = augmentations(image_tensor, training=True)

        # Remove batch dimension and convert back to numpy array
        augmented_image = tf.squeeze(augmented_image_tensor).numpy()

        image_array.append(augmented_image)

        data['original_image_name'].append(image_file)
        data['original_image_path'].append(images_fp)
        data['augmented_image_name'].append(image_file + '_augmented')
        data['augmented_image_path'].append(output_arr_dir)

    data = pd.DataFrame(data)
    data.to_excel(output_info_dir + '/augmented_images.xlsx')
    print(f'Augmentation information saved to {output_info_dir}')

    np.save(output_arr_dir + '/augmented_images.npy', image_array)

def image_to_array(mapping, image_fp, split, image_size=224, batch_size=16):

    data = pd.read_excel(mapping)
    data = data[data['split']==split]

    total_samples = len(data)
    image_array = []
    y_array = []

    for start_idx in tqdm.tqdm(range(0, total_samples, batch_size)):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_data = data.iloc[start_idx:end_idx]

        batch_image_array = []
        batch_y_array = []

        for _, row in batch_data.iterrows():
            frame_path = os.path.join(image_fp, row['video_name_frame'])
            img = image.load_img(frame_path, target_size=(image_size, image_size, 3))
            img = image.img_to_array(img) / 255.0  # Normalize
            batch_image_array.append(img)
            batch_y_array.append(row['label'])

        image_array.extend(batch_image_array)
        y_array.extend(batch_y_array)

    X = np.array(image_array)
    y = np.array(y_array)

    return X, y