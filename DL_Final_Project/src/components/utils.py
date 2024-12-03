import os
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.preprocessing import image
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, AdamW, SGD
#%%

def plot_accuracy(history, model_name, layers_info, image_name):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_name} Accuracy: {layers_info}')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'{image_name}.png')
    print(f'-----{image_name}.png SAVED-----')

def plot_loss(history, model_name, layers_info, image_name):
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
        predictions = Dense(self.num_classes, activation='softmax')(x)
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

        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, train_data, val_data, epochs=10, batch_size=8):
        """
        Train the model on the given dataset.

        Parameters:
        - train_data: tuple, training data (X_train, y_train).
        - val_data: tuple, validation data (X_val, y_val).
        - epochs: int, number of epochs to train (default: 10).
        - batch_size: int, size of the training batches (default: 8).

        Returns:
        - history: training history object.
        """
        X_train, y_train = train_data
        X_val, y_val = val_data

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=2
        )
        return history

    def save_model(self, filepath):
        """
        Save the trained model to the specified filepath.

        Parameters:
        - filepath: str, path to save the model.
        """
        self.model.save(filepath)

def image_to_array(mapping, image_fp, split, image_size=224):
    image_array = []

    data = pd.read_excel(mapping)
    data = data[data['split']==split]

    y_array = []

    for frame in tqdm.tqdm(range(len(data.video_name_frame))):
        # loading the image and keeping the target size as (224,224,3)

        frame_path = os.path.join(image_fp, data.iloc[frame]['video_name_frame'])

        img = image.load_img(frame_path, target_size=(image_size, image_size, 3))

            # converting it to array

        img = image.img_to_array(img)

        # normalizing the pixel value

        img = img / 255

        # appending the image to the image_array list

        image_array.append(img)
        y_array.append(data.iloc[frame]['label'])

    X = np.array(image_array)
    return X, y_array


def capture_frames(filepath, output_dir, num_frames, label, split=0.8):
    """
        Captures frames from videos in the specified filepath and organizes the data into a dataframe.

        Args:
            filepath (str): Path to the directory containing video files.
            output_dir (str): Directory to save captured frames as images.
            num_frames (int): Number of frames to capture from each video.
            label (str): Label for the frames (e.g., class label).
            split (float): Fraction of frames to use as training data (0 < split <= 1).

        Returns:
            pd.DataFrame: DataFrame containing metadata for captured frames.
        """

    data = {'video_name':[],
            'frame':[],
            'video_name_frame':[],
            'label':[],
            'split':[]}

    print(f'----------THERE ARE {len(os.listdir(filepath))} VIDEOS----------')
    for video_file in tqdm.tqdm(os.listdir(filepath)):
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

            filename = os.path.join(output_dir, f"{video_file}-frm-{count}.jpg")
            data['video_name'].append(video_file)
            data['frame'].append(count)
            data['video_name_frame'].append(f"{video_file}-frm-{count}.jpg")
            data['label'].append(label)

            # Assign split type
            split_type = 'train' if count < split * num_frames else 'test'
            data['split'].append(split_type)

            # Save the frame as an image
            cv2.imwrite(filename, frame)
            count += 1

            # Stop if the desired number of frames is captured
            if count >= num_frames:
                break

        cap.release()
    # Convert dictionary to DataFrame
    df = pd.DataFrame(data)
    print(f'----------THERE ARE {len(df.video_name_frame)} IMAGES----------')
    return df


def create_dataset_metadata(video_folder, folder_name, label=0):
    """
    Processes videos in a folder and creates a dataset with video metadata and frames.

    Parameters:
    - video_folder: Path to the folder containing video files.
    - folder_name: The name of the folder containing video files
    - label: Label to assign to each video (e.g., 1 for fake, 0 for real).

    Returns:
    - DataFrame with columns:  'folder_name', 'video_name', 'label'
    """

    data = {'folder_name':[],
        'video_name': [],
        'label': []
    }

    for video_file in tqdm.tqdm(os.listdir(video_folder)):
        video_path = os.path.join(video_folder, video_file)

        # Check if it's a valid video file
        if not os.path.isfile(video_path):
            continue

        # Append data to the dictionary
        data['folder_name'].append(folder_name)
        data['video_name'].append(video_file)
        data['label'].append(label)

    # Convert dictionary to DataFrame
    df = pd.DataFrame(data)
    return df