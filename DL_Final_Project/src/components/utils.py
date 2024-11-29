import os
import gc
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.preprocessing import image
from keras.applications import VGG16, ResNet50, InceptionV3, EfficientNetB6, EfficientNetV2S
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten
from sklearn.metrics import classification_report
#%%

class CustomModel:
    def __init__(self, input_shape):
        # Initialize the model
        self.model = Sequential([
            Dense(512, activation='relu', input_shape=(input_shape,)),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(2, activation='softmax')
        ])
        print("Model initialized.")

    def compile_model(self):
        # Compile the model
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        print("Model compiled.")

    def train_model(self, X_train, y_train, batch_size=32, epochs=10, validation_data=None, callbacks=None):
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            verbose=1,
            callbacks=callbacks
        )
        print("Training completed.")
        return history

    def save_model(self, file_path):
        # Save model weights
        self.model.save_weights(file_path)
        print(f"Weights saved to {file_path}.")

    def load_model(self, file_path):
        # Load model weights
        self.model.load_weights(file_path)
        print(f"Weights loaded from {file_path}.")

    def test_model(self, test_images, test_labels):
        # Test the model
        scores = self.model.evaluate(test_images, test_labels, verbose=0)
        print(f"Test Loss: {scores[0]:.4f}, Test Accuracy: {scores[1]:.4f}")

        # Generate predictions
        predictions = self.model.predict(test_images)
        predicted_classes = predictions.argmax(axis=1)
        true_classes = test_labels.argmax(axis=1)

        # Classification Report
        report = classification_report(true_classes, predicted_classes, target_names=['Class 0', 'Class 1'])
        print("\nClassification Report:\n", report)

        return scores, report


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

def process_input(X, y, model, num_classes, input_type):
    """
        Processes input data for training or testing based on the specified model and input type.

        Args:
            X (numpy.ndarray): Input data containing images.
            y (numpy.ndarray): Labels corresponding to the input data.
            model (str): The name of the pre-trained model to use for feature extraction.
                Options include:
                    - 'VGG16'
                    - 'ResNet50'
                    - 'InceptionV3'
                    - 'EfficientNetB6'
                    - 'EfficientNetV2S'
            num_classes (int): Number of output classes for the classification task.
            input_type (str): Specifies the type of input processing.
                - 'train': Splits data into training and testing sets, extracts features, and reshapes.
                - 'test': Extracts features from the input data without splitting.

        Returns:
            If `input_type` is 'train':
                tuple: (X_train, X_test, y_train, y_test)
                    - X_train (numpy.ndarray): Processed training features.
                    - X_test (numpy.ndarray): Processed testing features.
                    - y_train (numpy.ndarray): One-hot encoded training labels.
                    - y_test (numpy.ndarray): One-hot encoded testing labels.
            If `input_type` is 'test':
                tuple: (X, y)
                    - X (numpy.ndarray): Processed input features.
                    - y (numpy.ndarray): One-hot encoded labels."""

    y = to_categorical(y, num_classes=num_classes)

    if input_type == 'train':
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6303, test_size=0.2, stratify=y)

        if model == 'VGG16':
            base_model = VGG16(weights='imagenet', include_top=False)
            X_train = base_model.predict(X_train)
            X_train = X_train.reshape(X_train[0], X_train[1]*X_train[2]*X_train[3])


            X_test = base_model.predict(X_test)
            X_test = X_test.reshape(X_test[0], X_test[1] * X_test[2] * X_test[3])

        elif model == 'ResNet50':
            base_model = ResNet50(weights='imagenet', include_top=False)
            batch_size = 32  # Set based on available memory
            X_train_features = []
            X_test_features = []

            # Processing X_train in batches
            for i in range(0, X_train.shape[0], batch_size):
                batch = X_train[i:i + batch_size]
                features = base_model.predict(batch)
                X_train_features.append(features)
                del batch  # Free up memory
                gc.collect()  # Force garbage collection

            X_train = np.vstack(X_train_features)
            X_train = X_train.reshape(X_train.shape[0], -1)

            # Processing X_test in batches
            for i in range(0, X_test.shape[0], batch_size):
                batch = X_test[i:i + batch_size]
                features = base_model.predict(batch)
                X_test_features.append(features)
                del batch  # Free up memory
                gc.collect()  # Force garbage collection

            X_test = np.vstack(X_test_features)
            X_test = X_test.reshape(X_test.shape[0], -1)

        elif model == 'InceptionV3':
            base_model = InceptionV3(weights='imagenet', include_top=False)
            X_train = base_model.predict(X_train)
            X_train = X_train.reshape(X_train[0], X_train[1] * X_train[2] * X_train[3])

            X_test = base_model.predict(X_test)
            X_test = X_test.reshape(X_test[0], X_test[1] * X_test[2] * X_test[3])

        elif model == 'EfficientNetB6':
            base_model = EfficientNetB6(weights='imagenet', include_top=False)
            X_train = base_model.predict(X_train)
            X_train = X_train.reshape(X_train[0], X_train[1] * X_train[2] * X_train[3])

            X_test = base_model.predict(X_test)
            X_test = X_test.reshape(X_test[0], X_test[1] * X_test[2] * X_test[3])

        elif model == 'EfficientNetV2S':
            base_model = EfficientNetV2S(weights='imagenet', include_top=False)
            X_train = base_model.predict(X_train)
            X_train = X_train.reshape(X_train[0], X_train[1] * X_train[2] * X_train[3])

            X_test = base_model.predict(X_test)
            X_test = X_test.reshape(X_test[0], X_test[1] * X_test[2] * X_test[3])

        return X_train, X_test, y_train, y_test

    elif input_type == 'test':
        if model == 'VGG16':
            base_model = VGG16(weights='imagenet', include_top=False)
            X = base_model.predict(X)
            X = X.reshape(X[0], X[1]*X[2]*X[3])

        elif model == 'ResNet50':
            print('-----EXTRACTING FEATURES FOR TEST IMAGES-----')
            base_model = ResNet50(weights='imagenet', include_top=False)
            batch_size = 32  # Set based on available memory
            X_test_features = []

            # Processing X_train in batches
            for i in range(0, X.shape[0], batch_size):
                batch = X[i:i + batch_size]
                features = base_model.predict(batch)
                X_test_features.append(features)
                del batch  # Free up memory
                gc.collect()  # Force garbage collection
            print(f'-----FINISHED PROCESSING ALL TEST IMAGE BATCHES-----')

            X = np.vstack(X_test_features)

            print(f'-----RESHAPING IMAGE SHAPES-----')
            X = X.reshape(X.shape[0], -1)

            print(f'-----INPUT SHAPE: {X.shape}-----')

        elif model == 'InceptionV3':
            base_model = InceptionV3(weights='imagenet', include_top=False)
            X = base_model.predict(X)
            X = X.reshape(X[0], X[1] * X[2] * X[3])

        elif model == 'EfficientNetB6':
            base_model = EfficientNetB6(weights='imagenet', include_top=False)
            X = base_model.predict(X)
            X = X.reshape(X[0], X[1] * X[2] * X[3])

        elif model == 'EfficientNetV2S':
            base_model = EfficientNetV2S(weights='imagenet', include_top=False)
            X = base_model.predict(X)
            X = X.reshape(X[0], X[1] * X[2] * X[3])

        return X, y


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

def read_images(filepath):
    data = pd.read_excel(filepath)

    X = []
    for img_name in data['video_name']:
        img = plt.imread('' + img_name)
        X.append(img)
    X = np.array(X)