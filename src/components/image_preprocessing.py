import os
import pandas as pd
import tqdm
import cv2
import numpy as np
from PIL import Image
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Input, Dense, RandomCrop, Resizing, RandomFlip, Conv2D, Multiply
)
#%%
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
        random_crop = RandomCrop(height=180, width=180)  # Crop smaller region
        resize = Resizing(height=224, width=224)
        flip = RandomFlip(mode="horizontal", seed=seed)
        contrast = tf.keras.layers.RandomContrast(0.2)
        brightness = tf.keras.layers.RandomBrightness(0.2)

        augmentations = tf.keras.Sequential([random_crop, resize, flip, contrast, brightness])

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

        # Rescale to [0, 255] for saving as an image
        augmented_image = (augmented_image * 255).astype(np.uint8)

        # Convert to PIL Image for saving
        augmented_image_pil = Image.fromarray(augmented_image)

        # Create augmented image filename
        augmented_image_name = os.path.splitext(image_file)[0] + '_augmented.jpg'
        augmented_image_path = os.path.join(output_arr_dir, augmented_image_name)

        # Save the augmented image as a JPG file
        augmented_image_pil.save(augmented_image_path)

        data['original_image_name'].append(image_file)
        data['original_image_path'].append(images_fp)
        data['augmented_image_name'].append(image_file + '_augmented')
        data['augmented_image_path'].append(output_arr_dir)

    data = pd.DataFrame(data)
    data.to_excel(output_info_dir + '/augmented_images.xlsx')
    print(f'Augmentation information saved to {output_info_dir}')

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