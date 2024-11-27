import os
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.preprocessing import image

#%%

def image_to_array(mapping, image_fp, image_size=224):
    train_image = []
    data = pd.read_excel(mapping)

    for frame in tqdm.tqdm(range(len(data.video_name_frame))):
        # loading the image and keeping the target size as (224,224,3)
        frame_path = os.path.join(image_fp, data.loc[frame, 'video_name_frame'])

        img = image.load_img(frame_path, target_size=(image_size, image_size, 3))

            # converting it to array

        img = image.img_to_array(img)

        # normalizing the pixel value

        img = img / 255

        # appending the image to the train_image list

        train_image.append(img)

    X = np.array(train_image)
    return X



def capture_frames(filepath, output_dir, num_frames, label):

    data = {'video_name':[],
            'frame':[],
            'video_name_frame':[],
            'label':[]}

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