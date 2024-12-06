import os
import cv2
import pandas as pd

# Base directory for video data
video_base_dir = '/home/ubuntu/Final_Project/data'  # Update this path
output_dir = os.path.join(video_base_dir, "frames")  # Folder to save frames

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to extract frames from video at 100 FPS and save them
def convert_video_to_frames(video_path, output_dir, label):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 100

    # Calculate frame interval to achieve 100 FPS
    if original_fps > 0:
        frame_interval = max(1, int(original_fps / target_fps))
    else:
        print(f"Warning: Unable to retrieve FPS for {video_path}. Skipping video.")
        return []

    frames_info = []
    count = 0
    frame_count = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_name = f"{video_name}_{frame_count + 1}.jpg"
            frame_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(frame_path, frame)  # Save the frame as an image
            frames_info.append({"image_name": frame_name, "label": label})
            frame_count += 1
        count += 1

    cap.release()
    return frames_info

# Function to process all videos in specified folders and create an Excel file
def process_videos_and_create_excel(folders, output_dir, excel_file_path):
    all_frames_info = []

    # Iterate through specified folders
    for folder in folders:
        folder_path = os.path.join(video_base_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"Folder {folder} does not exist. Skipping...")
            continue

        # Assign label based on folder name
        if 'real' in folder.lower():
            label = 1
        elif 'synthesis' in folder.lower():
            label = 0
        else:
            label = -1  # Assign a default label for unknown categories

        # Process each video in the folder
        for video_file in os.listdir(folder_path):
            video_path = os.path.join(folder_path, video_file)
            if not video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add other formats if needed
                continue

            print(f"Processing: {video_file} in {folder}")
            frames_info = convert_video_to_frames(video_path, output_dir, label)
            all_frames_info.extend(frames_info)

    # Create a DataFrame and save it to an Excel file
    df = pd.DataFrame(all_frames_info)
    df.to_excel(excel_file_path, index=False)
    print(f"Excel file created at: {excel_file_path}")

# Main execution
folders_to_process = ['Celeb-real', 'Celeb-synthesis', 'Youtube-real']
excel_file_path = os.path.join(video_base_dir, "frames_labels.xlsx")
process_videos_and_create_excel(folders_to_process, output_dir, excel_file_path)