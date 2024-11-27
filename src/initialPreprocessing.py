import os
import cv2

def video_to_frames(input_dir, output_dir):
    """
    Converts videos in a directory to sequences of JPG frames.

    Args:
    input_dir (str): Path to the directory containing videos.
    output_dir (str): Path to the directory where frames will be saved.
    """

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each video file in the input directory
    for filename in os.listdir(input_dir):
        video_path = os.path.join(input_dir, filename)

        # Skip non-video files
        if not os.path.isfile(video_path) or not video_path.lower().endswith(('.mp4', '.avi', '.mkv')):
            print(f"Skipping non-video file: {filename}")
            continue

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video {video_path}")
            continue

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Save the frame as a JPG image with a unique name
            # Include video name in the filename to avoid collisions
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}_frame_{frame_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            frame_count += 1

        cap.release()
        print(f"Extracted {frame_count} frames from {filename} to {output_dir}")


# Paths for real and synthetic videos
real_videos_path = "/home/ubuntu/Group6_DL_FTP/data/Celeb-real"
synthetic_videos_path = "/home/ubuntu/Group6_DL_FTP/data/Celeb-synthesis"

# Output directories for extracted frames
output_dir_real = "/home/ubuntu/Group6_DL_FTP/data/Images/real"
output_dir_synthetic = "/home/ubuntu/Group6_DL_FTP/data/Images/fake"

# Process the videos
print("Processing real videos...")
video_to_frames(real_videos_path, output_dir_real)

print("Processing synthetic videos...")
video_to_frames(synthetic_videos_path, output_dir_synthetic)