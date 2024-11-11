import os
import cv2


def extract_frames(video_path, output_folder, fps=5):
    """
    Extract frames from a video at a specified frame rate and save them to an output folder.

    Args:
    video_path (str): Path to the video file.
    output_folder (str): Path to the folder where frames will be saved.
    fps (int): Number of frames per second to extract.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get the video frame rate
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    frame_count = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save only every `frame_interval` frame
        if frame_count % frame_interval == 0:
            # Save the frame as a JPEG file
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_frame_count} frames from {video_path} and saved in {output_folder}")


def process_videos(input_folder, output_folder_real, output_folder_fake, fps=5):
    """
    Process videos in the input folders and save frames to respective output folders.

    Args:
    input_folder (str): Path to the folder containing videos.
    output_folder_real (str): Path to save frames from real videos.
    output_folder_fake (str): Path to save frames from fake videos.
    fps (int): Number of frames per second to extract.
    """
    # Paths for real and fake video folders
    real_videos_path = os.path.join(input_folder, "Celeb-real")
    fake_videos_path = os.path.join(input_folder, "Celeb-synthesis")

    # Process real videos
    for video_file in os.listdir(real_videos_path):
        video_path = os.path.join(real_videos_path, video_file)
        if os.path.isfile(video_path):
            output_path = os.path.join(output_folder_real, os.path.splitext(video_file)[0])
            extract_frames(video_path, output_path, fps=fps)

    # Process fake videos
    for video_file in os.listdir(fake_videos_path):
        video_path = os.path.join(fake_videos_path, video_file)
        if os.path.isfile(video_path):
            output_path = os.path.join(output_folder_fake, os.path.splitext(video_file)[0])
            extract_frames(video_path, output_path, fps=fps)


if __name__ == "__main__":
    # Define the paths
    input_folder = "/home/ubuntu/Group6_DL_FTP/data"
    output_folder_real = os.path.join(input_folder, "Images/real")
    output_folder_fake = os.path.join(input_folder, "Images/fake")

    # Process the videos
    process_videos(input_folder, output_folder_real, output_folder_fake, fps=5)