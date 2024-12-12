import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from io import BytesIO  # Import BytesIO for in-memory operations
import os

def generate_ela_image(frame, quality=95, brightness_factor=1.5):
    """
    Generate an ELA (Error Level Analysis) image from a video frame.

    Args:
        frame: A single video frame in BGR format (NumPy array).
        quality: JPEG compression quality for ELA (default: 95).
        brightness_factor: Factor to adjust brightness (default: 1.5).

    Returns:
        ela_frame: Brightened ELA-transformed frame as a NumPy array.
    """
    # Convert frame to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Save frame as compressed JPEG to a memory buffer
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    
    # Reload compressed image
    compressed_image = Image.open(buffer)
    
    # Compute the difference between the original and compressed images
    ela_image = ImageChops.difference(pil_image, compressed_image)
    
    # Enhance the ELA image to amplify differences
    ela_image = ela_image.point(lambda x: x * 10)  # Adjust amplification factor as needed
    
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(ela_image)
    bright_ela_image = enhancer.enhance(brightness_factor)
    
    # Convert back to NumPy array (BGR format)
    ela_frame = cv2.cvtColor(np.array(bright_ela_image), cv2.COLOR_RGB2BGR)
    return ela_frame

def process_video(video_path, output_folder, quality=95, brightness_factor=1.5):
    """
    Process a video to generate brightened ELA frames for each frame in the video.

    Args:
        video_path: Path to the input video.
        output_folder: Folder to save the ELA frames.
        quality: JPEG compression quality for ELA (default: 95).
        brightness_factor: Factor to adjust brightness (default: 1.5).
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Processing video: {video_path}")
    print(f"Total Frames: {frame_count}, FPS: {fps}, Resolution: {width}x{height}")
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Generate ELA frame
        try:
            ela_frame = generate_ela_image(frame, quality=quality, brightness_factor=brightness_factor)
            
            # Save ELA frame to output folder
            output_path = os.path.join(output_folder, f"frame_{frame_idx:05d}_ela.jpg")
            cv2.imwrite(output_path, ela_frame)
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
        
        frame_idx += 1
    
    cap.release()
    print(f"ELA frames saved in: {output_folder}")

# Example usage
video_path = "id2_id4_0000.mp4"  # Replace with your video path
output_folder = "/Users/vishal/PycharmProjects/DeepLearning_FTP/Group6_DL_FTP/data"   # Folder to save ELA frames
process_video(video_path, output_folder, quality=95, brightness_factor=1.5)
