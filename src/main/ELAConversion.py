import os
from PIL import Image, ImageChops, ImageEnhance
import tqdm
# Define paths for the original and ELA folders
base_path = "../../data/YT_Frames_test"  # Update with your base folder path
folders = ["Class_1", "Class_2"]

# Output ELA folder paths
output_folders = {folder: os.path.join(base_path, f"{folder}_ela") for folder in folders}

# Create output directories if they don't exist
for folder, output_path in output_folders.items():
    os.makedirs(output_path, exist_ok=True)


def convert_to_ela(image_path, output_path, quality=90):
    """
    Convert an image to ELA format and save it to the output path.

    Args:
        image_path (str): Path to the original image.
        output_path (str): Path to save the ELA image.
        quality (int): JPEG compression quality for ELA.
    """
    # Open the original image
    original = Image.open(image_path).convert("RGB")

    # Save a compressed version of the image
    compressed_path = f"{output_path}_temp.jpg"
    original.save(compressed_path, "JPEG", quality=quality)

    # Open the compressed version and compute the difference
    compressed = Image.open(compressed_path)
    ela_image = ImageChops.difference(original, compressed)

    # Enhance the difference for better visualization
    extrema = ela_image.getextrema()
    max_diff = max([ext[1] for ext in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    # Save the ELA image
    ela_image.save(output_path)

    # Remove the temporary compressed file
    os.remove(compressed_path)


# Process each folder
for folder in folders:
    input_folder = os.path.join(base_path, folder)
    output_folder = output_folders[folder]

    for image_name in tqdm.tqdm(os.listdir(input_folder)):
        input_path = os.path.join(input_folder, image_name)
        output_path = os.path.join(output_folder, image_name)

        if os.path.isfile(input_path):
            try:
                convert_to_ela(input_path, output_path)
                # print(f"Processed {image_name} -> {output_path}")
            except Exception as e:
                print(f"Error processing {image_name}: {e}")

