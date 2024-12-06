import os
import numpy as np
import pandas as pd
import tensorflow as tf

# Paths to data
excel_file_path = "/home/ubuntu/Final_Project/data/frames_labels.xlsx"  # Path to labels Excel file
images_dir_path = "/home/ubuntu/Final_Project/data/frames/"  # Path to frames directory


# Function to calculate mean RGB values for real and fake images
def calculate_mean_rgb(excel_file, images_dir, img_size=(64, 64)):
    # Load the Excel file
    df = pd.read_excel(excel_file)

    real_images = []
    fake_images = []

    for _, row in df.iterrows():
        image_path = os.path.join(images_dir, row['image_name'])

        # Load image and resize
        img = tf.keras.utils.load_img(image_path, target_size=img_size)
        img_array = tf.keras.utils.img_to_array(img) / 255.0  # Normalize pixel values

        if row['label'] == 1:  # Real image
            real_images.append(img_array)
        else:  # Fake image
            fake_images.append(img_array)

    # Calculate mean RGB values for real and fake images
    mean_real_rgb = np.mean(np.array(real_images), axis=(0, 1, 2))
    mean_fake_rgb = np.mean(np.array(fake_images), axis=(0, 1, 2))

    print(f"Mean RGB values for Real Images: {mean_real_rgb}")
    print(f"Mean RGB values for Fake Images: {mean_fake_rgb}")

    return mean_real_rgb, mean_fake_rgb


# Main Execution
if __name__ == "__main__":
    mean_real_rgb, mean_fake_rgb = calculate_mean_rgb(excel_file_path, images_dir_path)