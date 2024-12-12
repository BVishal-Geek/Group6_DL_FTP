import os
import numpy as np
import pandas as pd
import tensorflow as tf

# Paths to data
excel_file_path = "/home/ubuntu/Final_Project/data/frames_labels.xlsx"  # Path to labels Excel file
images_dir_path = "/home/ubuntu/Final_Project/data/frames/"  # Path to frames directory


# Function to calculate running mean for images in sets
def calculate_running_mean(excel_file, images_dir, set_size=10000, img_size=(64, 64)):
    # Load the Excel file
    df = pd.read_excel(excel_file)

    # Separate real and fake image rows
    real_df = df[df['label'] == 1]
    fake_df = df[df['label'] == 0]

    # Initialize variables for real and fake means
    real_sum = np.zeros(3)  # To store the sum of RGB values for real images
    fake_sum = np.zeros(3)  # To store the sum of RGB values for fake images
    real_count = 0  # To store the count of processed real images
    fake_count = 0  # To store the count of processed fake images

    # Process real images in sets
    print("Processing Real Images...")
    for i in range(0, len(real_df), set_size):
        chunk = real_df.iloc[i:i + set_size]
        for _, row in chunk.iterrows():
            image_path = os.path.join(images_dir, row['image_name'])
            img = tf.keras.utils.load_img(image_path, target_size=img_size)
            img_array = tf.keras.utils.img_to_array(img) / 255.0  # Normalize pixel values

            real_sum += np.sum(img_array, axis=(0, 1))  # Add RGB sum for this image
            real_count += img_array.shape[0] * img_array.shape[1]  # Add total pixels in this image

        # Print intermediate results for debugging
        print(f"Processed {min(i + set_size, len(real_df))}/{len(real_df)} Real Images")

    # Process fake images in sets
    print("Processing Fake Images...")
    for i in range(0, len(fake_df), set_size):
        chunk = fake_df.iloc[i:i + set_size]
        for _, row in chunk.iterrows():
            image_path = os.path.join(images_dir, row['image_name'])
            img = tf.keras.utils.load_img(image_path, target_size=img_size)
            img_array = tf.keras.utils.img_to_array(img) / 255.0  # Normalize pixel values

            fake_sum += np.sum(img_array, axis=(0, 1))  # Add RGB sum for this image
            fake_count += img_array.shape[0] * img_array.shape[1]  # Add total pixels in this image

        # Print intermediate results for debugging
        print(f"Processed {min(i + set_size, len(fake_df))}/{len(fake_df)} Fake Images")

    # Calculate final means by dividing sums by total counts
    mean_real_rgb = real_sum / real_count if real_count > 0 else np.zeros(3)
    mean_fake_rgb = fake_sum / fake_count if fake_count > 0 else np.zeros(3)

    print(f"Final Mean RGB values for Real Images: {mean_real_rgb}")
    print(f"Final Mean RGB values for Fake Images: {mean_fake_rgb}")

    return mean_real_rgb, mean_fake_rgb


# Main Execution
if __name__ == "__main__":
    mean_real_rgb, mean_fake_rgb = calculate_running_mean(excel_file_path, images_dir_path)