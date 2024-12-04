import os
import cv2
import numpy as np

# Base directory where datasets are located
base_dir = '/home/ubuntu/Final_Project/data/'

# Data folders
folders = ['ffhq', 'pggan_v1', 'pggan_v2']

# Initialize counts and pixel sums
real_images_count = 0
fake_images_count = {'pggan_v1': 0, 'pggan_v2': 0}
real_images_pixel_sum = np.zeros((3,))
fake_images_pixel_sum = {'pggan_v1': np.zeros((3,)), 'pggan_v2': np.zeros((3,))}

# Iterate over each folder and split
for folder in folders:
    for split in ['train', 'validation', 'test']:
        folder_path = os.path.join(base_dir, folder, split)
        print(f'Processing folder: {folder_path}')  # Print folder being processed
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.jpg') or file.endswith('.png'):
                    # Load image
                    image_path = os.path.join(folder_path, file)
                    image = cv2.imread(image_path)
                    if image is not None:
                        # Resize image to a common size if needed (e.g., 224x224)
                        image = cv2.resize(image, (224, 224))
                        # Calculate total pixel sum
                        total_pixel = np.sum(image, axis=(0, 1))

                        if folder == 'ffhq':  # Real images
                            real_images_pixel_sum += total_pixel
                            real_images_count += 1
                        elif folder in ['pggan_v1', 'pggan_v2']:  # Fake images
                            fake_images_pixel_sum[folder] += total_pixel
                            fake_images_count[folder] += 1

# Calculate average pixel values for real images
if real_images_count > 0:
    real_avg = real_images_pixel_sum / (real_images_count * 224 * 224)
else:
    real_avg = None

# Calculate average pixel values for each fake dataset
fake_avg = {}
for key in fake_images_pixel_sum.keys():
    if fake_images_count[key] > 0:
        fake_avg[key] = fake_images_pixel_sum[key] / (fake_images_count[key] * 224 * 224)
    else:
        fake_avg[key] = None

# Print the results
print(f'Total Real Images: {real_images_count}')
print(f'Average Pixel Values for Real Images (ffhq): {real_avg}')

for key in fake_avg.keys():
    print(f'Total Fake Images in {key}: {fake_images_count[key]}')
    print(f'Average Pixel Values for Fake Images ({key}): {fake_avg[key]}')


'''
The output of the file is:
Total Real Images: 19999
Average Pixel Values for Real Images (ffhq): [ 99.01338834 112.72465243 147.16333224]
Total Fake Images in pggan_v1: 19943
Average Pixel Values for Fake Images (pggan_v1): [ 93.78390484 112.38124008 149.01953568]
Total Fake Images in pggan_v2: 19962
Average Pixel Values for Fake Images (pggan_v2): [ 95.487471   113.51148909 149.9662285 ]
'''