#%%
import os
import random

import sys
sys.path.append('../')

from components.utils import generate_augmentations
#%%

seed = 6303
random.seed(6303)

images_fp = '../../data/Celeb-real/Frames'
output_arr_dir = '../../data/Celeb-real/Frames/Augmentations'
output_info_dir = '../../data/'

images = os.listdir(images_fp)
ratio = int(len(images) * 0.3)

print(f'\n\n----- {ratio} IMAGES TO BE USED FOR AUGMENTATIONS -----\n\n')
images_sample = random.sample(images, ratio)

print(f'\n\n----- BEGINNING IMAGE AUGMENTATIONS -----\n\n')
generate_augmentations(images_fp, images_sample, output_arr_dir=output_arr_dir, output_info_dir=output_info_dir)

print(f'\n\n----- IMAGE AUGMENTATIONS COMPLETE-----\n\n')
