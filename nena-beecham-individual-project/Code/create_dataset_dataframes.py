#%%
import sys
sys.path.append('../')

from components.utils import *

#%%
# Generate video metadata and create train-test splits
# 591 Videos
real = capture_frames(filepath='../../data/Celeb-real',
                      output_dir_train='../../data/Frames_train/Class_1',
                      output_dir_test='../../data/Frames_test/Class_1',
                      output_dir_valid='../../data/Frames_valid/Class_1',
                      num_frames=10,
                      label=1) #THERE ARE 5891 IMAGES
print(f'Here is a sample of the Celeb-real data: {real.head()}')
real.to_excel('../../data/Real_Frame_Metadata.xlsx')

# 2341 Videos
fake = capture_frames(filepath='../../data/Celeb-synthesis',
                      output_dir_train='../../data/Frames_train/Class_2',
                      output_dir_test='../../data/Frames_test/Class_2',
                      output_dir_valid='../../data/Frames_valid/Class_2',
                      num_frames=10,
                      label=0) #THERE ARE 23310 IMAGES
print(f'Here is a sample of the Celeb-synthesis data: {fake.head()}')
fake.to_excel('../../data/Fake_Frame_Metadata.xlsx')


