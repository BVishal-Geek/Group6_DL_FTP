#%%
import sys
sys.path.append('../')

from components.utils import capture_frames

#%%
real = capture_frames(filepath='../../data/Celeb-real-sample',
                      output_dir='../../data/Celeb-real-sample/Frames',
                                  num_frames=5,
                                  label=1)
print(f'Here is a sample of the Celeb-real-sample data: {real.head()}')
real.to_excel('../../data/Celeb-real-sample.xlsx')

fake = capture_frames(filepath='../../data/Celeb-synthesis-sample',
                      output_dir='../../data/Celeb-synthesis-sample/Frames',
                                  num_frames=5,
                                  label=0)
print(f'Here is a sample of the Celeb-synthesis-sample data: {fake.head()}')
fake.to_excel('../../data/Celeb-synthesis-sample.xlsx')
