#%%
import sys
sys.path.append('../')

from components.utils import capture_frames

#%%
real = capture_frames(filepath='../../data/Celeb-real',
                      output_dir='../../data/Celeb-real/Frames',
                                  num_frames=5,
                                  label=1)
print(f'Here is a sample of the Celeb-real data: {real.head()}')
real.to_excel('../../data/Celeb-real.xlsx')

fake = capture_frames(filepath='../../data/Celeb-synthesis',
                      output_dir='../../data/Celeb-synthesis/Frames',
                                  num_frames=5,
                                  label=0)
print(f'Here is a sample of the Celeb-synthesis data: {fake.head()}')
fake.to_excel('../../data/Celeb-synthesis.xlsx')


