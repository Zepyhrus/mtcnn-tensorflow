# In[]
import sys
sys.path.append('..')

from preprocess.utils import mkdir
from preprocess.BBox_utils import getDataFromTxt

import os
from os.path import join
import shutil

import numpy as np
import numpy.random as npr
import pandas as pd

import cv2

from tqdm import tqdm


#%% bounding boxes calibration for celeba dataset
bboxes_file = 'data/Anno/list_bbox_celeba.txt'
landmarks_file = 'data/Anno/list_landmarks_celeba.txt'

img_dir = 'data/Img/img_celeba.7z/img_celeba'


bboxes = pd.read_csv(bboxes_file, delim_whitespace=True, skiprows=1)
landmarks = pd.read_csv(landmarks_file, delim_whitespace=True, skiprows=1)

labels = bboxes.merge(landmarks)

#%%
# sample around 10000 images and draw the bounding boxes and 
sub_labels = labels.sample(labels.shape[0] // 20)


for i in tqdm(range(sub_labels.shape[0])):
  src_img = join(img_dir, sub_labels.image_id.iloc[i])
  des_img = join('pictures', sub_labels.image_id.iloc[i])

  try:
    shutil.copy(src_img, des_img)
  except:
    print(src_img)
    



# validation of Celeba bounding boxes labels
"""
for i in tqdm(range(2000)):
  img = cv2.imread(join(img_dir, sub_labels.image_id.iloc[i]))

  box = sub_labels.iloc[i, 1:5].values.astype(np.int)
  landmark = sub_labels.iloc[i, 5:].values.astype(np.int)

  cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

  for j in range(5):
    cv2.circle(img, (landmark[j*2], landmark[j*2+1]), 0, (0, 0, 255), 5)
  
  cv2.imwrite(join('picture', str(i)+'.jpg'), img)
"""

#%%
