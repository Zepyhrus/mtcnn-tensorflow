import sys
sys.path.extend(['prepare_data', 'train_models', 'detection'])

from os.path import join, split

import pandas as pd

import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

plt.rcParamsDefault['figure.autolayout'] = True

#%%
# validation training examples on rnet
label_file = 'images/48/landmark_48_aug.txt'
images_file = 'images/48/train_ONet_landmark_aug'


labels = pd.read_csv(label_file, sep=' ', header=None)




#%%
lsize = 5  # landmark size

for i in range(labels.shape[0]):
  image = cv2.imread(labels.iloc[i, 0])
  
  landmarks = labels.iloc[i, -2*lsize:].astype(float).values
  ht, wd, dp = image.shape
  
  for j in range(lsize):
#    print((int(landmarks[2*j]*ht), int(landmarks[2*j+1]*wd)))
    cv2.circle(image,
               (int(landmarks[2*j]*ht), int(landmarks[2*j+1]*wd)),
               1, (0, 255, 0), -1)
  
  
  cv2.imwrite(join('val', split(labels.iloc[i, 0])[-1]), image)
  
  if i % 200 == 0:
    print('%d out of %d saved.' % (i, labels.shape[0]))
#  plt.imshow(image[:, ::-1])







