# In[]
from tqdm import tqdm, tqdm_notebook
import cv2
import pandas as pd
import numpy.random as npr
import numpy as np
import shutil
from os.path import join
import os


import matplotlib.pyplot as plt

import sys
sys.path.append('train')

from model_factory import P_Net, R_Net, O_Net
from detection.detector import Detector
from detection.fcn_detector import FcnDetector
from detection.MtcnnDetector import MtcnnDetector
from preprocess.BBox_utils import getDataFromTxt
from preprocess.utils import mkdir

import time


#%% read celeba data
# bounding boxes calibration for celeba dataset
bboxes_file = 'data/Anno/list_bbox_celeba.txt'
landmarks_file = 'data/Anno/list_landmarks_celeba.txt'

img_dir = 'data/Img/img_celeba'


bboxes = pd.read_csv(bboxes_file, delim_whitespace=True, skiprows=1)
landmarks = pd.read_csv(landmarks_file, delim_whitespace=True, skiprows=1)

sub_labels = bboxes.merge(landmarks)

#%% model initialize
# test_mode = config.test_mode
test_mode = 'ONet'
thresh = [0.6, 0.7, 0.8]
min_face_size = 24
stride = 2
detectors = [None, None, None]

# 模型放置位置
model_path = ['model/PNet/', 'model/RNet/', 'model/ONet']
batch_size = [2048, 256, 32]
PNet = FcnDetector(P_Net, model_path[0])  # detecotors for PNet
detectors[0] = PNet

# in and output path
path = 'picture'
out_path = 'output'

detectors[1] = Detector(R_Net, 24, batch_size[1], model_path[1])
detectors[2] = Detector(O_Net, 48, batch_size[2], model_path[2])

# Use the three detectors to construct a
mtcnn_detector = MtcnnDetector(
  detectors=detectors,
  min_face_size=min_face_size,
  stride=stride, 
  threshold=thresh,
  scale_factor=0.909)

#%%
#%%


def box_of_landmarks(bboxes, landmark):
  """
  pick the box contains all the landmarks in bboxes
  :bboxes: all bboxes, must be shape of (_, 5), sorted descending by probility
  :landmarks: 
  """
  assert landmark.shape == (10,), 'error landmark shape'
  assert (bboxes.shape[0] > 0) & (bboxes.shape[1] == 5), 'error bboxes shape'

  return None


def in_box(landmark, bbox):
  """
  check if the landmark is in the box
  """
  assert landmark.shape == (10,), 'error landmark shape'
  assert bbox.shape == (5, ), 'error bbox shape'

  xs = landmark.reshape(-1, 2)[:, 0]
  ys = landmark.reshape(-1, 2)[:, 1]

  return True if ((xs > bbox[0]).all() & (xs < bbox[2]).all()) &\
      ((ys > bbox[1]).all() & (ys < bbox[3]).all()) else False

#%% generate the landmark file
celeba_landmark_list = open('data/celeba_trainImageList.txt', 'w')
err_idx = 0


for i in tqdm(range(sub_labels.shape[0])):
  src_img = join(img_dir, sub_labels.image_id.iloc[i])
  src_write = join('Img/img_celeba', sub_labels.image_id.iloc[i])
  # des_img = join(out_path, sub_labels.image_id.iloc[i])

  img = cv2.imread(src_img)

  bboxes, landmarks = mtcnn_detector.detect(img)

  label_landmark = sub_labels.iloc[i, -10:].values

  if (bboxes.shape[0] > 0):
    if (in_box(label_landmark, bboxes[0, :])):
      bboxes_list = [str(x) for x in list(bboxes[0, [0, 2, 1, 3]].astype(np.int))]
      landmarks_list = [str(x) for x in list(label_landmark)]

      ln = ' '.join([src_write] + bboxes_list + landmarks_list) + '\n'

      celeba_landmark_list.write(ln)
    else:
      err_idx += 1
      cv2.imwrite(join('error', sub_labels.image_id.iloc[i]), img)
  else:
    err_idx += 1
    cv2.imwrite(join('error', sub_labels.image_id.iloc[i]), img)

print('%d of images has no face detected.' % err_idx)

celeba_landmark_list.close()


#%%
