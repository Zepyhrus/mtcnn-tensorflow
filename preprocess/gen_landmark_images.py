# In[]
from BBox_utils import getDataFromTxt, BBox
from utils import iou, mkdir
from tqdm import tqdm
import argparse
import os
from os.path import join, split
import random
import sys
import cv2
import numpy as np
npr = np.random
data_dir = '../data'

#%%
def flip(face, landmark):
  #镜像
  face_flipped_by_x = cv2.flip(face, 1)
  landmark_ = np.asarray([(1-x, y) for (x, y) in landmark])
  landmark_[[0, 1]] = landmark_[[1, 0]]
  landmark_[[3, 4]] = landmark_[[4, 3]]
  return (face_flipped_by_x, landmark_)


# In[5]: rotaition augumentation
def rotate(img, box, landmark, alpha):
  #旋转
  center = ((box.left+box.right)/2, (box.top+box.bottom)/2)
  #TODO: What is cv2.getRotationMatrix2D(center, alpha, 1)
  rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
  #TODO: What is cv2.warpAffine?
  img_rotated_by_alpha = cv2.warpAffine(
    img, rot_mat, (img.shape[1], img.shape[0]))
  landmark_ = np.asarray([
    (rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
    rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmark])
  face = img_rotated_by_alpha[box.top:box.bottom+1, box.left:box.right+1]
  return (face, landmark_)

#%% parse arguments
def parse_arguments(argv):
  parser = argparse.ArgumentParser()

  parser.add_argument(
    'input_size', typu=int,
    help='The input size of the Net'
  )

  return parser.parse_args(argv)

#%% generate landmark images for PNet
size = 12  # the size of PNet input
argument = True  # argumentation for images
net = 'PNet'

image_id = 0
OUTPUT = join(data_dir, str(size))
mkdir(OUTPUT)

dstdir = join(OUTPUT, 'train_%s_landmark_aug' % net)
mkdir(dstdir)

# trainImageList.txt for lfw dataset
# data/Anno/list_landmarks_celeba.txt

ftxt = join(data_dir, 'Anno/trainImageList.txt')
f = open(os.path.join(OUTPUT, 'landmark_%d_aug.txt' % (size)), 'w')

