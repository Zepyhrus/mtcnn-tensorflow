#%%
import sys
sys.path.extend(['train', 'detection', 'preprocess'])

import numpy as np
import argparse
import os
from os.path import join, split
import pickle
import cv2
from tqdm import tqdm
from utils import read_anno, iou, mkdir, convert_to_square, pick_boxes
from loader import TestLoader
from model_factory import P_Net, R_Net, O_Net
import config as config
from detector import Detector
from fcn_detector import FcnDetector
from MtcnnDetector import MtcnnDetector

# In[2]:
print('='*82)
print('Start generating hard examples...')

def parse_arguments(argv):

  parser = argparse.ArgumentParser()

  parser.add_argument('input_size', type=int,
                      help='The input size for specific net')

  return parser.parse_args(argv)

try:
  args = parse_arguments(sys.argv[1:])
  size = args.input_size
except:
  args = None
  size = 48
# In[3]:
'''通过PNet或RNet生成下一个网络的输入'''
batch_size = config.batches
min_face_size = config.min_face
stride = config.stride
thresh = config.thresh
scale_factor = config.scale_factor
#模型地址
model_path = ['model/PNet/', 'model/RNet/', 'model/ONet']
if size == 12:
  net = 'PNet'
  save_size = 24
elif size == 24:
  net = 'RNet'
  save_size = 48
elif size == 48:
  net = 'SNet'
  save_size = 96

# 图片数据地址
base_dir = 'data/WIDER_train/'
#处理后的图片存放地址
neg_dir = join('data', str(save_size), 'negative')
pos_dir = join('data', str(save_size), 'positive')
part_dir = join('data', str(save_size), 'part')

for dir_path in [neg_dir, pos_dir, part_dir]:
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

detectors = [None, None, None]
PNet = FcnDetector(P_Net, model_path[0], using_cpu=True)
detectors[0] = PNet
if net == 'RNet':
  RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
  detectors[1] = RNet

filename = 'data/wider_face_train_celeba.txt'

# 读取文件的image和box对应函数在utils中
data = read_anno(base_dir, filename)
# using MTCNN Detector to generate all boxes of all images
mtcnn_detector = MtcnnDetector(
  detectors=detectors,
  min_face_size=min_face_size,
  stride=stride,
  threshold=thresh,
  scale_factor=scale_factor)

#%% Generating images from the detection output of PNet

print('开始生成图像...')

im_idx_list = data['images']

gt_boxes_list = data['bboxes']

# save files
neg_label_file = "data/%d/neg_%d.txt" % (save_size, save_size)
neg_file = open(neg_label_file, 'w')

pos_label_file = "data/%d/pos_%d.txt" % (save_size, save_size)
pos_file = open(pos_label_file, 'w')

part_label_file = "data/%d/part_%d.txt" % (save_size, save_size)
part_file = open(part_label_file, 'w')

# this part 
n_idx = 0
p_idx = 0
d_idx = 0
counter = 0

for image, boxes in tqdm(zip(im_idx_list, gt_boxes_list)):
  #TODO: dtype = np.int, origin is np.float32
  boxes = np.array(boxes, dtype=np.int).reshape(-1, 4)
  # print(image)
  img = cv2.imread(image)
  dets, _ = mtcnn_detector.detect(img)
  
  if dets.shape[0] == 0:
    continue  # if no boxes generate ,continue

  dets = convert_to_square(dets)
  dets = dets.astype(np.int)

  neg_num = 0
  pos_num = 0
  part_num = 0

  for det in dets:
    x1, y1, x2, y2, _ = det
    wd = x2 - x1 + 1
    ht = y2 - y1 + 1

    if wd < 24 or ht < 24 or x1 < 0 or y1 < 0 or\
      x2 > img.shape[0] or y2 > img.shape[0]:
      continue
    
    det_iou = iou(det, boxes)
    cropped_im = img[y1:y2 + 1, x1:x2 + 1, :]
    resized_im = cv2.resize(
      cropped_im, (save_size, save_size),
      interpolation=cv2.INTER_LINEAR)

    # TODO: original upper boundary is 60, changed to 40 to balance data examples
    if np.max(det_iou) < 0.3 and neg_num < 75:
      save_file = os.path.join(neg_dir, "%s.jpg" % n_idx)

      neg_file.write(save_file + ' 0\n')
      cv2.imwrite(save_file, resized_im)
      neg_num += 1
      n_idx += 1
    else:
      idx = np.argmax(det_iou)
      assigned_gt = boxes[idx]
      x1_, y1_, x2_, y2_ = assigned_gt

      #偏移量
      offset_x1 = (x1_ - x1) / float(wd)
      offset_y1 = (y1_ - y1) / float(ht)
      offset_x2 = (x2_ - x2) / float(wd)
      offset_y2 = (y2_ - x2) / float(ht)

      # pos和part
      if np.max(det_iou) >= 0.65:
        save_file = os.path.join(pos_dir, "%s.jpg" % p_idx)
        pos_file.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
            offset_x1, offset_y1, offset_x2, offset_y2))
        cv2.imwrite(save_file, resized_im)
        pos_num += 1
        p_idx += 1
      # TODO: original upper boundary is no limit, changed to 40 to balance data examples
      elif np.max(det_iou) >= 0.4:
        save_file = os.path.join(part_dir, "%s.jpg" % d_idx)
        part_file.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
            offset_x1, offset_y1, offset_x2, offset_y2))
        cv2.imwrite(save_file, resized_im)
        part_num += 1
        d_idx += 1

  # if too little postive images are generated from previous net output,
  #   add positive images manually
  if pos_num < neg_num // 3 and (pos_num+1) / (boxes.shape[0]+1) < 1:
    boxes_filtered = boxes[((boxes[:, 2] - boxes[:, 0]) > 24)]
    iou_thresh = 0.65
    cnt = 5 * boxes_filtered.shape[0]

    # try not to disturb 
    try:
      boxes_add = pick_boxes(boxes_filtered, img.shape, iou_thresh, cnt)
      for box in boxes_add:
        cropped_im = img[box[1]:box[3] + 1, box[0]:box[2] + 1, :]
        resized_im = cv2.resize(
            cropped_im, (save_size, save_size),
            interpolation=cv2.INTER_LINEAR)

        save_file = os.path.join(pos_dir, "%s.jpg" % p_idx)
        pos_file.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
            offset_x1, offset_y1, offset_x2, offset_y2))
        cv2.imwrite(save_file, resized_im)
        pos_num += 1
        p_idx += 1
    except:
      print('ERROR: adding additional boxes at ' + image)
      pass

  # print(pos_num, end='\t')
  # print(neg_num, end='\t')
  # print(part_num, end='\t')
  # print(boxes.shape[0])

neg_file.close()
part_file.close()
pos_file.close()




print('%d positive generated.' % p_idx)
print('%d part generated.' % d_idx)
print('%d negative generated.' % n_idx)

print('Generating hard example for %s finished!' % net)
print('='*82)


#%%
