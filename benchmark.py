
# coding: utf-8

# In[1]:
import sys

# from detection folder
from detection.MtcnnDetector import MtcnnDetector
from detection.detector import Detector
from detection.fcn_detector import FcnDetector

# from train folder
from train.model_factory import P_Net, R_Net, O_Net
import train.config as config
from preprocess.utils import iou

import cv2
import os
from os.path import join, split
import numpy as np
from tqdm import tqdm
#%% 
def boxes_extract(xml_file):
  """
    extract boxes from given xml file
    @xml_file: input xml file name
  """
  import xml.etree.ElementTree as et
  if not os.path.isfile(xml_file):
    return None
  else:
    root = et.parse(xml_file).getroot()
    def box_extract(x): return [int(t.text) for t in x[-1]]
    boxes = [box_extract(x) for x in root[6:]]

    return np.asarray(boxes)

# In[ ]:
# test_mode = config.test_mode
test_mode = 'ONet'
thresh = [0.6, 0.7, 0.9]
min_face_size = 24
stride = 2
detectors = [None, None, None]

scale_factor = 0.79

# 模型放置位置
model_path = ['model/PNet/', 'model/RNet/', 'model/ONet']
batch_size = config.batches

detectors[0] = FcnDetector(P_Net, model_path[0])  # detecotors for PNet
if test_mode in ['RNet', 'ONet']:
  detectors[1] = Detector(R_Net, 24, batch_size[1], model_path[1])

  if test_mode == 'ONet':
    detectors[2] = Detector(O_Net, 48, batch_size[2], model_path[2])

# Use the three detectors to construct a 
mtcnn_detector = MtcnnDetector(
  detectors=detectors,
  min_face_size=min_face_size,
  stride=stride,
  threshold=thresh,
  scale_factor=scale_factor)


start_path = '/home/sherk/Workspace/mtcnn-pytorch/'
filenames = os.listdir(start_path + 'img')

missing_detection = 0
false_detection = 0
all_detection = 0
all_labels = 0

for filename in tqdm(filenames):
  iou_threshold = 0.4

  image = start_path + 'img/{}'.format(filename)
  img = cv2.imread(image)

  boxes_det, _ = mtcnn_detector.detect(img)
  boxes_det = boxes_det.astype(np.int)


  xml_file = start_path + 'anno/{}.xml'.\
    format( '.'.join( filename.split('.')[:-1] ) )
  boxes_lab = boxes_extract(xml_file)

  if boxes_lab is None:
    if boxes_det is not None:
      false_detection += len(boxes_det)
    continue

  if boxes_det.shape[0] == 0:
    if boxes_lab is not None:
      missing_detection += len(boxes_lab)
    continue

  for box in boxes_lab:
    if max(iou(box, boxes_det)) < iou_threshold:
      missing_detection += 1
      # Blue stands for missings
      cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    # Green is from label
    # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

  for box in boxes_det:
    if max(iou(box, boxes_lab)) < iou_threshold:
      false_detection += 1
      # red stands for false detection
      cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    # Red is from detector
    # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

  cv2.imwrite('output/{}'.format(filename), img)
  all_detection += len(boxes_det)
  all_labels += len(boxes_lab)

print('Detect\tMissing\tFalse\tAll')
print('{}\t{}\t{}\t{}'.format(all_detection,
                              missing_detection, false_detection, all_labels))

precision = 1 - false_detection / all_detection
print('Precision: {}'.format(round(precision, 4)))

recall = 1 - missing_detection / all_labels
print('Recall: {}'.format(round(recall, 4)))

f1_score = 2 * precision * recall / (precision + recall)
print('F1 score: {}'.format(round(f1_score, 4)))


  
