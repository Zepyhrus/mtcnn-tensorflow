
# coding: utf-8

# In[1]:
import sys
sys.path.extend(['detection', 'train'])

# from detection folder
from MtcnnDetector import MtcnnDetector
from detector import Detector
from fcn_detector import FcnDetector

# from train folder
from model_factory import P_Net, R_Net, O_Net
import config as config


import cv2
import os
from os.path import join, split
import numpy as np
from tqdm import tqdm


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

for filename in tqdm(filenames[:10]):
  iou_threshold = 0.4

  image = start_path + 'img/{}'.format(filename)
  img = cv2.imread(image)

  boxes_det = mtcnn_detector.detect(img)

  xml_file = start_path + 'anno/{}.xml'.\
    format( '.'.join( filename.split('.')[:-1] ) )


#%%
if 0:
  out_path = join('validate', test_mode) + '/'

  if config.input_mode == '1':
    #选用图片
    path = config.test_dir
    print(path)
    for item in tqdm(os.listdir(path)):
      img_path = os.path.join(path, item)
      img = cv2.imread(img_path)
      img_labeled = mtcnn_detector.detect_and_draw(img)

      cv2.imwrite(out_path + item, img_labeled)

  if config.input_mode == '2':
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_path+'out.mp4', fourcc, 10, (640, 480))
    while True:
        t1 = cv2.getTickCount()
        ret, frame = cap.read()
        if ret == True:
          boxes_c, landmarks = mtcnn_detector.detect(frame)
          t2 = cv2.getTickCount()
          t = (t2-t1)/cv2.getTickFrequency()
          fps = 1.0/t
          for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4]
            score = boxes_c[i, 4]
            corpbbox = [int(bbox[0]), int(bbox[1]),
                  int(bbox[2]), int(bbox[3])]

            #画人脸框
            cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                    (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
            #画置信度
            cv2.putText(frame, '{:.2f}'.format(score),
                  (corpbbox[0], corpbbox[1] - 2),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.5, (0, 0, 255), 2)
            #画fps值
          cv2.putText(frame, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
          #画关键点
          for i in range(landmarks.shape[0]):
            for j in range(len(landmarks[i])//2):
              cv2.circle(
                frame, (int(landmarks[i][2*j]), int(int(landmarks[i][2*j+1]))), 2, (0, 0, 255))
          a = out.write(frame)
          cv2.imshow("result", frame)
          if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
          break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
