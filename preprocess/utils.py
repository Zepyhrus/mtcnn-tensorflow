import numpy as np
import os
from os.path import join



#%%
def mkdir(dis_path):
  if not os.path.exists(dis_path):
    return os.makedirs(dis_path)

#%%
def iou(box, boxes):
  '''裁剪的box和图片所有人脸box的iou值
  参数：
  box：裁剪的box,当box维度为4时表示box左上右下坐标，维度为5时，最后一维为box的置信度
  boxes：图片所有人脸box,[n,4]
  返回值：
  iou值，[n,]
  '''
  #box面积
  box_area = (box[2]-box[0]+1)*(box[3]-box[1]+1)
  #boxes面积,[n,]
  area = (boxes[:, 2]-boxes[:, 0]+1)*(boxes[:, 3]-boxes[:, 1]+1)
  #重叠部分左上右下坐标
  xx1 = np.maximum(box[0], boxes[:, 0])
  yy1 = np.maximum(box[1], boxes[:, 1])
  xx2 = np.minimum(box[2], boxes[:, 2])
  yy2 = np.minimum(box[3], boxes[:, 3])

  #重叠部分长宽
  w = np.maximum(0, xx2-xx1+1)
  h = np.maximum(0, yy2-yy1+1)
  #重叠部分面积
  inter = w*h
  return inter/(box_area+area-inter+1e-10)


def convert_to_square(box):
  '''将box转换成更大的正方形
  参数：
  box：预测的box,[n,5]
  返回值：
  调整后的正方形box，[n,5]
  '''
  square_box = box.copy()
  h = box[:, 3]-box[:, 1]+1
  w = box[:, 2]-box[:, 0]+1
  #找寻正方形最大边长
  max_side = np.maximum(w, h)

  square_box[:, 0] = box[:, 0]+w*0.5-max_side*0.5
  square_box[:, 1] = box[:, 1]+h*0.5-max_side*0.5
  square_box[:, 2] = square_box[:, 0]+max_side-1
  square_box[:, 3] = square_box[:, 1]+max_side-1
  
  return square_box

#%% get data from wider_face_train_bbx_gt.txt
def read_annotation(base_dir, label_path):
  '''读取文件的image，box'''
  data = dict()
  images = []
  bboxes = []
  labelfile = open(label_path, 'r')
  while True:
    # 图像地址
    imagepath = labelfile.readline().strip('\n')
    if not imagepath:
      break
    imagepath = base_dir + '/images/' + imagepath
    images.append(imagepath)
    # 人脸数目
    nums = int(labelfile.readline().strip('\n'))
    
    if not nums:
      nums = 1

    one_image_bboxes = []
    for i in range(nums):

      bb_info = labelfile.readline().strip('\n').split(' ')
      #人脸框
      face_box = [float(bb_info[i]) for i in range(4)]

      xmin = face_box[0]
      ymin = face_box[1]
      xmax = xmin + face_box[2]
      ymax = ymin + face_box[3]

      one_image_bboxes.append([xmin, ymin, xmax, ymax])

    bboxes.append(one_image_bboxes)

  data['images'] = images
  data['bboxes'] = bboxes
  return data


#%% get data from wider_face_train.txt
def read_anno(base_dir, label_path):
  data = dict()
  bboxes = []
  images = []

  f = open(label_path, 'r')
  while True:
    ln = f.readline()

    if not ln:
      break

    ln = ln.split()

    image = join(base_dir, 'images', ln[0]) + '.jpg'
    box = [int(float(x)) for x in ln[1:]]

    bbox = []

    for i in range(len(ln) // 4):
      bbox.append(box[4*i:4*(i+1)])

    bboxes.append(bbox)
    images.append(image)

  f.close()
  data['images'] = images
  data['bboxes'] = bboxes
  
  return data
