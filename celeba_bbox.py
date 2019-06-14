"""
This script is used to generate bounding boxes from the .xml labels
Original images comes from CelebA, which have not been recognized 
  by the model trained, around 1200 manually labeled images
"""

#%%
import os
from glob import glob
from os.path import join, split

import xml.etree.ElementTree as ET

#%% annotation files
xml_files = glob('anno/*.xml')

#%% bounding box file
bbox_file = 'data/wider_face_celeba_train.txt'

with open(bbox_file, 'a') as bf:
  for xml_file in xml_files:
    tree = ET.parse(xml_file)

    root = tree.getroot()

    bf.write(join(root[0].text, root[1].text.replace('.jpg', '')))

    bboxes = []

    for rt in root[6:]:
      for cd in rt[-1]:
        bboxes.append(cd.text)
    
    bf.write(' ' + ' '.join(bboxes) + '\n')



#%%
