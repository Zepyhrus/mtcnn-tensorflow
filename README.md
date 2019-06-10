## Description
This work is a tensorflow reproduction of MTCNN, which originally derived from https://github.com/AITTSMD/MTCNN-Tensorflow, a Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks.
Modified by Sherk.

## Prerequisites
1. You need CUDA-compatible GPUs to train the model.
2. You should first download [WIDER Face](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) and [Celeba](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).**WIDER Face** for face detection and **Celeba** for landmark detection(This is required by original paper.But I found some labels were wrong in Celeba. So I use [this dataset](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm) for landmark detection).

## Dependencies
* Tensorflow 1.12.0
* TF-Slim
* Python 3.6.8 Anacoda
* Ubuntu 18.04
* Cuda 9.0

## Models
All models are derived from folder `train_models`. Generally
* MTCNN_config.py:
  - Few simple network configurations;
* mtcnn_model.py:
	- The defination of the model;
* train.py:
	- The original training module;
* train_xxnet.py:
	- Train the 3 networks separately;

1. MTCNN_config.py:
* BATCH_SIZE = 384 
* CLS_OHEM = True
* CLS_OHEM_RATIO = 0.7
* BBOX_OHEM = False
* BBOX_OHEM_RATIO = 0.7
* EPS = 1e-14
* LR_EPOCH = [6,14,20]

2. mtcnn_model.py:
All the net_factory (including P_Net/R_Net/O_Net) comes from this file
* prelu(inputs): prelu activation layer, usually comes after conv layer;
* dense_to_one_hot(labels_dense, num_classes): One-hot encoder. [Question]When and where is it used? **NOT used**
* cls_ohem(cls_prob, label): **Classification Online Hard Exapmle Mining** 


## Detectors
**The model file is saved PER EPOCH!** [Question]Where is it defined?

All detectors are derived from folder `detection`. Generally: 
* detector.py
  - Define **rnet** and **onet** face detectors;
* fcn_detector.py
  - Define **pnet** face detectors;
* MtcnnDetecor.py
  - Combine the 3 detectors;

1. detector.py:
* takes at maximum 0.3 fraction of the GPU memory and disallow growth;


2. fcn_detector.py:
2.1 __init__(self, net_factory, model_path):
  1) create default graph;
  2) create placeholder for images;
  3) 





## Project description
### Trainig PNet 
1. Download Wider Face Training part only from Official Website , unzip to replace `WIDER_train` and put it into `prepare_data` folder.
2. Download landmark training data from [here]((http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm)),unzip and put them into `prepare_data` folder.
3. Run `gen_12net_data.py` to generate training data(Face Detection Part) for **PNet**.
  (1) generate around: A LOT of images;
  (2) take inputs from ground truth file: `prepare_data/wider_face_train.txt` and images from `images/WIDER_train/images`;
  (3) generated images are saved to `images/12/*` folders, `*` refers to `negative/part/positive`;
  (4) around: 
4. Run `gen_landmark_aug_12.py` to generate training data(Face Landmark Detection Part) for **PNet**.
  (1) take input label contianing all the landmarkd of the training data from `images/trainImageList.txt` and images from `images/lfw_5590` folder;
  (2) only around 5590 images available, generated 178443 images;
  (3) save generated images and labels to `images/12/train_PNet_landmark_aug` folder and `images/12/landmark_12_aug.txt` file;
  (4) example records: images/12/train_PNet_landmark_aug/180201.jpg -2 lm0~9;
5. Run `gen_imglist_pnet.py` to merge two parts of training data.
  (1) take all the .txts generated from above and listed in folder `images/12` and resample them to write them into `images/tfrecords/train_PNet_landmark.txt` file for future .tfrecords generating;
6. Run `gen_PNet_tfrecords.py` to generate tfrecord for **PNet**.
  (1) use `train_models/train_PNet.py` to train **PNet**;
  (2) take the label files from `images/tfrecords/train_PNet_landmark.txt` and save the shuffle tfrecords to `images/tfrecords/train_PNet_landmark.tfrecord_shuffle`;
  (3) a 30 epoches of training will be much enough;

### Training RNet
7. After training **PNet**, run `gen_hard_example.py  --test_mode PNet` to generate training data(Face Detection Part) for **RNet**.
  * take great care of `gen_hard_example.py` file;
    (1) images saved to `images/24`, `negative\part\positive` folders, and also label files to `neg_24\part_24\pos_24` .txt files;
  * set the net variable in the scripts to be `RNet` for generating training data for **RNet**;
  * There are 3 ##TODO flags in gen_hard_example.py;
  * The MtcnnDetector.detect_face takes great time, approximately 1.4s per image;
  * [Question]How is the threshold used?
8. Run `gen_landmark_aug_24.py` to generate training data(Face Landmark Detection Part) for **RNet**.
  (1) Defunctionalized. Take label file from `prepare_data/trainImageList.txt` and image from `images/lfw5590`;
  (2) images saved to `images/24/train_RNet_landmark_aug` folder;
  (3) labels (landmarks) file saved to `images/24/landmark_24_aug.txt` file;
  (4) "images/24/train_RNet_landmark_aug/179354.jpg -2 lm0 lm1 lm2 lm3 lm4 lm5 lm6 lm7 lm8 lm9"
  (5) around 180k landmark images generated;
9. Run `prepare_data/gen_imglist_rnet.py` to merge two parts of training data.
  (1) get no landmark images and labels from folder `images/no_LM24`, which contains:
    * pos_24.txt/neg_24.txt/part_24.txt: 3 .txt files for labels;
    * pos/neg/part: 3 folders for images;
    * which all these data are generated from `gen_hard_example.py` scripts;
  (2) take landmark image inputs and labels from `images/24`;
  (3) output to `images/imglists_noLM/RNet` folder, `train_RNet_landmark.txt` file;
  (4) using 77221 neg, 133122 pos, 740631 part, 179357 landmark;
10. Run `gen_RNet_tfrecords.py` to generate tfrecords for **RNet**.(**you should run this script four times to generate tfrecords of neg, pos, part and landmark respectively**)
  (1) change the name in the script (pos/neg/part/landmark) pespectively;
  (2) takes input from: `images/no_LM24/`
  (3) save tfrecords to `images/imglists_noLM/RNet/neg_landmark.tfrecord`;
  (4) For landmark, it takes input from: `images/24/train_RNet_landmark_aug` and `images/24/landmark_24_aug.txt`
  (5) use `train_models/train_RNet.py` to train **RNet** and save model to `data/MTCNN_model/PNet_landamrk/PNet` folder;
  (6) 30 epoches of training will be significant enough;

### Training ONet
11. After training **RNet**, run `prepare_data/gen_hard_example.py --test_mode RNet` to generate training data(Face Detection Part) for **ONet**.
  * the `gen_hard_example.py` is used twice, to generate pos/neg/part samples, WITHOUT landmarks from WIDER FACE dataset;
  * takes the input from `images/WIDER_train`;
  * restore the model from epoch [30, 30, 16], from `data/MTCNN_model/*Net_landmark/*` folders;
  * write the result labels to `images/no_LM48/*_48.txt` files;
  * write the result images to `images/no_LM48/*_48` folders;
  * use "--epoch" to use the latest model;
  * [Question]What is the stride used for?
12. Run `gen_landmark_aug_48.py` to generate training data(Face Landmark Detection Part) for **ONet**.
  * takes input images from `images/lfw_5590`, used to generate faces with landmarks from FDDB dataset;
  * takes input labels from `images/trainImageLists.txt`;
  * write output images to `images/train_ONet_landmark_aug` folder;
  * write output labels to `images/landmark_48_aug.txt` file;
13. Run `gen_imglist_onet.py` to merge two parts of training data.
  * merge the  4 label files generated from `gen_hard_example` and `gen_landmark_aug_48`;
  * save it to `images/imglists_noLM/ONet` for future .tfrecords generating;
14. Run `gen_ONet_tfrecords.py` to generate tfrecords for **ONet**.(**you should run this script four times to generate tfrecords of neg,pos,part and landmark respectively**)
  * There are 2 places need to be modified;
  * **item** in get_dataset function, redirect to the label .txt file;
  * **_get_output** function's returned .tfrecord name;
  * use `train_models/train_ONet.py` to train the ONet, a 30 epoches of the training will be much enough;


## Some Details
* You can use CUDA_VISIBLE_DEVICES="" as prefix to your command to run session on the CPU, to make full use of your computing resources;
* When training **PNet**,I merge four parts of data(pos,part,landmark,neg) into one tfrecord,since their total number radio is almost 1:1:1:3.But when training **RNet** and **ONet**,I generate four tfrecords,since their total number is not balanced.During training,I read 64 samples from pos,part and landmark tfrecord and read 192 samples from neg tfrecord to construct mini-batch.
* It's important for **PNet** and **RNet** to keep high recall radio.When using well-trained **PNet** to generate training data for **RNet**,I can get 14w+ pos samples.When using well-trained **RNet** to generate training data for **ONet**,I can get 19w+ pos samples.
* Since **MTCNN** is a Multi-task Network, we should pay attention to the format of training data.The format is:
 
  [path to image][cls_label][bbox_label][landmark_label]
  
  For pos sample,cls_label=1,bbox_label(calculate),landmark_label=[0,0,0,0,0,0,0,0,0,0].

  For part sample,cls_label=-1,bbox_label(calculate),landmark_label=[0,0,0,0,0,0,0,0,0,0].
  
  For landmark sample,cls_label=-2,bbox_label=[0,0,0,0],landmark_label(calculate).  
  
  For neg sample,cls_label=0,bbox_label=[0,0,0,0],landmark_label=[0,0,0,0,0,0,0,0,0,0].  

* Since the training data for landmark is less.I use transform,random rotate and random flip to conduct data augment (the result of landmark detection is not that good).


## Validation
1. val_onet.py: Take inputs and validat the landmark training datasets


## test
1. one_image_test.py


## Result

![result1.png](https://i.loli.net/2017/08/30/59a6b65b3f5e1.png)

![result2.png](https://i.loli.net/2017/08/30/59a6b6b4efcb1.png)

![result3.png](https://i.loli.net/2017/08/30/59a6b6f7c144d.png)

![reult4.png](https://i.loli.net/2017/08/30/59a6b72b38b09.png)

![result5.png](https://i.loli.net/2017/08/30/59a6b76445344.png)

![result6.png](https://i.loli.net/2017/08/30/59a6b79d5b9c7.png)

![result7.png](https://i.loli.net/2017/08/30/59a6b7d82b97c.png)

![result8.png](https://i.loli.net/2017/08/30/59a6b7ffad3e2.png)

![result9.png](https://i.loli.net/2017/08/30/59a6b843db715.png)

**Result on FDDB**
![result10.png](https://i.loli.net/2017/08/30/59a6b875f1792.png)

## License
MIT LICENSE

## References
1. Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, Yu Qiao , " Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks," IEEE Signal Processing Letter
2. [MTCNN-MXNET](https://github.com/Seanlinx/mtcnn)
3. [MTCNN-CAFFE](https://github.com/CongWeilin/mtcnn-caffe)
4. [deep-landmark](https://github.com/luoyetx/deep-landmark)
