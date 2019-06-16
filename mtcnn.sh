# training pnet
python preprocess/gen_12net_data.py
python preprocess/gen_landmark_aug.py 12
python preprocess/gen_imglist_pnet.py
python preprocess/gen_tfrecords.py 12
python train/train_models.py 12

# training onet
python preprocess/gen_hard_example.py 12
python preprocess/gen_landmark_aug 24
python preprocess/gen_tfrecords 24
python train/train_models.py 24

# training rnet
python preprocess/gen_hard_example.py 24
python preprocess/gen_landmark_aug 48
python preprocess/gen_tfrecords 48
python train/train_models.py 48


# test
python test.py