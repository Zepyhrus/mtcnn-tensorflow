#%%
import os
from os.path import join, split
import tensorflow as tf
import numpy as np


config = tf.ConfigProto(
    log_device_placement=False,
    # allow soft placement
    allow_soft_placement=True
)


net = 'ONet'

ckpt_file = join('model', net)  # model/PNet
with open(join(ckpt_file, 'checkpoint')) as f:  # model/PNet/checkpoint
  # PNet-30.meta
  latest_meta = f.readline().split()[-1].replace('"', '') + '.meta'

meta_file = join(ckpt_file, latest_meta)  # model/PNet/PNet-30.meta

#%%

with tf.Session(config=config) as sess:
  new_saver = tf.train.import_meta_graph(meta_file)
  for var in tf.trainable_variables():
    print(var.name)
  new_saver.restore(sess, tf.train.latest_checkpoint(ckpt_file))
  all_vars = tf.trainable_variables()
  for v in all_vars:
    name = v.name
    fname = name + '.prototxt'
    fname = join('postprocess', net, fname.replace('/', '_'))  # PNet/...
    print(fname)
    v_4d = np.array(sess.run(v))

    with open(fname, 'w') as f:
      if v_4d.ndim == 4:
        #v_4d.shape [ H, W, I, O ]
        v_4d = np.swapaxes(v_4d, 0, 2)  # swap H, I
        v_4d = np.swapaxes(v_4d, 1, 3)  # swap W, O
        v_4d = np.swapaxes(v_4d, 0, 1)  # swap I, O
        #v_4d.shape [ O, I, H, W ]
        # f = open(fname, 'w+')
        vshape = v_4d.shape[:]
        v_1d = v_4d.reshape(v_4d.shape[0]*v_4d.shape[1]*v_4d.shape[2]*v_4d.shape[3])
        f.write('  blobs {\n')
        
        for vv in v_1d:
          f.write('    data: %8f' % vv)
          f.write('\n')
        f.write('    shape {\n')
        for s in vshape:
          f.write('      dim: ' + str(s))  # print dims
          f.write('\n')
        f.write('    }\n')
        f.write('  }\n')
      # Added by Sherk
      #   During full connection layers, the number of dimension of the tesnor
      #   will be 2 rather than 4 or 1, so this would cause blob missing
      elif v_4d.ndim == 2:
        # for full connection layers
        # v_4d.shape [ I, O ]
        v_4d = np.swapaxes(v_4d, 0, 1)  # swap I, O
        # v_4d.shape [ O, I ]
        # f = open(fname, 'w+')
        vshape = v_4d.shape[:]
        v_1d = v_4d.reshape(v_4d.shape[0]*v_4d.shape[1]) # flatten the array
        f.write('  blobs {\n')
        for vv in v_1d:
          f.write('    data: %8f' % vv)
          f.write('\n')
        f.write('    shape {\n')
        for s in vshape:
          f.write('      dim: ' + str(s))  # print dims
          f.write('\n')
        f.write('    }\n')
        f.write('  }\n')
        pass
      elif v_4d.ndim == 1:  # do not swap
        # f = open(fname, 'w+')
        f.write('  blobs {\n')
        for vv in v_4d:
          f.write('    data: %.8f' % vv)
          f.write('\n')
        f.write('    shape {\n')
        f.write('      dim: ' + str(v_4d.shape[0]))  # print dims
        f.write('\n')
        f.write('    }\n')
        f.write('  }\n')
    
    # f.close()


#%%
