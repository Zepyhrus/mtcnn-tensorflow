"""
Restore ckpt from meta file and JUST display the shape/name of all the layers
"""

import tensorflow as tf
import numpy as np



config = tf.ConfigProto(
  log_device_placement=True,
  allow_soft_placement=True
)

with tf.Session(config=config) as sess:
  new_saver = tf.train.import_meta_graph(
      '../save_model/seperate/onet/onet-850000.meta')  # load graph
  for var in tf.trainable_variables(): #get the param names
    print(var.name) #print parameters' names
    
    new_saver.restore(sess, tf.train.latest_checkpoint(
        '../save_model/seperate/onet/'))  # find the newest training result
    all_vars = tf.trainable_variables()
    for v in all_vars:
      v_4d = np.array(sess.run(v)) #get the real parameters
      print(v_4d.shape)

print("finished!")
