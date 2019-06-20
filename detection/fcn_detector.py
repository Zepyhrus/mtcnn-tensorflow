
# coding: utf-8

# In[1]:
import tensorflow as tf


# In[2]:


class FcnDetector:
  '''识别单张图片'''

  def __init__(self, net_factory, model_path, using_cpu=False):
    graph = tf.Graph()
    with graph.as_default():
      self.image_op = tf.placeholder(tf.float32, name='input_image')
      self.width_op = tf.placeholder(tf.int32, name='image_width')
      self.height_op = tf.placeholder(tf.int32, name='image_height')
      image_reshape = tf.reshape(
        self.image_op, [1, self.height_op, self.width_op, 3])
      
      # 预测值
      self.cls_prob, self.bbox_pred, _ = net_factory(
        image_reshape, training=False)
      if using_cpu:
        config = tf.ConfigProto(
          device_count={'GPU': 0},
          allow_soft_placement=True
        )

        self.sess = tf.Session(config=config)
      else:
        self.sess = tf.Session()
      
      # 重载模型
      saver = tf.train.Saver()
      model_file = tf.train.latest_checkpoint(model_path)
      saver.restore(self.sess, model_file)

  def predict(self, databatch):
    height, width, _ = databatch.shape
    cls_prob, bbox_pred = self.sess.run([self.cls_prob, self.bbox_pred],
                      feed_dict={self.image_op: databatch,
                             self.width_op: width,
                             self.height_op: height})

    return cls_prob, bbox_pred
