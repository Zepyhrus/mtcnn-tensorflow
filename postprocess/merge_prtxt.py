import os
from glob import glob


from caffe.proto import caffe_pb2
import google.protobuf.text_format


model_name = 'postprocess/onet.prototxt'

net = caffe_pb2.NetParameter()
with open(model_name, 'r') as f:
  net = google.protobuf.text_format.Merge(str(f.read()), net)



files = glob('postprocess/*.prototxt')

files = [f for f in files if not f == model_name]



with open(files[0], 'r') as f:
  blob = google.protobuf.text_format.Merge(str(f.read()), net)

print('\nmodel converting finished!')




