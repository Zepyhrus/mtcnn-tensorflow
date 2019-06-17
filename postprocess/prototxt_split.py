import os
from glob import glob
import re

"""
  This script is used to split original .prototxt file (which does not contain
    blob data) into multiple .prototxt files according to the layer defination
"""

ptfile = '48net.prototxt'

with open(ptfile, 'r') as f:
  for ln in f:
    if ln.endswith('}\n') or ln.endswith('{\n'):
      print(ln, end='')

# Strip blobs' data from _full.prototxt created from .caffemodel
md_in_file = '12net_full.prototxt'
md_out_file = '12net_strip.prototxt'

switch = 1

with open(md_in_file, 'r') as fin, open(md_out_file, 'w') as fout:
  for i, ln in enumerate(fin):
    if not ln.strip().startswith('data:'):
      fout.write(ln)
    
    if i % 10000 == 0:
      print(i)

print('Rewrite done!')


md_in_file = '12net.prototxt'

j = 0
with open(md_in_file, 'r') as fin:
  fout = open(str(j)+'.prototxt', 'w')

  for ln in fin:
    fout.write(ln)
    if re.search(r'top: "conv.*"\n$', ln):
      fout.close()
      j += 1
      fout = open(str(j)+'.prototxt', 'w')
  
  fout.close()