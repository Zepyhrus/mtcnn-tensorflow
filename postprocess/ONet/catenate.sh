#!/bin/bash
 
cat index.txt | while read line
do
cat $line >> onet_full.prototxt
done