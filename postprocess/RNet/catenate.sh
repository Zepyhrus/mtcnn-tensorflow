#!/bin/bash
 
cat index.txt | while read line
do
cat $line >> rnet_full.prototxt
done