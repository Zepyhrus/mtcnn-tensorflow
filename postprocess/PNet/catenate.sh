#!/bin/bash
 
cat index.txt | while read line
do
cat $line >> pnet_full.prototxt
done
