import os
import sys

"""
This scripts transfers names of annotations to txt file
"""

train = open("../dataset/train.txt", "w")
val = open("../dataset/val.txt", "w")

if len(sys.argv)==2:
    val_split = float(sys.argv[1])
else:
    val_split = 0.1
count=0
for file in os.listdir('../dataset/annotations'):
    count+=1
    file = file.split('.')
    file = file[0]+'.'+file[1]
    if count%(1/val_split) == 0:
	    val.write(str(file)+'\n') 
    else:
	    train.write(str(file)+'\n') 
