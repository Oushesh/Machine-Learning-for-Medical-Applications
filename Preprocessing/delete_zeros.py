#Oushesh Haradhun
#Program to automatically delete patches which contain only zeros 
#to avoid high class imbalancing

import os 

import numpy as np
import glob

from PIL import Image
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.cross_validation import train_test_split


filelist=[]   
#print(glob.glob("PATCHES/label_patches_balanced/*.npy")) #Print all files with the extension .npy in the given folder we want,  
filelist.append(glob.glob("PATCHES/label_patches_balanced/*.npy"))

for idx_files in range(len(filelist)):
    im = Image.open(filelist[idx_files])
    imarray = np.array(im)
    if sum(imarray)==0: #if sum is zero then all elements are 0s.we have also only non-ve values
        np.delete(imarray)   
    


