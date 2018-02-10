# SVM; Histrogram,   oushesh Haradhun 
#Database to be used: ETIS-LaribPolypDB
import os 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np
import cv2
import glob
import time

from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import image
from sklearn.cross_validation import train_test_split
from tempfile import TemporaryFile
from PIL import Image
from skimage import io


# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    hist_features = np.array(img.astype(np.float64))
     # Normalize the result
    norm_features = hist_features / np.sum(hist_features)
    # Return the feature vector
    return norm_features

hist_bins=32 
hist_range=(0, 256)
#Read in Original Images and Labels
dirnameOriginal = 'ETIS-LaribPolypDB/'
original = []
features_data = []
for fname in os.listdir(dirnameOriginal): #Filename of the individual images
    im = Image.open(os.path.join(dirnameOriginal, fname)) # Concateneate the directory name + Filenmae into 1 string
    imarray = np.array(im)
    grey = (imarray - imarray.mean())/imarray.std()
    grey= sum(sum(grey[:,:,0],grey[:,:,1]),grey[:,:,2])
    #print (grey.shape)
    hist_features = color_hist(grey, nbins=hist_bins, bins_range=hist_range)
    features_data.append(hist_features)
    original.append(grey)

features_data =  np.asarray(features_data)
X = np.reshape (features_data, (features_data.shape[0]*features_data.shape[1]*features_data.shape[2],1)) 
print (X.shape)
#print (X)
original = np.asarray(original) 
#print (original.shape) # shape = (2, 966, 1225, 3)

dirnameGT = 'Ground Truth/'
labels = []
labels_features = []
for fname in os.listdir(dirnameGT): #Filename of the individual images
    im = Image.open(os.path.join(dirnameGT, fname)) # Concateneate the directory name + Filenmae into 1 string
    imarray = np.array(im)
    labels.append(imarray)

labels = np.asarray(labels)
#labels = (labels - labels.mean()) / labels.std() #Convert labels to interger and not let in float!!!!!!!!!!!!!!
Y = np.reshape (labels,(labels.shape[0]*labels.shape[1]*labels.shape[2]))

np.save('labels.npy', Y)
np.save('features_data.npy',X) 


#Extract Features
#Build 2 Features Matrix: 0 for nonpolpyp (Background) and 1 for polyp
polyp =original[np.where(labels > 0)]
notpolyp = original[np.where(labels == 0)]

#Use RBF Kernel
clf = svm.SVC()
clf.fit(X, Y)#  reshape x and y to 1 Dimensional Vectors

