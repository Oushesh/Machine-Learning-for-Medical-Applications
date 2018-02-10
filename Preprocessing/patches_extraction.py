#Oushesh Haradhun 
#Database to be used: ETIS-LaribPolypDB Extract Patches  161x245
#Image is of size 1225 x 966 x 3  
import os 

import numpy as np
import glob

from PIL import Image
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.cross_validation import train_test_split

# Define a function to compute color histogram features  
hist_bins=32 
hist_range=(0, 256)
def color_hist(img, nbins=32, bins_range=(0, 256)):
    hist_features = np.array(img.astype(np.float64))
     # Normalize the result
    norm_features = hist_features / np.sum(hist_features)
    # Return the feature vector
    return norm_features

#Read in the original Data
dirnameOriginal = 'ETIS-LaribPolypDB/'
#labels
dirnameGT = 'Ground Truth/'
labels = []
labels_features = []
original = []
features_data = []

for fname in os.listdir(dirnameOriginal): #Filename of the individual images
    im = Image.open(os.path.join(dirnameOriginal, fname)) # Concateneate the directory name + Filenmae into 1 string
    imarray = np.array(im)
    grey = (imarray - imarray.mean())/imarray.std()
    grey= sum(sum(grey[:,:,0],grey[:,:,1]),grey[:,:,2])
    histogram = color_hist(grey)
    original.append(grey)    

original =  np.asarray(original)
print (original.shape)

for fname in os.listdir(dirnameGT): #Filename of the individual images
    im = Image.open(os.path.join(dirnameGT, fname)) # Concateneate the directory name + Filenmae into 1 string
    imarray = np.array(im)
    labels.append(imarray)

labels = np.asarray(labels)
number_images=labels.shape[0]
X_labels=np.zeros((labels.shape[1],labels.shape[2]))

#window =(224,224)
window = (100,100) #for VGG_CNN network
#to test with VGG_CNN we use a window of (100,100) to fit the images into the network
number_images=original.shape[0]
X=np.zeros((original.shape[1], original.shape[2]))
for idx_images in range(number_images):
    extension = '.npy'
    X = np.reshape (original[idx_images,:],(original.shape[1], original.shape[2]))
    X_labels = np.reshape (labels[idx_images,:],(labels.shape[1], labels.shape[2]))
    #print(X.shape)
    #Image_patches = extract_patches_2d(X,window,30) #Transform the whole into patches of 300x300
    Image_patches = extract_patches_2d(X,window,108)  #For VGG Net we need 100*100 pixel -> 12.25 * 9.66 patches approx. 108 patches 
    Label_patches = extract_patches_2d(X_labels,window,108) 
    #print (Image_patches.shape)
    for idx_patches in range(Image_patches.shape[0]): #iterate through all the number of patches in 1 image(should be around 30
        if np.sum(Label_patches[idx_patches,:])>0:
            np.save('histpatches' +str(idx_images) + '_' + str(idx_patches)  + extension,Image_patches[idx_patches,:]) #save with new name 'string' + suffix
            np.save('histlabels' +str(idx_images) + '_' + str(idx_patches)  + extension,Label_patches[idx_patches,:]) #save with new name 'string' + suffix
        else:
            idx_patches+=1

