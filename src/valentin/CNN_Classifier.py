#!/usr/bin/env python
# coding: utf-8

# In[1]:
import keras_segmentation as ks


# In[2]:


# data has been splitted manually between train and validation sets
import sys
IN_COLAB = 'google.colab' in sys.modules
if (IN_COLAB):
    from google.colab import drive
    drive.mount('/content/gdrive')
    PATH_OF_DATA= '/content/gdrive/"My Drive"/Documents/EPFL/ML_Project_2/data'
else:
    PATH_OF_DATA= './data'
IMAGES_SUFFIX='images'
GD_SUFFIX='groundtruth'
TRAIN_PREFIX='train'
VALIDATION_PREFIX='val'
PREPPED='prepped'


# In[3]:


#model = ks.models.unet.vgg_unet(n_classes=2, input_height=400, input_width=400) #TODO: try to use builtin model with cropped images
from keras.models import *
from keras.layers import *
n_classes = 2
input_width = 400
input_height = input_width

img_input = Input(shape=(input_height, input_width, 3))

conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
conv1 = Dropout(0.2)(conv1)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D((2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Dropout(0.2)(conv2)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D((2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Dropout(0.2)(conv3)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

up1 = concatenate([UpSampling2D((2, 2))(conv3), conv2], axis=-1)
conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
conv4 = Dropout(0.2)(conv4)
conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

up2 = concatenate([UpSampling2D((2, 2))(conv4), conv1], axis=-1)
conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
conv5 = Dropout(0.2)(conv5)
conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

out = Conv2D(n_classes, (1, 1), padding='same')(conv5)

from keras_segmentation.models.model_utils import get_segmentation_model

model = get_segmentation_model(img_input ,  out) # this would build the segmentation model


# In[5]:


model.train( 
    train_images =  f'{PATH_OF_DATA}/{TRAIN_PREFIX}_{PREPPED}_{IMAGES_SUFFIX}',
    train_annotations = f'{PATH_OF_DATA}/{TRAIN_PREFIX}_{PREPPED}_{GD_SUFFIX}',
    checkpoints_path = f'{PATH_OF_DATA}/vgg_unet_1' , epochs=5
)


# In[ ]:




