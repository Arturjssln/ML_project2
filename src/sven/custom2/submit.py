#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


MODEL_V3 = {
    "path": 'training/V3.001.h5',
    "batchnorm": True,
    "residual": True,
}

MODEL_TO_IMPORT = MODEL_V3 # can be either a model (dict) or a string

DEFAULT = {
    "mode": 'rgb',
    "seed": 1000,
    "lk_alpha": .1,
    "channels_size": 3,
    "batchnorm": False,
    "residual": False,
}
IMAGE_SIZE = 608
PATCH_SIZE = 16
PTHRESHOLD = 0.25
NB_TST_IMG = 50


# In[ ]:


import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
import re


# # Load Model

# In[ ]:


def get_model_prop(prop_name):
    if isinstance(MODEL_TO_IMPORT, str) or prop_name not in MODEL_TO_IMPORT:
        return DEFAULT[prop_name]
    return MODEL_TO_IMPORT[prop_name]


# In[ ]:


from CNNv2 import CNN

model = CNN(
    rootdir='.',
    window_size=IMAGE_SIZE,
    lk_alpha=get_model_prop('lk_alpha'),
    random_seed=get_model_prop('seed'),
    channels_size=get_model_prop('channels_size'),
    batchnorm=get_model_prop('batchnorm'),
    residual=get_model_prop('residual'),
)
model.load(MODEL_TO_IMPORT if isinstance(MODEL_TO_IMPORT, str) else MODEL_TO_IMPORT['path'])
test_dir = "test_set_images/test_"


# # Visualization 

# In[ ]:


from helpers import *
from skimage.color import rgb2hsv, rgb2lab, rgb2hed, rgb2yuv

def load_image(filename, mode = 'rgb'):
    if mode == 'hsv':
        img = rgb2hsv(mpimg.imread(filename))
    elif mode == 'lab':
        img = rgb2lab(mpimg.imread(filename))
    elif mode == 'hed':
        img = rgb2hed(mpimg.imread(filename))
    elif mode == 'yuv':
        img = rgb2yuv(mpimg.imread(filename))
    elif mode == 'rgb':
        img = mpimg.imread(filename)
    else:
        raise NotImplemented
    return np.expand_dims(img, axis=0)

def get_path_for_img_nb(img_nb):
    return 'test_set_images/test_'+str(img_nb)+'/test_' + str(img_nb) + '.png'

def get_image_filenames(img_nb = None):
    image_filenames = []
    if img_nb == None:
        for i in range(1, NB_TST_IMG+1):
            image_filenames += [get_path_for_img_nb(i)]
    elif type(img_nb) is int:
        image_filenames += [get_path_for_img_nb(img_nb)]
    else:
        for i in img_nb:
            image_filenames += [get_path_for_img_nb(i)]
    return image_filenames
    
def visualize_step(idx, input_image, Xi_raw, Xi, ground, animate = False):
    input_image = np.squeeze(input_image)
    fig, axs = plt.subplots(1, 4, figsize=(16, 16))
    axs[0].imshow(input_image)
    #axs[0].set_title(f'image {idx+1}')
    axs[1].imshow(np.squeeze(Xi_raw))
    axs[1].set_title('real prediction')
    axs[2].imshow(np.squeeze(Xi))
    axs[2].set_title('thresholded prediction')
    axs[3].imshow(ground)
    axs[3].set_title('label prediction')
    # remove the x and y ticks
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    #display.clear_output(wait=True)
    if animate:
        plt.show()

def get_predicted_mask(image_filename, mode):
    input_image = load_image(image_filename, mode)
    Xi_raw = model.model.predict(input_image)
    Xi = np.where(Xi_raw>0.5, 1, 0)
    Xi = np.squeeze(Xi)
    return Xi_raw, Xi

def visualize(img_nb = None, mode = 'rgb', save_masks = False):
    image_filenames = get_image_filenames(img_nb)
    for i, filename in enumerate(image_filenames[0:]):
        img_input = load_image(filename, mode)
        Xi_raw, Xi = get_predicted_mask(filename, mode)
        ground = get_ground_img(Xi, patch_size = PATCH_SIZE, foreground_threshold = PTHRESHOLD)
        visualize_step(i, img_input, Xi_raw, Xi, ground)
        #if save_masks:
            #mpimg.imsave(os.path.join('./masks', f'test_{i+1}.png'), np.squeeze(Xi_raw[0]))
    plt.show()
        
def generate_submission(img_nb = None, plot = True, submission_filename = "submission.csv", mode = DEFAULT['mode']):
    """ Generate a .csv containing the classification of the test set. """
    image_filenames = get_image_filenames(img_nb)
    print('Generating file: {}...'.format(submission_filename))
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for i, filename in enumerate(image_filenames[0:]):
            img_input = load_image(filename, mode)
            Xi_raw, Xi = get_predicted_mask(filename, mode)
            ground = get_ground_img(Xi, patch_size = PATCH_SIZE, foreground_threshold = PTHRESHOLD)
            if plot:
                visualize_step(i, img_input, Xi_raw, Xi, ground, True)
            else:
                print('img {}...'.format(i+1))
                mpimg.imsave(os.path.join('./masks', 'test_{}.png'.format(i+1)), np.squeeze(Xi_raw[0]))
            f.writelines([
                "{:03d}_{}_{},{}\n".format(i+1, j*PATCH_SIZE, k*PATCH_SIZE, ground[k,j])
                for j in range(ground.shape[1]) for k in range(ground.shape[0])
            ])
    print('Submission generated at {}!'.format(submission_filename))


generate_submission(mode=get_model_prop('mode'), plot = False)
