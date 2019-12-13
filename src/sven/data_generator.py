import os
import cv2
import numpy as np
import tensorflow as tf
import albumentations as albu
from albumentations import Resize
from keras.utils import Sequence
from skimage.io import imread
from sklearn.utils import shuffle

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
            
from albumentations import (Blur, Compose, HorizontalFlip, HueSaturationValue,
                            IAAEmboss, IAASharpen, JpegCompression, OneOf,
                            RandomBrightness, RandomBrightnessContrast,
                            RandomContrast, RandomCrop, RandomGamma,
                            RandomRotate90, RGBShift, ShiftScaleRotate,
                            Transpose, VerticalFlip, ElasticTransform, GridDistortion, OpticalDistortion)
 

def aug_with_crop(image_size=256, crop_prob=1):
    comp = Compose([
        RandomCrop(width=image_size, height=image_size, p=crop_prob),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        Transpose(p=0.5),
        ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
        RandomBrightnessContrast(p=0.5),
        RandomGamma(p=0.25),
        IAAEmboss(p=0.25),
        Blur(p=0.01, blur_limit = 3),
        OneOf([
            ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(p=0.5),
            OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)                  
        ], p=0.8)
    ], p = 1)
    return comp

class DataGeneratorFolder (Sequence):

    def __init__(self, image_path, mask_path,
                 batch_size=1, image_size=256, nb_y_features=1,
                 augmentation=None,
                 suffle=False):

        self.image_filenames = list(
            map(lambda x: f'{image_path}/{x}', listdir_nohidden(image_path))
            )

        self.mask_filenames = list(
            map(lambda x: f'{image_path}/{x}', listdir_nohidden(mask_path))
            )

        self.batch_size = batch_size
        self.augmentation = augmentation
        self.image_size = image_size
        self.nb_y_features = nb_y_features
        self.suffle = suffle

    def __len__(self):
        """
        Calculates size of batch
        """
        return int(np.ceil(len(self.image_filenames) / (self.batch_size)))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.suffle == True:
            self.image_filenames, self.mask_filenames = shuffle(
                self.image_filenames, self.mask_filenames)

    def read_image_mask(self, image_name, mask_name):
        return (imread(image_name)/255).astype(np.float32), (imread(mask_name, as_gray=True) > 0).astype(np.int8)

    def __getitem__(self, index):
        data_index_min = int(index * self.batch_size)
        data_index_max = int(
            min((index + 1) * self.batch_size, len(self.image_filenames)))
        images_indexes = self.image_filenames[data_index_min:data_index_max]
        masks_indexes = self.mask_filenames[data_index_min:data_index_max]

        this_batch_size = len(images_indexes)

        X = np.empty((this_batch_size, self.image_size,
                      self.image_size, 3), dtype=np.float32)
        y = np.empty((this_batch_size, self.image_size,
                      self.image_size, self.nb_y_features), dtype=np.uint8)

        for i in range(len(images_indexes)):
            X_sample, Y_sample = self.read_image_mask(
                images_indexes[i], masks_indexes[i])

            if self.augmentation is not None:

                # Augmentation code
                #augmented = self.augmentation(self.image_size)(image=X_sample, mask=Y_sample)
                augmented = aug_with_crop(self.image_size)(image=X_sample, mask=Y_sample)
                image_augm = augmented['image']
                mask_augm = augmented['mask'].reshape(self.image_size, self.image_size, self.nb_y_features)

                if len(image_augm.shape) > 2 and image_augm.shape[2] == 4:
                    image_augm = cv2.cvtColor(image_augm, cv2.COLOR_BGRA2RGB)
                X[i] = np.clip(image_augm, a_min=0, a_max=1)
                y[i] = mask_augm

            elif self.augmentation is None and self.batch_size == 1:
                augmented = Resize(height=(X_sample.shape[0]//32)*32, width=(
                    X_sample.shape[1]//32)*32)(image=X_sample, mask=Y_sample)

                X_sample, Y_sample = augmented['image'], augmented['mask']

                if len(X_sample.shape) > 2 and X_sample.shape[2] == 4:
                    X_sample = cv2.cvtColor(X_sample, cv2.COLOR_BGRA2RGB)

                return X_sample.reshape(1, X_sample.shape[0], X_sample.shape[1], 3).astype(np.float32), Y_sample.reshape(1, X_sample.shape[0], X_sample.shape[1], self.nb_y_features).astype(np.uint8)

        return X, y