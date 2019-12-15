import os

import numpy as np

from keras.layers import Input
from keras.optimizers import *
from keras.utils import np_utils
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras import losses

from sklearn.model_selection import train_test_split

import imgaug as ia
import imgaug.augmenters as iaa

from helpers import *
from unet import unet

class CNN:
    def __init__(self,
                 rootdir = './',
                 window_size = 256,
                 channels_size = 3,
                 dropout_rate = 0.5,
                 nb_epochs = 20,
                 verbose = 1,
                 nb_classes = 1,
                 validation_size = 0.2,
                 train_batch_size = 32,
                 val_batch_size = 16):
        """ Construct a CNN segmenter. """
        assert nb_classes > 0
        assert train_batch_size > 0
        assert val_batch_size > 0
        assert window_size%32 == 0# TODO: how do we crop?
        self.rootdir = rootdir
        self.window_size = window_size
        self.channels_size = channels_size
        self.dropout_rate = dropout_rate
        self.nb_epochs = nb_epochs
        self.verbose = verbose
        self.nb_classes = nb_classes
        self.validation_size = validation_size
        self.val_batch_size = val_batch_size
        self.train_batch_size = train_batch_size
        self.conv_size = 3
        self.random_seed = 1000
        self.init_model()
        self.init_augmenter()
        # TODO: option to set if keep rgb or convert to hsv
        #self.patch_size = 16 # TODO: use it in penalizer and predict patches function
        

    def init_model(self):
        """ Initialize model. """
        pool_size = (2, 2)
        conv_size = self.conv_size
        upconv_size = 2
        nb_conv_1 = 64
        nb_conv_2 = 128
        nb_conv_3 = 256
        nb_conv_4 = 512
        nb_conv_5 = 1024
        dropout_rate = self.dropout_rate
        #lk_alpha = 0.1
        inputs = Input((self.window_size, self.window_size, self.channels_size))
        self.model = unet(inputs, dropout_rate, pool_size,
            conv_size, upconv_size, nb_conv_1, nb_conv_2, nb_conv_3, nb_conv_4, nb_conv_5)
        # select loss function and metrics, as well as optimizer
        self.model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        # TODO: use better metrics? 
        # TODO: use a better loss? https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
        # TODO: loss using patches (we can pass a function as loss param, returning such as losses.binary_crossentropy(y_true, y_pred))


    def init_augmenter(self):
        self.augmenter = iaa.Sequential([
            iaa.Fliplr(0.25), # horizontally flip 25% of the images
            iaa.Flipud(0.25), # vertically flip 25% of all images
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 1))), # blur 50 % of the images
            iaa.Sometimes(0.5, iaa.Affine(rotate=(-180, 180))), # rotate 50% of the images
            # Make some images brighter and some darker.
            # In 10% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            #iaa.Multiply((0.8, 1.2), per_channel=0.1),# TODO: deactivated to ensure HSV compatibility
            #iaa.GammaContrast((0.5, 1.5)) # TODO: deactivated to ensure HSV compatibility
        ], random_order=True)


    # def __pad_images__(self, X, Y, padding):
    #     # Pad training set images (by appling mirror boundary conditions)
    #     X_new = np.empty((X.shape[0],
    #                      X.shape[1] + 2*padding, X.shape[2] + 2*padding,
    #                      X.shape[3]))
    #     Y_new = np.empty((Y.shape[0],
    #                      Y.shape[1] + 2*padding, Y.shape[2] + 2*padding))
    #     for i in range(X.shape[0]):
    #         X_new[i] = pad_image(X[i], padding)
    #         Y_new[i] = pad_image(Y[i], padding)
    #     return (X_new, Y_new)


    def __augment__(self, img, seg):
        aug_det = self.augmenter.to_deterministic() 
        image_aug = aug_det.augment_image(img)
        segmap = ia.augmentables.segmaps.SegmentationMapsOnImage(np.array(seg).astype('uint8'), shape=img.shape)
        segmap_aug = aug_det.augment_segmentation_maps( segmap )
        segmap_aug = 1*segmap_aug.get_arr()
        return image_aug , segmap_aug


    def crop_corner(self, img, seg, corner = None):
        shape = img.shape # x,y
        if corner is None:
            corner = np.random.choice(4)
        # compute indices for slice
        x_from = int(corner/2)*(shape[0]-self.window_size)
        y_from = int(corner%2)*(shape[1]-self.window_size)
        x_to = x_from+self.window_size
        y_to = y_from+self.window_size
        # return sliced images
        return img[x_from:x_to, y_from:y_to], seg[x_from:x_to, y_from:y_to]


    def __generator__(self, X, Y, batch_size = 32):
        """
        Procedure for real-time minibatch creation and image augmentation.
        This runs in a parallel thread while the model is being trained.
        """
        while 1:
            # Generate one minibatch
            X_batch = np.empty((batch_size, self.window_size, self.window_size, self.channels_size))
            # We use an integer value to label the pixel class
            Y_batch = np.empty((batch_size, self.window_size, self.window_size, 1))
            for i in range(batch_size):
                # Select a random image
                idx = np.random.choice(X.shape[0])
                shape = X[idx].shape
                # Crop random part from the corner (augment #1)
                img, seg = self.crop_corner(X[idx], Y[idx])
                # Apply random transformations (augment #2)
                img, seg = self.__augment__(img, seg)
                X_batch[i], Y_batch[i] = img, np.expand_dims(seg, axis=2)
            yield (X_batch, Y_batch)


    def split_data(self, X, Y, rate):
        return train_test_split(X, Y, test_size=rate, random_state=self.random_seed)


    def train(self, X, Y, initial_epoch = 0):
        """
        Train this model with the given dataset.
        """
        #X, Y = self.__pad_images__(X, Y, self.conv_size) # TODO: should we pad images?

        X_train, X_val, y_train, y_val = self.split_data(X, Y, self.validation_size)
        
        print('Training on:', X_train.shape, 'Validating on:', X_val.shape)
        SPLIT_RATE = 4 # we divide each image in four parts
        AUGMENTATION_RATE = 15 # let's say we want X more images to train on! - Arbitrary, relies on randomness of generate_minibatch
        samples_per_epoch = X.shape[0]*SPLIT_RATE*AUGMENTATION_RATE

        np.random.seed(self.random_seed) # Ensure determinism
        
        # This callback reduces the learning rate when the training accuracy does not improve any more
        lr_callback = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=5, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
        # Stops the training process upon convergence
        stop_callback = EarlyStopping(monitor='acc', min_delta=0.0001, patience=11, verbose=1, mode='auto')
        # Save the latest best model to rootdir
        mode_autosave = ModelCheckpoint(os.path.join(self.rootdir, 'best.h5'), monitor='acc', mode='max', save_best_only=True, verbose=1, period=1)
        
        try:
            self.model.fit_generator(
                            self.__generator__(X_train, y_train, self.train_batch_size),
                            validation_data=self.__generator__(X_val, y_val, self.val_batch_size),# TODO: random split at input
                            validation_steps=samples_per_epoch*self.validation_size//self.val_batch_size,
                            samples_per_epoch=samples_per_epoch//self.train_batch_size,
                            nb_epoch=self.nb_epochs,
                            initial_epoch=initial_epoch,
                            verbose=self.verbose,
                            callbacks=[lr_callback, stop_callback, mode_autosave])
                            # TODO: enable use_multiprocessing=True; ensure picke_safe before
        except KeyboardInterrupt:
            # Do not throw away the model in case the user stops the training process
            pass

        print('Training completed')


    def test(self, X, Y, steps = None):
        if steps is None:
            steps = X.shape[0]*4*10
        return self.model.evaluate_generator(
                        self.__generator__(X, Y),
                        steps=steps,
                        verbose=self.verbose,
                        use_multiprocessing=True)


    def save(self, filename):
        """ Save the weights of this model. """
        self.model.save_weights(filename)


    def load(self, filename):
        """ Load the weights for this model from a file. """
        self.model.load_weights(filename)
        

    def predict_segment(self, X, auto_patch = True):
        shape = X.shape
        # assert size for network input
        assert shape[1] == shape[2]
        if auto_patch:
            assert shape[1] == self.window_size*4
        else:
            assert shape[1] == self.window_size
        # TODO

    # def classify(self, X):
    #     """
    #     Classify an unseen set of samples.
    #     This method must be called after "train".
    #     Returns a list of predictions.
    #     """
    #     # Subdivide the images into blocks
    #     img_patches = create_patches(X, self.patch_size, 16, self.padding)
        
    #     # Run prediction
    #     Z = self.model.predict(img_patches)
    #     Z = (Z[:,0] < Z[:,1]) * 1
        
    #     # Regroup patches into images
    #     return group_patches(Z, X.shape[0])
        