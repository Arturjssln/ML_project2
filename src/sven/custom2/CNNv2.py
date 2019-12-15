import os

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import np_utils
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras import losses

from helpers import *

import imgaug as ia
import imgaug.augmenters as iaa


class CNN:
    def __init__(self,
                 rootdir = './',
                 window_size = 256,
                 channels_size = 3,
                 dropout_rate = 0.5,
                 nb_epochs = 20,
                 verbose = 1,
                 nb_classes = 1):
        """ Construct a CNN segmenter. """
        assert window_size%32 == 0# TODO: how do we crop?
        self.rootdir = rootdir
        self.window_size = window_size
        self.channels_size = channels_size
        self.dropout_rate = dropout_rate
        self.nb_epochs = nb_epochs
        self.verbose = verbose
        self.nb_classes = nb_classes
        self.conv_size = 3
        self.init_model()
        self.init_augmenter()
        # TODO: option to set if keep rgb or convert to hsv
        #self.patch_size = 16 # TODO: use it in penalizer and predict patches function
        

    def init_model(self):
        """ Initialize model. """
        # UNET, cf: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
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
        
        # TODO: LeakyReLU(alpha=0.1) is better than relu (cf. lectures)
        inputs = Input((self.window_size, self.window_size, self.channels_size))
        conv1 = Conv2D(nb_conv_1, conv_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(nb_conv_1, conv_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        
        pool1 = MaxPooling2D(pool_size=pool_size)(conv1)
        conv2 = Conv2D(nb_conv_2, conv_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(nb_conv_2, conv_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        
        pool2 = MaxPooling2D(pool_size=pool_size)(conv2)
        conv3 = Conv2D(nb_conv_3, conv_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(nb_conv_3, conv_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        
        pool3 = MaxPooling2D(pool_size=pool_size)(conv3)
        conv4 = Conv2D(nb_conv_4, conv_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(nb_conv_4, conv_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(dropout_rate)(conv4)
        
        pool4 = MaxPooling2D(pool_size=pool_size)(drop4)
        conv5 = Conv2D(nb_conv_5, conv_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(nb_conv_5, conv_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(dropout_rate)(conv5)

        up6 = Conv2D(nb_conv_4, upconv_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = pool_size)(drop5))
        merge6 = concatenate([drop4, up6], axis = 3)
        conv6 = Conv2D(nb_conv_4, conv_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(nb_conv_4, conv_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(nb_conv_3, upconv_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = pool_size)(conv6))
        merge7 = concatenate([conv3, up7], axis = 3)
        conv7 = Conv2D(nb_conv_3, conv_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(nb_conv_3, conv_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(nb_conv_2, upconv_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = pool_size)(conv7))
        merge8 = concatenate([conv2, up8], axis = 3)
        conv8 = Conv2D(nb_conv_2, conv_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(nb_conv_2, conv_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(nb_conv_1, upconv_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = pool_size)(conv8))
        merge9 = concatenate([conv1, up9], axis = 3)
        conv9 = Conv2D(nb_conv_1, conv_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(nb_conv_1, conv_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, conv_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        
        # final segmentation layer: pixel-wise classifier
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

        self.model = Model(input = inputs, output = conv10) # TODO: ensure this is doing pixel-wise segmentation

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
        segmap = ia.augmentables.segmaps.SegmentationMapsOnImage(seg, shape=img.shape)
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
    
    def train(self, X, Y):
        """
        Train this model with the given dataset.
        """
        #X, Y = self.__pad_images__(X, Y, self.conv_size) # TODO: should we pad images?
        
        print('Training set shape: ', X.shape)
        SPLIT_RATE = 4 # we divide each image in four parts
        AUGMENTATION_RATE = 30 # let's say we want 30x more images to train on! - Arbitrary, relies on randomness of generate_minibatch
        #samples_per_epoch = X.shape[0]*SPLIT_RATE*AUGMENTATION_RATE # TODO
        samples_per_epoch=1000

        np.random.seed(1000) # Ensure determinism
        
        # This callback reduces the learning rate when the training accuracy does not improve any more
        lr_callback = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=5, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
        # Stops the training process upon convergence
        stop_callback = EarlyStopping(monitor='acc', min_delta=0.0001, patience=11, verbose=1, mode='auto')
        # Save the latest best model to rootdir
        mode_autosave = ModelCheckpoint(os.path.join(self.rootdir, 'best.h5'), monitor='acc', mode='max', save_best_only=True, verbose=1, period=1)
        
        try:
            self.model.fit_generator(
                            self.__generator__(X, Y),
                            samples_per_epoch=samples_per_epoch,
                            nb_epoch=self.nb_epochs,
                            verbose=self.verbose,
                            callbacks=[lr_callback, stop_callback, mode_autosave])
        except KeyboardInterrupt:
            # Do not throw away the model in case the user stops the training process
            pass

        print('Training completed')
        
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
        