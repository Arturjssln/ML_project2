import os

import numpy as np

from keras.layers import Input
from keras.optimizers import *
from keras.utils import np_utils
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from keras import losses

from sklearn.model_selection import train_test_split

import imgaug as ia
import imgaug.augmenters as iaa

from helpers import *
from unet import unet


def f1(true, pred):  # considering sigmoid activation, threshold = 0.5
    pred = K.cast(tf.greater(pred, 0.5), K.floatx())

    groundPositives = K.sum(true) + K.epsilon()
    correctPositives = K.sum(true * pred) + K.epsilon()
    predictedPositives = K.sum(pred) + K.epsilon()

    precision = correctPositives / predictedPositives
    recall = correctPositives / groundPositives

    m = (2 * precision * recall) / (precision + recall)

    return m


def iou(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou_coef = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou_coef


class CNN:
    def __init__(
        self,
        rootdir="./",
        window_size=256,
        channels_size=3,
        dropout_rate=0.5,
        nb_epochs=20,
        verbose=1,
        nb_classes=1,
        validation_size=0.2,
        train_batch_size=32,
        val_batch_size=16,
        random_seed=1000,
        augmentation_coef=30,
        adjust_metric="val_f1",
        save_metric="val_f1",
        best_model_filename="best.h5",
        log_csv_filename="log.csv",
        lk_alpha=0.01,
    ):
        """ Construct a CNN segmenter. """
        assert nb_classes > 0
        assert train_batch_size > 0
        assert val_batch_size > 0
        assert window_size % 32 == 0
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
        self.random_seed = random_seed
        self.augmentation_coef = augmentation_coef
        self.adjust_metric = adjust_metric
        self.save_metric = save_metric
        self.best_model_filename = best_model_filename
        self.log_csv_filename = log_csv_filename
        self.lk_alpha = lk_alpha
        self.init_model()
        self.init_augmenter()

    def init_model(self):
        """ Initialize model. """
        pool_size = (2, 2)
        conv_size = 3
        upconv_size = 2
        nb_conv_1 = 64
        nb_conv_2 = 128
        nb_conv_3 = 256
        nb_conv_4 = 512
        nb_conv_5 = 1024
        dropout_rate = self.dropout_rate
        # lk_alpha = 0.1
        inputs = Input((self.window_size, self.window_size, self.channels_size))
        self.model = unet(
            inputs,
            dropout_rate,
            pool_size,
            conv_size,
            upconv_size,
            nb_conv_1,
            nb_conv_2,
            nb_conv_3,
            nb_conv_4,
            nb_conv_5,
            self.lk_alpha,
        )
        # select loss function and metrics, as well as optimizer
        self.model.compile(
            optimizer=Adam(lr=1e-4),
            loss="binary_crossentropy",
            metrics=[iou, f1, "accuracy",],
        )
        # TODO: use a better loss? https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
        # TODO: loss using patches (we can pass a function as loss param, returning such as losses.binary_crossentropy(y_true, y_pred))

    def init_augmenter(self):
        self.augmenter1 = iaa.Sequential(
            [
                iaa.Fliplr(0.25),
                iaa.Flipud(0.25),
                iaa.Affine(
                    rotate=(-180, 180), mode="reflect"
                ),  # rotate all images with random angle
            ]
        )
        self.augmenter2 = iaa.Sequential(
            [
                iaa.Sometimes(
                    0.1, iaa.GaussianBlur(sigma=(0, 1))
                ),  # blur 50 % of the images
                # Make some images brighter and some darker.
                iaa.Multiply((0.8, 1.2), per_channel=0.1),
                iaa.GammaContrast((0.5, 1.5))
            ]
        )

    def __augment__(self, img, seg):
        aug_det1 = self.augmenter1.to_deterministic()
        # change only orienation and border using mirror boundaries for both img and groundtruth
        seg_aug = aug_det1.augment_image(seg)
        segmap_aug = ia.SegmentationMapsOnImage(seg_aug, shape=img.shape)
        segmap_aug = 1 * segmap_aug.get_arr()
        image_aug1 = aug_det1.augment_image(img)
        # Add some noise and and blurring on the image
        image_aug = self.augmenter2.augment_image(image_aug1)
        return image_aug, segmap_aug

    def crop_corner(self, img, seg, corner=None):
        shape = img.shape  # x,y
        if corner is None:
            corner = np.random.choice(4)
        # compute indices for slice
        x_from = int(corner / 2) * (shape[0] - self.window_size)
        y_from = int(corner % 2) * (shape[1] - self.window_size)
        x_to = x_from + self.window_size
        y_to = y_from + self.window_size
        # return sliced images
        return img[x_from:x_to, y_from:y_to], seg[x_from:x_to, y_from:y_to]

    def __generator__(self, X, Y, batch_size=16):
        """
        Procedure for real-time minibatch creation and image augmentation.
        This runs in a parallel thread while the model is being trained.
        """
        while 1:
            # Generate one minibatch
            X_batch = np.empty(
                (batch_size, self.window_size, self.window_size, self.channels_size)
            )
            # We use an integer value to label the pixel class
            Y_batch = np.empty((batch_size, self.window_size, self.window_size, 1))
            for i in range(batch_size):
                # Select a random image
                idx = np.random.choice(X.shape[0])
                shape = X[idx].shape
                img, seg = X[idx], Y[idx]
                # img, seg = self.crop_corner(X[idx], Y[idx])
                img, seg = self.__augment__(img, seg)
                img, seg = random_crop(img, seg, (self.window_size, self.window_size))
                X_batch[i], Y_batch[i] = (
                    unsqueeze(img) if self.channels_size == 1 else img,
                    unsqueeze(seg),
                )
            yield (X_batch, Y_batch)

    def split_data(self, X, Y, rate, seed=None):
        if seed is None:
            seed = self.random_seed
        return train_test_split(X, Y, test_size=rate, random_state=seed)

    def train(self, X, Y, initial_epoch=0):
        """
        Train this model with the given dataset.
        """
        X_train, X_val, y_train, y_val = self.split_data(X, Y, self.validation_size)

        print("Training on:", X_train.shape, "Validating on:", X_val.shape)
        SPLIT_RATE = 4  # we divide each image in four parts
        # let's say we want X more images to train on! - Arbitrary, relies on randomness of generate_minibatch
        samples_per_epoch = X.shape[0] * SPLIT_RATE * self.augmentation_coef

        np.random.seed(self.random_seed)  # Ensure determinism

        # Save metadata to a file (useful when colab goes in background)
        csv_logger = CSVLogger(
            filename=os.path.join(self.rootdir, self.log_csv_filename), append=True
        )
        # This callback reduces the learning rate when the training accuracy does not improve any more
        lr_callback = ReduceLROnPlateau(
            monitor=self.adjust_metric,
            factor=0.5,
            patience=5,
            verbose=1,
            mode="auto",
            epsilon=0.0001,
            cooldown=0,
            min_lr=0,
        )
        # Stops the training process upon convergence
        stop_callback = EarlyStopping(
            monitor=self.adjust_metric,
            min_delta=0.0001,
            patience=11,
            verbose=1,
            mode="auto",
        )
        # Save the latest best model to rootdir
        mode_autosave = ModelCheckpoint(
            os.path.join(self.rootdir, self.best_model_filename),
            monitor=self.save_metric,
            mode="max",
            save_best_only=True,
            verbose=1,
            period=1,
        )

        try:
            history = self.model.fit_generator(
                self.__generator__(X_train, y_train, self.train_batch_size),
                validation_data=self.__generator__(X_val, y_val, self.val_batch_size),
                validation_steps=samples_per_epoch
                * self.validation_size
                // self.val_batch_size,
                samples_per_epoch=samples_per_epoch // self.train_batch_size,
                nb_epoch=self.nb_epochs,
                initial_epoch=initial_epoch,
                verbose=self.verbose,
                callbacks=[lr_callback, stop_callback, csv_logger, mode_autosave],
                use_multiprocessing=True,
            )
            print("Training completed")
            return history
        except KeyboardInterrupt:
            # Do not throw away the model in case the user stops the training process
            pass

    def test(self, X, Y, steps=None):
        if steps is None:
            steps = X.shape[0] * 4 * 10
        results = self.model.evaluate_generator(
            self.__generator__(X, Y),
            steps=steps,
            verbose=self.verbose,
            use_multiprocessing=True,
        )
        labels = self.model.metrics_names
        return dict(zip(labels, results))

    def save(self, filename):
        """ Save the weights of this model. """
        self.model.save_weights(filename)

    def load(self, filename):
        """ Load the weights for this model from a file. """
        self.model.load_weights(filename)
