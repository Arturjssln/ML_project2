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

# from unet import unet
from unetV2 import unet


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


def create_weighted_binary_crossentropy(zero_weight, one_weight):
    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1.0 - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy


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
        stop_metric="val_f1",
        best_model_filename="best.h5",
        log_csv_filename="log.csv",
        lk_alpha=0.0,
        net_depth=4,
        batchnorm=False,
        residual=False,
        use_multiprocessing=True,
        first_conv_size=64,
        patch_size=16,
        loss_fn="binary_crossentropy",
    ):
        """ Construct a CNN segmenter. """
        assert nb_classes > 0, "classify at least in one category!"
        assert train_batch_size > 0, "cannot train on 0 images"
        assert val_batch_size > 0, "cannot validate on 0 images"  # TODO: allow it
        assert window_size % (2**net_depth) == 0, "unet reduces on x{} convolution size".format(2**net_depth)
        assert window_size % patch_size == 0, "unet reduces on x{} patches".format(
            patch_size
        )
        assert (
            lk_alpha >= 0
        ), "leaky alpha has to be set to 0 to switch to simple relu activation function"
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
        self.stop_metric = stop_metric
        self.best_model_filename = best_model_filename
        self.log_csv_filename = log_csv_filename
        self.lk_alpha = lk_alpha
        self.net_depth = net_depth
        self.batchnorm = batchnorm
        self.residual = residual
        self.first_conv_size = first_conv_size
        self.patch_size = patch_size
        self.use_multiprocessing = use_multiprocessing
        self.loss_fn = loss_fn
        self.init_model()
        self.init_augmenter()

    def init_model(self):
        """ Initialize model. """
        self.model = unet(
            self.window_size,
            channels=self.channels_size,
            dropout=self.dropout_rate,
            lk_alpha=self.lk_alpha,
            start_ch=self.first_conv_size,
            depth=self.net_depth,
            batchnorm=self.batchnorm,
            maxpool=True,
            upconv=True,
            residual=self.residual,
        )
        loss_fn = self.__get_loss_fn__()
        # select loss function and metrics, as well as optimizer
        self.model.compile(
            optimizer=Adam(lr=1e-4), loss=loss_fn, metrics=[iou, f1, "accuracy",],
        )  # TODO: use a better loss? https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
        print(self.model.summary())

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
                iaa.GammaContrast((0.5, 1.5)),
            ]
        )

    def __get_loss_fn__(self):
        if self.loss_fn == "weighted_binary_crossentropy":
            w0 = 0.11
            w1 = 0.89
            print(
                "using weighted_binary_crossentropy with class weights (0,1)=({},{})".format(
                    w0, w1
                )
            )
            return create_weighted_binary_crossentropy(w0, w1)
        return self.loss_fn

    def __augment__(self, img, seg):
        aug_det1 = self.augmenter1.to_deterministic()
        aug_det2 = self.augmenter2.to_deterministic()
        img = aug_det1.augment_image(img)
        seg = aug_det1.augment_image(seg)
        seg = ia.SegmentationMapsOnImage(seg, shape=img.shape)
        seg = 1 * seg.get_arr()
        img = aug_det2.augment_image(img)
        return img, seg

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

    def __transform__(self, img, seg):
        img_cp, seg_cp = img.copy(), seg.copy()
        img_cp, seg_cp = self.__augment__(img_cp, seg_cp)
        img_cp, seg_cp = random_crop(
            img_cp, seg_cp, (self.window_size, self.window_size)
        ) if self.channels_size > 1 else random_crop_1(
            img_cp, seg_cp, (self.window_size, self.window_size)
        )
        return (
            unsqueeze(img_cp) if self.channels_size == 1 else img_cp,
            unsqueeze(seg_cp),
        )

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
                X_batch[i], Y_batch[i] = self.__transform__(X[idx], Y[idx])
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
        callbacks = [csv_logger]
        # This callback reduces the learning rate when the training accuracy does not improve any more
        if self.adjust_metric is not None:
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
            callbacks += [lr_callback]
        # Stops the training process upon convergence
        if self.stop_metric is not None:
            stop_callback = EarlyStopping(
                monitor=self.stop_metric,
                min_delta=0.0001,
                patience=11,
                verbose=1,
                mode="auto",
            )
            callbacks += [stop_callback]
        # Save the latest best model to rootdir
        if self.save_metric is not None and self.best_model_filename is not None:
            mode_autosave = ModelCheckpoint(
                os.path.join(self.rootdir, self.best_model_filename),
                monitor=self.save_metric,
                mode="max",
                save_best_only=True,
                verbose=1,
                period=1,
            )
            callbacks += [mode_autosave]

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
                callbacks=callbacks,
                use_multiprocessing=self.use_multiprocessing,
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
            use_multiprocessing=self.use_multiprocessing,
        )
        labels = self.model.metrics_names
        return dict(zip(labels, results))

    def save(self, filename):
        """ Save the weights of this model. """
        self.model.save_weights(filename)

    def load(self, filename):
        """ Load the weights for this model from a file. """
        self.model.load_weights(filename)
