#!/usr/bin/env python
# coding: utf-8

BEST_MODEL = "V3.002.h5"
MODEL_NAME = "V3.003.PP"

import numpy as np
import os
import sys

from helpers import *
from CNNv2 import CNN

root_dir = "training/"

import math
import cv2


def load_and_adjust(img_path):
    img = load_image(img_path)
    dest_size = math.ceil(img.shape[0] / 32) * 32
    return cv2.resize(img, (dest_size, dest_size))


image_dir = root_dir + "images/"
files = os.listdir(image_dir)
n = len(files)
print("Loading " + str(n) + " images")
imgs = np.asarray([load_and_adjust(image_dir + files[i]) for i in range(n)])

gt_dir = root_dir + "groundtruth/"
print("Loading " + str(n) + " groundtruth")
gt_imgs = np.asarray([load_and_adjust(gt_dir + files[i]) for i in range(n)])

model_predict = CNN(
    rootdir=root_dir,
    window_size=imgs.shape[1],
    channels_size=imgs[0].shape[2],
    net_depth=5,
    lk_alpha=0.1,
    batchnorm=True,
    residual=True,
    random_seed=1000,
    use_multiprocessing=False,
)
model_predict.load(os.path.join(root_dir, BEST_MODEL))


def predict(img, threshold=0.5, model_predict=model_predict, coef=255):
    Xi = model_predict.model.predict(np.array([img]))
    if threshold is not None:
        Xi = np.where(Xi > threshold, 1, 0)
    return np.squeeze(Xi * coef).astype("uint8")


predictions = [predict(img) for img in imgs]

model_predict = None

model = CNN(
    rootdir=root_dir,
    window_size=256,
    channels_size=1,
    first_conv_size=128,
    lk_alpha=0.1,
    batchnorm=True,
    residual=True,
    nb_epochs=50,
    adjust_metric="val_f1",
    save_metric="val_iou",
    stop_metric="f1",
    train_batch_size=2,
    val_batch_size=1,
    random_seed=657,
    validation_size=0.1,
    best_model_filename="{}.h5".format(MODEL_NAME),
    log_csv_filename="{}.csv".format(MODEL_NAME),
    use_multiprocessing=False,
)

# SAME SEED AS BEST MODEL
TEST_PROPORTION = 0.05
X_train, X_test, y_train, y_test = model.split_data(
    np.array(predictions), gt_imgs, rate=TEST_PROPORTION, seed=42
)
# _, Z_test, _, _ = model.split_data(imgs, gt_imgs, rate=TEST_PROPORTION, seed=42)
print(X_test.shape[0], "test images;", X_train.shape[0], "train images")


# model.load(os.path.join(root_dir, f'{MODEL_NAME}-saved_weights.h5'))
# model.load(os.path.join(root_dir, '{}.h5'.format(MODEL_NAME)))
# model.load(os.path.join(root_dir, 'models', 'another_model.h5'))
INITIAL_EPOCH = 0

model.train(X_train, y_train, initial_epoch=INITIAL_EPOCH)

model.save(os.path.join(root_dir, "{}-saved_weights.h5".format(MODEL_NAME)))

results = model.test(X_test, y_test)
print("--- report: saved weights ---")
print(results)

model.load(os.path.join(root_dir, "{}.h5".format(MODEL_NAME)))
results = model.test(X_test, y_test)
print("--- report: {}.h5 ---".format(MODEL_NAME))
print(results)
