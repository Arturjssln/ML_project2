#!/usr/bin/env python
# coding: utf-8

MODEL_NAME="V3.002"

import numpy as np
import os
import sys

from helpers import *
from CNNv2 import CNN

root_dir = "training/"

image_dir = root_dir + "images/"
files = os.listdir(image_dir)
n = len(files)
print("Loading " + str(n) + " images")
imgs = np.asarray([load_image(image_dir + files[i]) for i in range(n)])

gt_dir = root_dir + "groundtruth/"
print("Loading " + str(n) + " groundtruth")
gt_imgs = np.asarray([load_image(gt_dir + files[i]) for i in range(n)])


model = CNN(
    rootdir=root_dir,
    window_size=320,
    net_depth=5,
    lk_alpha=0.1,
    batchnorm=True,
    residual=True,
    nb_epochs=50,
    adjust_metric='val_f1',
    save_metric='val_iou',
    stop_metric='f1',
    train_batch_size=4,
    val_batch_size=4,
    random_seed=657,
    validation_size=0.1,
    best_model_filename='{}.h5'.format(MODEL_NAME),
    log_csv_filename='{}.csv'.format(MODEL_NAME),
    use_multiprocessing=False)

TEST_PROPORTION = 0.1
X_train, X_test, y_train, y_test = model.split_data(imgs, gt_imgs, rate=TEST_PROPORTION, seed=42)
print(X_test.shape[0], 'test images;', X_train.shape[0], 'train images')


#model.load(os.path.join(root_dir, f'{MODEL_NAME}-saved_weights.h5'))
#model.load(os.path.join(root_dir, '{}.h5'.format(MODEL_NAME)))
#model.load(os.path.join(root_dir, 'models', 'another_model.h5'))
INITIAL_EPOCH = 0


## Train on whole data set
#X_train, y_train = imgs, gt_imgs
history = model.train(X_train, y_train, initial_epoch=INITIAL_EPOCH)


model.save(os.path.join(root_dir, '{}-saved_weights.h5'.format(MODEL_NAME)))


def plot_metric(metric_name):
  # Plot training & validation accuracy values
  f = plt.figure()
  f.plot(history.history[metric_name])
  f.plot(history.history['val_{}'.format(metric_name)])
  f.title('Model {}'.format(metric_name))
  f.ylabel(''+metric_name)
  f.xlabel('Epoch')
  f.legend(['Train', 'Test'], loc='upper left')
  f.savefig(os.path.join(root_dir, '{}-{}.png'.format(MODEL_NAME, metric_name)))
  f.clear()
  plt.close(f)
  #plt.show()

plot_metric('loss')
plot_metric('acc')
plot_metric('f1')
plot_metric('iou')

results = model.test(X_test, y_test)
print('--- report: saved weights ---')
print(results)

model.load(os.path.join(root_dir, '{}.h5'.format(MODEL_NAME)))
results = model.test(X_test, y_test)
print('--- report: {}.h5 ---'.format(MODEL_NAME))
print(results)