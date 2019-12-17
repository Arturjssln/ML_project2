from keras.models import *
from keras.layers import *
import tensorflow as tf

def unet(inputs, dropout_rate, pool_size, conv_size, upconv_size,
    nb_conv_1, nb_conv_2, nb_conv_3, nb_conv_4, nb_conv_5, lk_alpha = 0.1):
    """
    UNET, cf: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
    """
    lrelu = lambda x: tf.keras.activations.relu(x, alpha=lk_alpha)
    lrelu = lrelu if lk_alpha > 0 else "relu"
    conv1 = Conv2D(nb_conv_1, conv_size, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(nb_conv_1, conv_size, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(conv1)

    pool1 = MaxPooling2D(pool_size=pool_size)(conv1)
    conv2 = Conv2D(nb_conv_2, conv_size, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(nb_conv_2, conv_size, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(conv2)

    pool2 = MaxPooling2D(pool_size=pool_size)(conv2)
    conv3 = Conv2D(nb_conv_3, conv_size, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(nb_conv_3, conv_size, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(conv3)

    pool3 = MaxPooling2D(pool_size=pool_size)(conv3)
    conv4 = Conv2D(nb_conv_4, conv_size, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(nb_conv_4, conv_size, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(dropout_rate)(conv4)

    pool4 = MaxPooling2D(pool_size=pool_size)(drop4)
    conv5 = Conv2D(nb_conv_5, conv_size, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(nb_conv_5, conv_size, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(dropout_rate)(conv5)

    up6 = Conv2D(nb_conv_4, upconv_size, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = pool_size)(drop5))
    merge6 = concatenate([drop4, up6], axis = 3)
    conv6 = Conv2D(nb_conv_4, conv_size, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(nb_conv_4, conv_size, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(nb_conv_3, upconv_size, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = pool_size)(conv6))
    merge7 = concatenate([conv3, up7], axis = 3)
    conv7 = Conv2D(nb_conv_3, conv_size, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(nb_conv_3, conv_size, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(nb_conv_2, upconv_size, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = pool_size)(conv7))
    merge8 = concatenate([conv2, up8], axis = 3)
    conv8 = Conv2D(nb_conv_2, conv_size, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(nb_conv_2, conv_size, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(nb_conv_1, upconv_size, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = pool_size)(conv8))
    merge9 = concatenate([conv1, up9], axis = 3)
    conv9 = Conv2D(nb_conv_1, conv_size, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(nb_conv_1, conv_size, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, conv_size, activation = lrelu, padding = 'same', kernel_initializer = 'he_normal')(conv9)

    # final segmentation layer: pixel-wise classifier
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    return Model(input = inputs, output = conv10)