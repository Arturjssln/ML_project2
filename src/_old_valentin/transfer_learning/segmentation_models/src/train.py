import segmentation_models as sm
import keras
from helpers import *


BACKBONE = "resnet34"
preprocess_input = sm.get_preprocessing(BACKBONE)

# Loaded a set of images
root_dir = "./data/"

x_train_dir = os.path.join(root_dir, 'training/images')
y_train_dir = os.path.join(root_dir, 'training/groundtruth')

x_valid_dir = os.path.join(root_dir, 'validation/images')
y_valid_dir = os.path.join(root_dir, 'validation/groundtruth')

# image_dir = root_dir + "images/"
# files = os.listdir(image_dir)
# n = min(10, len(files))  # Load maximum 20 images
# print("Loading " + str(n) + " images")
# imgs = np.array([load_image(image_dir + files[i]) for i in range(n)])
# print(files[0], imgs[0].shape)

# gt_dir = root_dir + "groundtruth/"
# print("Loading " + str(n) + " images")
# gt_imgs = np.array([load_image(gt_dir + files[i]) for i in range(n)])
# print(files[0], gt_imgs[0].shape)


# define optomizer
optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]


# define model
model = sm.Unet(BACKBONE, encoder_weights="imagenet")
model.compile(
    "Adam", loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score],
)

model_checkpoint = ModelCheckpoint(
    "model_saved.hdf5", monitor="loss", verbose=1, save_best_only=True
)

# fit model
# if you use data generator use model.fit_generator(...) instead of model.fit(...)
# more about `fit_generator` here: https://keras.io/models/sequential/#fit_generator
model.fit(
    x=x_train,
    y=y_train,
    batch_size=16,
    epochs=100,
    validation_data=(x_val, y_val),
    callbacks=[model_checkpoint],
)
