from model import *
from data import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

STEPS_PER_EPOCH=100  # 300
EPOCHS=1000 # 1

data_gen_args = dict(
    rotation_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest",
)
myGene = trainGenerator(
    2, "data/roads", "images", "groundtruth", data_gen_args, save_to_dir=None
)

model = unet()
model_checkpoint = ModelCheckpoint(
    "unet_roads.hdf5", monitor="loss", verbose=1, save_best_only=True
)
model.fit_generator(
    myGene, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks=[model_checkpoint]
)

# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")

