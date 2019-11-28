from keras.models import model_from_json
from model import *
from data import *

# json_file = open("model.json", "r")
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# # load weights into new model
# model.load_weights("model.h5")
# print("Loaded model from disk")

model = unet()
model.load_weights("unet_roads.hdf5")

testGene = testGenerator("data/roads/images", img_name_pattern="satImage_00$i.png")
results = model.predict_generator(testGene, 2, verbose=1)
print(results[0])
saveResult("data/roads/images/test", results, img_name_pattern="satImage_00$i.png")
