import numpy as np
from helpers import *
from cross_validation import *

# Load the training set
root_dir = "training/"

image_dir = root_dir + "images/"
files = os.listdir(image_dir)
n = len(files)
print("Loading " + str(n) + " images")
imgs = np.asarray([load_image(image_dir + files[i]) for i in range(n)])

gt_dir = root_dir + "groundtruth/"
print("Loading " + str(n) + " images")
gt_imgs = np.asarray([load_image(gt_dir + files[i]) for i in range(n)])

#from naive_model import NaiveModel
from cnn_model import CnnModel
#from logistic_model import LogisticModel

#model = NaiveModel()
model = CnnModel()
#model = LogisticModel()

np.random.seed(1) # Ensure reproducibility

model.model.summary()
model.train(gt_imgs, imgs)

# Save weights to disk
model.save('weights_sven.h5')