import numpy as np
import glob
from tqdm import tqdm 
from lucid.misc.io import load
from VAE_model_images import build_model
import cv2

latent_n = 4
cutoff = 2**4

f_h5 = 'results/VAE_weights.h5'

intermediate_n = 512
batch_size = 128
n_epochs = 1000

test_train_split = 0.8
F_IMAGES = glob.glob("results/VAE_base_images/*")[:cutoff]
IMG = [cv2.imread(f_img) for f_img in tqdm(F_IMAGES)]

IMG = np.array(IMG).astype(float)
IMG /= 255

img_x, img_y, img_c = IMG[0].shape

M = build_model(img_x, None, None)
print(IMG[0].shape)
exit()

# Load the save models
raw_params = [load(f_model) for f_model in tqdm(F_MODELS)]

# Measure the shape of the covnet
shapes = list(map(lambda x:x.shape, raw_params[0]))

# Unpack them all
X = np.array([unpack(p) for p in raw_params])

input_n = X[0].size


#######################################################################
VAE, encoder, decoder = build_model(
    input_n, intermediate_n, latent_n)
print(VAE.summary())

n_train = int(len(X)*test_train_split)
X_train, X_test = X[:n_train], X[n_train:]

VAE.fit(
    X_train,
    epochs=n_epochs,
    batch_size=batch_size,
    validation_data=(X_test, None)
)

f_h5 = 'results/VAE_weights.h5'
VAE.save_weights(f_h5)
print (f"Saved model to {f_h5}")

