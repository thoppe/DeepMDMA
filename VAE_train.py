import numpy as np
import glob
from tqdm import tqdm 
from lucid.misc.io import load
from VAE_model import build_model, pack, unpack

latent_n = 12
cutoff = 128**10

f_h5 = 'results/VAE_weights.h5'

intermediate_n = 512
batch_size = 32
n_epochs = 1000

test_train_split = 0.8

F_MODELS = glob.glob("results/VAE_base_models/*.npy")[:cutoff]

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

