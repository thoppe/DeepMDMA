import numpy as np
import glob
from tqdm import tqdm 
from lucid.misc.io import load

#https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
latent_n = 2

def unpack(X):
    return np.hstack([x.ravel() for x in X])

def pack(X, shapes):
    data, i = [], 0
    for size in shapes:
        n = np.prod(size)
        
        data.append(X[i:i+n].reshape(size))
        i+=n
    return np.array(data)

F_MODELS = glob.glob("results/VAE_base_models/*.npy")[:5]

# Load the save models
raw_params = [load(f_model) for f_model in tqdm(F_MODELS)]

# Measure the shape of the covnet
shapes = list(map(lambda x:x.shape, raw_params[0]))

# Unpack them all
X = np.array([unpack(p) for p in raw_params])



import seaborn as sns
import pylab as plt
for x in X:
    sns.distplot(x)
plt.show()
print(X.shape)
exit()

f_model = 'results/VAE_base_models/mixed4a_1_0.npy'
params = load(f_model)

u, shapes = unpack(params)
params2 = pack(u,shapes)

for x,y in zip(params,params2):
    print ((x==y).all())

    
