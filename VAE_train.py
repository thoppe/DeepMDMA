import numpy as np
from lucid.misc.io import load

f_model = 'results/VAE_base_models/mixed4a_1_0.npy'
params = load(f_model)

shapes = list(map(lambda x:x.shape, params))

def unpack(X):
    return np.hstack([x.ravel() for x in X])

def pack(X, shapes):
    data, i = [], 0
    for size in shapes:
        n = np.prod(size)
        
        data.append(X[i:i+n].reshape(size))
        i+=n
    return np.array(data)

u = unpack(params)
print (u)
params2 = pack(u,shapes)

for x,y in zip(params,params2):
    print ((x==y).all())

    
