from lucid.misc.io.serialize_array import _normalize_array
from lucid.misc.io import load
from scipy.misc import imsave
from lucid.misc.tfutil import create_session
import numpy as np
import tensorflow as tf
import os, glob
from CPPN_activations import image_cppn
from tqdm import tqdm
size_n = 200

save_dest = "results/interpolation"
os.system(f'mkdir -p {save_dest}')

sess = create_session()
t_size = tf.placeholder_with_default(size_n, [])
t_image = image_cppn(t_size)
train_vars = sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

def render_params(params, size=224):
  feed_dict = dict(zip(train_vars, params))
  feed_dict[t_size] = size
  return sess.run(t_image, feed_dict)[0]

'''
f_models = glob.glob("results/models/*")[:5]
print("Loading models")
MODELS = list(map(load, tqdm(f_models)))

fps = 30
duration = 5
n_frames = int(fps*duration)

T = np.linspace(0, duration, n_frames)
Y = np.random.uniform(size=(len(MODELS), n_frames))
Y /= Y.sum(axis=0)

for n, y in tqdm(enumerate(Y.T)):
    params = sum([x*model for (x,model) in zip(y,MODELS)])
    img = render_params(params, size=400)

    f_image = os.path.join(save_dest, f"{n:08d}.jpg")
    imsave(f_image, img)
'''


f_models = [
    'results/models/mixed4a_3x3_pre_relu_17.npy',
    'results/models/mixed4a_3x3_pre_relu_11.npy',
    'results/models/mixed4a_3x3_pre_relu_4.npy',
]
print("Loading models")
MODELS = list(map(load, tqdm(f_models)))

fps = 30
duration = 15
n_frames = int(fps*duration)

T = np.linspace(0, duration, n_frames)
P = np.array([2.0, -3.0, 5.0]).reshape(-1,1)
#P = np.array([2.0, -3.0]).reshape(-1,1)
Y = (np.cos(P*T)+1)/2
#Y /= Y.sum(axis=0)

for n, y in tqdm(enumerate(Y.T)):
    params = sum([x*model for (x,model) in zip(y,MODELS)])
    img = render_params(params, size=400)

    f_image = os.path.join(save_dest, f"{n:08d}.jpg")
    imsave(f_image, img)
    
    print(n)

