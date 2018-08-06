from lucid.misc.io.serialize_array import _normalize_array
from lucid.misc.io import load
from scipy.misc import imsave
from lucid.misc.tfutil import create_session
import numpy as np
import tensorflow as tf
import os, glob
from lucid.optvis.param import cppn

from tqdm import tqdm
size_n = 200

save_dest = "results/interpolation_smooth"
os.system(f'mkdir -p {save_dest}')

sess = create_session()
t_size = tf.placeholder_with_default(size_n, [])
t_image = cppn(t_size)
train_vars = sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

def render_params(params, size=224):
  feed_dict = dict(zip(train_vars, params))
  feed_dict[t_size] = size
  return sess.run(t_image, feed_dict)[0]

f_models = sorted(glob.glob("results/smooth_models/*000*.npy"))


print("Loading models")
MODELS = list(map(load, tqdm(f_models)))


bps = 125
fps = 30
pi = np.pi

period = 2*pi
duration = 60.0/bps
n_frames = (fps*duration)

T = np.linspace(0, duration, n_frames).reshape(-1,1)
period = [period, period]
phase  = [0, np.pi]

Y = np.cos(period*T+phase)
Y = np.sin(Y*(pi/2))

#Y = np.sin(Y*(pi/2))
Y = (Y+1)/2

frame_n = 0

for i in range(len(f_models)-1):
  print ("MODEL", i)
  m0 = MODELS[i]
  m1 = MODELS[i+1]

  Y1 = np.linspace(0, 1, n_frames)
  Y0 = 1-Y1

  for y0, y1 in Y:

    params = y0*m0 + y1*m1

    #params *= 1.0 + y0*(1.0-y0)
    params *= 1.0 + (y0*(1.0-y0)) ** 2
    
    img = render_params(params, size=600)

    f_image = os.path.join(save_dest, f"{frame_n:08d}.png")
    imsave(f_image, img)

    frame_n += 1
    print (f"Rendered {f_image}")


