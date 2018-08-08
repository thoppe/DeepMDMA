from lucid.misc.io.serialize_array import _normalize_array
from lucid.misc.io import load
from scipy.misc import imsave
from lucid.misc.tfutil import create_session
import numpy as np
import tensorflow as tf
import os, glob
from lucid.optvis.param import cppn
from IPython import embed

from tqdm import tqdm

render_size = 640
model_cutoff = 34
extension = 'png'


save_dest = "results/interpolation_smooth"
os.system(f'mkdir -p {save_dest}')

sess = create_session()
t_size = tf.placeholder_with_default(200, [])
t_image = cppn(t_size)
train_vars = sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

# avconv -y -r 30  -i "%08d.png"  -b:v 2400k ../wiggle6.mp4

def render_params(params, size=224):
  feed_dict = dict(zip(train_vars, params))
  feed_dict[t_size] = size
  return sess.run(t_image, feed_dict)[0]

f_models = sorted(glob.glob("results/smooth_models/*000*.npy"))


print(f"Loading models. Found {len(f_models)} total.")


MODELS = list(map(load, tqdm(f_models[:model_cutoff])))


########################################################
N_frames = len(MODELS)
beats_per_frame = 4
sigma_weight = 1/1.5

bpm = 80
fps = 30

bps = bpm/60.0

seconds_per_mark =  beats_per_frame/bps
total_seconds = seconds_per_mark*N_frames

T = np.linspace(0, total_seconds, fps*total_seconds)

WEIGHTS = np.zeros(shape=(N_frames, len(T)))

for k in range(N_frames):
    
    s = seconds_per_mark
    WEIGHTS[k] = np.exp(-(T-k*s)**2/(s*sigma_weight))
    
    if k==0:
        WEIGHTS[k] += np.exp(-(T-N_frames*s)**2/(s*sigma_weight))

WEIGHTS /= WEIGHTS.sum(axis=0)

# Exegeration
WEIGHTS += 0.005*np.cos((np.pi/seconds_per_mark*beats_per_frame)*T.reshape(1,-1))**2
#########################################################################
os.system(f'rm -rf {os.path.join(save_dest,"*")}')

for k, w in tqdm(enumerate(WEIGHTS.T), total=len(T)):
  params = sum(w.reshape(-1,1)*MODELS)
  img = render_params(params, size=render_size)


  f_image = os.path.join(save_dest, f"{k:08d}.{extension}")
  imsave(f_image, img)

f_movie = "demo.mp4"

F_IMG = os.path.join(save_dest, f"%08d.{extension}")
cmd = f"avconv -y -r {fps} -i '{F_IMG}' -b:v 2400k {f_movie}"
os.system(cmd)

os.system(f'xdg-open {f_movie}')
        
