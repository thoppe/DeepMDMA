from lucid.misc.io import load
from scipy.misc import imsave
from lucid.misc.tfutil import create_session
import tensorflow as tf
from lucid.optvis.param import cppn
import numpy as np
import librosa
import os, glob
from tqdm import tqdm

render_duration = 200.0
render_size = 1280
extension = 'png'
model_cutoff = 200
#bitrate = 2400
#bitrate = 6400
bitrate = 12800

beats_per_frame = 4
sigma_weight = 1/2.5
exageration_weight = 0.10
exageration_sigma = 1/5.0
fps = 30

f_wav = "sound/secret_crates.wav"
WAV,sr = librosa.load(f_wav,duration=render_duration)
total_seconds = librosa.get_duration(WAV, sr)

save_dest = "results/interpolation_matching_beats"
os.system(f'mkdir -p {save_dest}')

f_beats = f_wav + '_beats.npy'
f_onset = f_wav + '_onset.npy'
beats = np.load(f_beats)
onsets = np.load(f_onset)

sess = create_session()
t_size = tf.placeholder_with_default(200, [])
t_image = cppn(t_size)
train_vars = sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

def render_params(params, size=224):
  feed_dict = dict(zip(train_vars, params))
  feed_dict[t_size] = size
  return sess.run(t_image, feed_dict)[0]

f_models = sorted(glob.glob("results/smooth_models/*.npy"))

print("Loading models")
MODELS = list(map(load, tqdm(f_models[:model_cutoff])))

N_frames = len(MODELS)


T = np.linspace(0, total_seconds, fps*total_seconds)
WEIGHTS = np.zeros(shape=(N_frames, len(T)))

print("Building weights")

for k in range(N_frames):
    mu = beats[k]*beats_per_frame
    WEIGHTS[k] = np.exp(-(T-mu)**2/sigma_weight)

WEIGHTS /= WEIGHTS.sum(axis=0)
    
for mu in onsets:
    X = exageration_weight*np.exp(-(T-mu)**2/exageration_sigma**2)  
    WEIGHTS += WEIGHTS*X

print (WEIGHTS)

#########################################################################
print("Rendering")
os.system(f'rm -rf {os.path.join(save_dest,"*")}')


for k, w in tqdm(enumerate(WEIGHTS.T), total=len(T)):
  params = sum(w.reshape(-1,1)*MODELS)
  img = render_params(params, size=render_size)

  f_image = os.path.join(save_dest, f"{k:08d}.{extension}")
  imsave(f_image, img)

f_movie = "demo.mp4"

F_IMG = os.path.join(save_dest, f"%08d.{extension}")
cmd = f"avconv -y -r {fps} -i '{F_IMG}' -i {f_wav} -c:a aac -ab 112k -c:v libx264 -shortest -b:v {bitrate}k {f_movie} "
print(cmd)
os.system(cmd)

os.system(f'xdg-open {f_movie}')
