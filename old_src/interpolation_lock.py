from lucid.misc.io import load
from scipy.misc import imsave
from lucid.misc.tfutil import create_session
import numpy as np
import tensorflow as tf
import os, glob, collections
from tqdm import tqdm
import pylab as plt
import joblib
from src.locked_cppn import create_locked_network

#from IPython import embed

figure_size = 400
batch_size = 6
channel, cn = 'mixed4a_3x3_pre_relu', 25

num_layers = 8
num_shared_layers = 2


load_model_dest = 'results/models_lock/'
f_model = os.path.join(
  load_model_dest, 
  f"{channel}_{cn}_batches_{batch_size}.ckpt")


save_dest = "results/smooth_lock"
os.system(f'mkdir -p {save_dest}')

sess = create_session()

T = create_locked_network(
  batch_size, figure_size,
  num_layers=num_layers,
  num_shared_layers=num_shared_layers,
)

saver = tf.train.Saver()
saver.restore(sess, f_model)


class weighted_layers:
  def __init__(self):

    # Map the names of the layers to the TF variables
    layer_ref = {v.name:v for v in tf.trainable_variables()}

    # Extract the weights for blending later
    #W = dict(zip(layer_ref, sess.run( list(layer_ref.keys()))))
    self.ORG = {key : sess.run(layer_ref[key]) for key in layer_ref}

    # This is where we will write everything
    self.X = self.ORG.copy()

    self.clear()

  def clear(self):
    for key,val in self.X.items():
      self.X[key] = np.zeros_like(val)
      #self.X[key] += 1 ## WGM

    # Force 100% of the shared layers
    for key in self.X:      
      if f"CPPN_shared" not in key:
        continue      
      self.X[key] += self.ORG[key]

    self.weights = np.zeros(batch_size)

  def add_fraction(self, i, j, fraction):

    for key in self.X:      
      if f"CPPN_layer_{i}" not in key:
        continue

      key2 = key.replace(f"CPPN_layer_{i}", f"CPPN_layer_{j}")

      self.X[key] += self.ORG[key2]*fraction
      #self.X[key] *= self.ORG[key2]**fraction ## WGM
      
    self.weights[i] += fraction

  def multiply(self, fraction):
    for key in self.X:
      if f"CPPN_shared" in key:
        continue      
      self.X[key] *= fraction

  def render(self):  
    images = sess.run(T, feed_dict=self.X)
    return images

  def show(self, i):
    plt.imshow(self.render()[i])

W = weighted_layers()

#embed()
total_frames = 30

def save_image(img, frame_idx):
  f_image = os.path.join(save_dest, f"{frame_idx:08d}.png")
  imsave(f_image, img)

dfunc = joblib.delayed(save_image)
MP = joblib.Parallel(-1)


for frame_n in range(total_frames):
  print("Frameset",frame_n)
  
  t = frame_n / float(total_frames)
  t = (1.0-np.cos(np.pi*t))/2.0

  W.clear()

  for i in range(batch_size):
    j = (i+1)%batch_size
    W.add_fraction(i, i, (1-t))
    W.add_fraction(i, j, t)

  # Exaggeraton step (comment out for smoothness)
  W.multiply(1 + t*(1-t))

  images = W.render()
  MP(dfunc(img, frame_n + k*(total_frames)) for k, img in enumerate(images))
