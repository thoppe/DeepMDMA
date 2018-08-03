from lucid.misc.io import load
from scipy.misc import imsave
from lucid.misc.tfutil import create_session
import numpy as np
import tensorflow as tf
import os, glob, collections
from lucid.optvis.param import cppn
from tqdm import tqdm
import pylab as plt
import joblib

#from IPython import embed

figure_size = 400
batch_size = 6
f_model = "results/models_direction/mixed4a_3x3_pre_relu_25_batches_6.ckpt"

save_dest = "results/direction_same_face"
os.system(f'mkdir -p {save_dest}')

sess = create_session()

def create_network():
  t_size = tf.placeholder_with_default(figure_size, [])
  nets = []
  for k in range(batch_size):
    with tf.variable_scope(f"CPPN_layer_{k}"):
      nets.append(cppn(t_size))
                
  return tf.concat(nets, axis=0)

T = create_network()
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

  def add_fraction(self, i, j, fraction):

    for key in self.X:      
      if f"CPPN_layer_{i}" not in key:
        continue

      key2 = key.replace(f"CPPN_layer_{i}", f"CPPN_layer_{j}")
      self.X[key] += fraction*self.ORG[key2]

  def multiply(self, fraction):
    for key in self.X:
      self.X[key] *= fraction

  def render(self):
    images = sess.run(T, feed_dict=self.X)
    return images

  def show(self, i):
    plt.imshow(self.render()[i])

W = weighted_layers()
W.add_fraction(0,1,1)
W.show(0)

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

  '''
  for k, img in enumerate(images):
    frame_idx = frame_n + k*(total_frames)
    print (frame_n, k, frame_idx)
    
    save_image(img, frame_idx)

    #f_image = os.path.join(save_dest, f"{frame_idx:08d}.png")
    #imsave(f_image, img)
  '''
