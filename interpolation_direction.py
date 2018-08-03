from lucid.misc.io import load
from scipy.misc import imsave
from lucid.misc.tfutil import create_session
import numpy as np
import tensorflow as tf
import os, glob
from lucid.optvis.param import cppn
from tqdm import tqdm
from IPython import embed
import pylab as plt

figure_size = 400
batch_size = 6
f_model = "results/models_direction/mixed4a_3x3_pre_relu_25_batches_6.ckpt"

save_dest = "results/direction_same_face"
os.system(f'mkdir -p {save_dest}')

sess = create_session()

def create_network():
  t_size = tf.placeholder_with_default(figure_size, [])
  nets = []
  with tf.variable_scope(f"CPPN"):
    for k in range(batch_size):
      with tf.variable_scope(f"CPPN_layer_{k}"):
        nets.append(cppn(t_size))
                
      return tf.concat(nets, axis=0)

T = create_network()

saver = tf.train.Saver()
saver.restore(sess, f_model)

for img in sess.run(T):
   print (img)

   plt.imshow(img)
   plt.show()
   exit()

embed()
exit()

def render_params(params, size=size_n):
  feed_dict = dict(zip(train_vars, params))
  feed_dict[t_size] = size
  return sess.run(T, feed_dict)


'''
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
'''

total_frames = 10


for frame_n in range(total_frames):
  print("Frameset",frame_n)

  links = [[j, j+1] for j in range(batch_size)]
  links[-1][-1] = 0
  print(links)

  t = frame_n / float(total_frames)

  
  for y0, y1 in Y:
    params = y0*m0 + y1*m1

    params *= 1.0 + y0*(1.0-y0)
    
    img = render_params(params, size=600)

    f_image = os.path.join(save_dest, f"{frame_n:08d}.png")
    imsave(f_image, img)

    frame_n += 1


