"""Render activations from the Inception model

Usage:
  VAE_build.py <channel> <k> [options]

Options:
  channel               Specify a channel (default, all channels)
  k                     Specify a color in the channel (default, all valid)
  --n_models=<n>        Number of models to sample [default: 256]
  --n_training=<n>      Number of training steps [default: 1024]
  -o --output_image_size=<n>      Square size of image [default: 600]
  --model_size=<n>      Size of the model CCN, don't change? [default: 200]
  -h --help             Show this screen.
"""
# http://louistiao.me/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/

import numpy as np
import os
from tqdm import tqdm

from scipy.misc import imsave

import tensorflow as tf
from lucid.modelzoo import vision_models
from lucid.misc.io import save, load
from lucid.optvis import objectives, render
from lucid.misc.tfutil import create_session
from lucid.optvis.param import cppn

def render_set(n, channel, m):

    f_image = os.path.join(save_image_dest, channel + f"_{n}_{m:04d}.jpg")
    f_model = os.path.join(save_model_dest, channel + f"_{n}_{m:04d}.npy")

    if os.path.exists(f_image):
        return False
    
    print ("Starting", channel, n, m)
    obj = objectives.channel(channel, n)

    # Add this to "sharpen" the image... too much and it gets crazy
    #obj += 0.001*objectives.total_variation()

    sess = create_session()
    t_size = tf.placeholder_with_default(size_n, [])

    optimizer = tf.train.AdamOptimizer(0.005)

    T = render.make_vis_T(
        model, obj,
        param_f=lambda: cppn(t_size),
        transforms=[],
        optimizer=optimizer, 
    )
    tf.global_variables_initializer().run()
    train_vars = sess.graph.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES)

    if not os.path.exists(f_model):
        for i in tqdm(range(training_steps)):
          _, loss = sess.run([T("vis_op"), T("loss"), ])

        # Save trained variables
        params = np.array(sess.run(train_vars), object)
        save(params, f_model)
    else:
        params = load(f_model)
        
    print (params)
    
    # Save final image
    feed_dict = dict(zip(train_vars, params))
    feed_dict[t_size] = image_size
    images = T("input").eval(feed_dict)
    img = images[0]
    sess.close()
    
    imsave(f_image, img)
    print(f"Saved to {f_image}")


from docopt import docopt
dargs = docopt(__doc__)

print (f"Start {dargs}")

print ("Loading model")
model = vision_models.InceptionV1()
model.load_graphdef()

size_n = int(dargs["--model_size"])
training_steps = int(float(dargs["--n_training"]))
image_size = int(dargs["--output_image_size"])

save_image_dest = "results/VAE_base_images"
save_model_dest = "results/VAE_base_models"
os.system(f'mkdir -p {save_model_dest}')
os.system(f'mkdir -p {save_image_dest}')

channel = dargs["<channel>"]
colorset = int(dargs["<k>"])

n_models = int(dargs["--n_models"])

for m in range(n_models):
    render_set(colorset, channel, m)
