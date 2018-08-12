"""Render activations from the Inception model

Usage:
  render_activations.py [<channel>] [<k>] [--n_training=<n>] [--output_image_size=<n>] [--model_size=<n>]

Options:
  channel               Specify a channel (default, all channels)
  k                     Specify a color in the channel (default, all valid)
  --n_training=<n>      Number of training steps [default: 1024]
  --output_image_size=<n>      Square size of image [default: 600]
  --model_size=<n>      Size of the model CCN, don't change? [default: 200]
  -h --help             Show this screen.
"""

from docopt import docopt
dargs = docopt(__doc__)

print (f"Start {dargs}")

import numpy as np
import os
from tqdm import tqdm

from scipy.misc import imsave

import tensorflow as tf
from tensorflow.contrib import slim

from lucid.modelzoo import vision_models
from lucid.misc.io import show, save, load
from lucid.optvis import objectives
from lucid.optvis import render
from lucid.misc.tfutil import create_session
from lucid.optvis.param import cppn

def render_set(n, channel):
    
    print ("Starting", channel, n)
    obj = objectives.channel(channel, n)

    # Add this to "sharpen" the image... too much and it gets crazy
    #obj += 0.001*objectives.total_variation()

    sess = create_session()
    t_size = tf.placeholder_with_default(size_n, [])

    T = render.make_vis_T(
        model, obj,
        param_f=lambda: cppn(t_size),
        transforms=[],
        optimizer=optimizer, 
    )
    tf.global_variables_initializer().run()
    
    for i in tqdm(range(training_steps)):
      _, loss = sess.run([T("vis_op"), T("loss"), ])

    # Save trained variables
    train_vars = sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    params = np.array(sess.run(train_vars), object)

    f_model = os.path.join(save_model_dest, channel + f"_{n}.npy")
    save(params, f_model)
  
    # Save final image
    images = T("input").eval({t_size: image_size})
    img = images[0]
    sess.close()
    
    f_image = os.path.join(save_image_dest, channel + f"_{n}.png")
    imsave(f_image, img)


print ("Loading model")
model = vision_models.InceptionV1()
model.load_graphdef()

size_n = int(dargs["--model_size"])
training_steps = int(float(dargs["--n_training"]))
image_size = int(dargs["--output_image_size"])

optimizer = tf.train.AdamOptimizer(0.005)
transforms=[]

save_image_dest = "results/images"
save_model_dest = "results/models"
os.system(f'mkdir -p {save_model_dest}')
os.system(f'mkdir -p {save_image_dest}')

if not dargs["<channel>"]:
    CHANNELS = [
        "mixed4a_3x3_pre_relu",
        "mixed4b_3x3_pre_relu",
        "mixed4c_3x3_pre_relu",
        "mixed4d_3x3_pre_relu",
        "mixed4e_3x3_pre_relu",
    ]
else:
    CHANNELS = [dargs["<channel>"],]

if not dargs["<k>"]:
    COLORSET = range(2**10)
else:
    COLORSET = [int(dargs["<k>"]),]


for channel in CHANNELS:
    for n in COLORSET:

        f_image = os.path.join(save_image_dest, channel + f"_{n}.jpg")
        if os.path.exists(f_image):
            continue
        print("Starting", f_image)
        try:
            render_set(n, channel)
        except Exception as EX:
            print("EXCEPTION", channel, EX)
            break

