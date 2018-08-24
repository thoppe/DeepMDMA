"""Render activations from the Inception model

Usage:
  render_activations.py [<channel> <k>] [options]

Options:
  channel               Specify a channel (default, all channels)
  k                     Specify a color in the channel (default, all valid)
  --n_training=<n>      Number of training steps [default: 256]
  -o --output_image_size=<n>      Square size of image [default: 600]
  --model_size=<n>      Size of the model CCN, don't change? [default: 200]
  -h --help             Show this screen.
"""

import numpy as np
import os
from tqdm import tqdm

from scipy.misc import imsave

import tensorflow as tf
from tensorflow.contrib import slim

from lucid.modelzoo import vision_models
from lucid.misc.io import show, save, load
from lucid.optvis import objectives, transform
from lucid.optvis import render
from lucid.misc.tfutil import create_session
from lucid.optvis.param import cppn
from lucid.optvis.param import image, image_sample

from lucid.optvis.render import import_model, make_transform_f, make_t_image

def render_set(n, channel):
    
    print ("Starting", channel, n)
    obj = objectives.channel(channel, n)

    # Add this to "sharpen" the image... too much and it gets crazy
    #obj += 0.001*objectives.total_variation()

    sess = create_session()
    t_size = tf.placeholder_with_default(size_n, [])
    
    f_model = os.path.join(save_model_dest, channel + f"_{n}.npy")

    '''
    T = render.make_vis_T(
        model, obj,
        param_f=lambda: cppn(t_size),
        transforms=[],
        optimizer=optimizer, 
    )
    '''

    #### NEED TO MAP THE T_IMAGES TOGETHER
    t_image = make_t_image(lambda:cppn(t_size))
    objective_f = objectives.as_objective(obj)
    optimizer = tf.train.AdamOptimizer(0.005)
    transform_f = make_transform_f([])

    global_step = tf.train.get_or_create_global_step()
    init_global_step = tf.variables_initializer([global_step])
    init_global_step.run()


    T = import_model(model, transform_f(t_image), t_image)

    obj2 = objectives.as_objective(obj)(T)

    #L1 = objective_f*((1.0-t_alpha_mean_full)*0.5)
    #L2 = obj*(1.0-t_alpha_mean_crop)

    #tf.losses.add_loss(-obj*(1.0-t_alpha_mean_full)*0.5)
    tf.losses.add_loss(-objective_f(T)*(1.0-t_alpha_mean_crop))
    
    #tf.losses.add_loss(-objective_f(T))
    
    t_loss = tf.losses.get_total_loss()

    vis_op = optimizer.minimize(t_loss, global_step=global_step)
    #vis_op = optimizer.minimize(-LX, global_step=global_step)

    
    tf.global_variables_initializer().run()
    train_vars = sess.graph.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES)


    if not os.path.exists(f_model):

        for i in tqdm(range(training_steps)):
            
          _, loss = sess.run([vis_op, t_loss])

        # Save trained variables
        params = np.array(sess.run(train_vars), object)
        save(params, f_model)
    else:
        params = load(f_model)
    
    # Save final image
    feed_dict = dict(zip(train_vars, params))
    feed_dict[t_size] = image_size
    images = T("input").eval(feed_dict)
    img = images[0]
    sess.close()
    
    f_image = os.path.join(save_image_dest, channel + f"_{n}.jpg")
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

optimizer = tf.train.AdamOptimizer(0.005)
transforms=[]


# Load the alpha channels
def linear2gamma(a):
  return a**(1.0/2.2)

def gamma2linear(a):
  return a**2.2

w, h = int(dargs["--model_size"]), int(dargs["--model_size"])
t_image = image(h, w, decorrelate=True, fft=True, alpha=True)
t_rgb = t_image[...,:3]
t_alpha = t_image[...,3:]
t_bg = image_sample([1, h, w, 3], sd=0.2, decay_power=1.5)
  
t_composed = t_bg*(1.0-t_alpha) + t_rgb*t_alpha
t_composed = tf.concat([t_composed, t_alpha], -1)
t_crop = transform.random_scale([0.6, 0.7, 0.8, 0.9, 1.0, 1.1])(t_composed)
t_crop = tf.random_crop(t_crop, [1, 160, 160, 4])
t_crop_rgb, t_crop_alpha = t_crop[..., :3], t_crop[..., 3:]
t_crop_rgb = linear2gamma(t_crop_rgb)
model.import_graph(t_crop_rgb)

t_alpha_mean_crop = tf.reduce_mean(t_crop_alpha)
t_alpha_mean_full = tf.reduce_mean(t_alpha)


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
        #try:
        render_set(n, channel)
        #except Exception as EX:
        #    print("EXCEPTION", channel, EX)
        #    break

