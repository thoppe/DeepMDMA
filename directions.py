import numpy as np
import os
from tqdm import tqdm
from IPython import embed

from scipy.misc import imsave

import tensorflow as tf
from lucid.modelzoo import vision_models
from lucid.misc.io import show, save, load
from lucid.optvis import objectives
from lucid.optvis import render
from lucid.misc.tfutil import create_session
from lucid.optvis.param import cppn
import lucid.optvis.param as param


print ("Loading model")
model = vision_models.InceptionV1()
model.load_graphdef()

batch_size = 6

# TEST: size_n lower and lower training steps still works w/lower quality

size_n = 100
starting_training_steps = 2**9

optimizer  = tf.train.AdamOptimizer(0.005)
transforms = []

save_image_dest = "results/images_direction"
os.system(f'mkdir -p {save_image_dest}')

save_model_dest = "results/models_direction"
os.system(f'mkdir -p {save_model_dest}')

###########################################################################
sess = create_session()

t_size = tf.placeholder_with_default(size_n, [])

def create_network():
    nets = []
    for k in range(batch_size):
        with tf.variable_scope(f"CPPN_layer_{k}"):
            nets.append(cppn(t_size))
                
    return tf.concat(nets, axis=0)

def render_set(n, channel, train_n):

    # Creates independent images
    param_f = create_network
    obj = sum(objectives.channel(channel, n, batch=i) for i in range(batch_size))

    obj += 0.01*objectives.alignment(channel, decay_ratio=3)

    # See more here
    # https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/differentiable-parameterizations/aligned_interpolation.ipynb#scrollTo=jOCYDhRrnPjp

    T = render.make_vis_T(
        model, obj,
        param_f=param_f,
        transforms=[],
        optimizer=optimizer, 
    )
    
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    
    for i in tqdm(range(train_n)):
      _, loss = sess.run([T("vis_op"), T("loss"), ])

      
    # Save trained variables
    f_model = os.path.join(save_model_dest, channel + f"_{n}_batches_{batch_size}.ckpt")
    save_path = saver.save(sess, f_model)
      
    # Return image
    images = T("input").eval({t_size: 600})
    return images
    

print("Training starting channel")

channel, cn = 'mixed4a_3x3_pre_relu', 25
images = render_set(cn, channel, starting_training_steps)

for k, img in enumerate(images):
    f_img = os.path.join(save_image_dest, channel + f"_{cn}_{k:06d}.png")
    imsave(f_img, img)    
