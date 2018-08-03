import numpy as np
import os
from tqdm import tqdm
#from IPython import embed

from scipy.misc import imsave

import tensorflow as tf
from tensorflow.contrib import slim

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

batch_size = 3

# TEST: size_n lower and lower training steps still works w/lower quality

size_n = 100
starting_training_steps = 2**9

optimizer  = tf.train.AdamOptimizer(0.005)
transforms = []

save_image_dest = "results/images_direction"
os.system(f'mkdir -p {save_image_dest}')

###########################################################################
sess = create_session()
t_size = tf.placeholder_with_default(size_n, [])
param_f = lambda: tf.concat([cppn(t_size) for _ in range(batch_size)], axis=0)

def render_set(n, channel, train_n):

    neuron1 = (channel, n)
    neuron2 = (channel, n)

    # Sum objective on each channel oversaturates
    obj = objectives.channel(*neuron1, batch=0) + objectives.channel(*neuron2, batch=1)

    # This doesn't oversaturate, but images are the same
    #obj = objectives.channel(*neuron1, batch=0)

    # This does absolutely nothing to mix images (either plus or minus)
    #obj -= 1e2*objectives.diversity("mixed5a")

    '''
    interpolation_objective = objectives.channel_interpolate(*neuron1, *neuron2)
    alignment_objective = (
        objectives.alignment('mixed4a', decay_ratio=5) +
        objectives.alignment('mixed4b', decay_ratio=5)
    )
    obj = interpolation_objective + 1e-1 * alignment_objective
    '''

    # See more here
    # https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/differentiable-parameterizations/aligned_interpolation.ipynb#scrollTo=jOCYDhRrnPjp
    #interpolation_objective = objectives.channel_interpolate(*neuron1, *neuron2)
    #alignment_objective = objectives.alignment('mixed3b', decay_ratio=5) + objectives.alignment('mixed4a', decay_ratio=5)

    T = render.make_vis_T(
        model, obj,
        param_f=param_f,
        transforms=[],
        optimizer=optimizer, 
    )
    tf.global_variables_initializer().run()
    
    for i in tqdm(range(train_n)):
      _, loss = sess.run([T("vis_op"), T("loss"), ])
      
    # Return image
    images = T("input").eval({t_size: 600})
    return images
    

print("Training starting channel")

channel, cn = 'mixed4a_3x3_pre_relu', 25
images = render_set(cn, channel, starting_training_steps)

for k, img in enumerate(images):
    f_img = os.path.join(save_image_dest, channel + f"_{cn}_{k:06d}.png")
    imsave(f_img, img)    




#f_image = 'demo.png'
#imsave(f_image, img)

#for k,y in tqdm(enumerate(Y)):
#    f_img = os.path.join(save_image_dest, channel + f"_{cn}_{k:06d}.png")
#    img = render_set(cn, channel, y, intermediate_training_steps)
#    imsave(f_img, img)

