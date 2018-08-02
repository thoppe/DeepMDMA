import numpy as np
import os
from tqdm import tqdm
#from IPython import embed

from CPPN_activations import *
from scipy.misc import imsave

import tensorflow as tf
from tensorflow.contrib import slim

from lucid.modelzoo import vision_models
from lucid.misc.io import show, save, load
from lucid.optvis import objectives
from lucid.optvis import render
from lucid.misc.tfutil import create_session

print ("Loading model")
model = vision_models.InceptionV1()
model.load_graphdef()

#size_n = 200
size_n = 200

#starting_training_steps = 2**10
#intermediate_training_steps = 2**6

starting_training_steps = 2**6
intermediate_training_steps = 2**4

optimizer = tf.train.AdamOptimizer(0.005)
transforms=[]

save_image_dest = "results/images_direction"
save_model_dest = "results/models_direction"
os.system(f'mkdir -p {save_model_dest}')
os.system(f'mkdir -p {save_image_dest}')

###########################################################################
pi = np.pi
dim = 204
N = 30
phase = np.random.uniform(0, 2*pi, dim)
period = np.random.uniform(0, pi/10, dim)
T = np.linspace(0,2*pi,N).reshape(-1,1)
Y = np.cos(period*T+phase)


sess = create_session()
t_size = tf.placeholder_with_default(size_n, [])
param_f = image_cppn(t_size)

def render_set(n, channel, vec, train_n):

    obj = objectives.direction_neuron(channel, vec)

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
    return images[0]
    

print("Training starting channel")
channel, cn = 'mixed4a_3x3_pre_relu', 25
img = render_set(cn, channel, Y[0], starting_training_steps)

f_image = 'demo.png'
imsave(f_image, img)

for k,y in tqdm(enumerate(Y)):
    f_img = os.path.join(save_image_dest, channel + f"_{cn}_{k:06d}.png")
    img = render_set(cn, channel, y, intermediate_training_steps)
    imsave(f_img, img)

