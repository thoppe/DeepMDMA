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

batch_size = 2
size_n = 200

#starting_training_steps = 2**10
#intermediate_training_steps = 2**6

starting_training_steps = 2**8
#intermediate_training_steps = 2**4

optimizer = tf.train.AdamOptimizer(0.005)
transforms=[]

save_image_dest = "results/images_direction"
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

#param_f = lambda: cppn(t_size, batch=batch_size)

param_f = lambda: param.image(200, batch=batch_size)

def render_set(n, channel, vec, train_n):

    neuron1 = ("mixed4a_pre_relu", 476)
    neuron2 = ("mixed4a_pre_relu", 460)
    obj = objectives.channel(*neuron1, batch=0) + objectives.channel(*neuron2, batch=1)

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
images = render_set(cn, channel, Y[0], starting_training_steps)

print (images[0] - images[1])
print (np.abs(images[0] - images[1]).sum())


for k, img in enumerate(images):
    f_img = os.path.join(save_image_dest, channel + f"_{cn}_{k:06d}.png")
    imsave(f_img, img)
    


#f_image = 'demo.png'
#imsave(f_image, img)

#for k,y in tqdm(enumerate(Y)):
#    f_img = os.path.join(save_image_dest, channel + f"_{cn}_{k:06d}.png")
#    img = render_set(cn, channel, y, intermediate_training_steps)
#    imsave(f_img, img)

