import numpy as np
import os
from tqdm import tqdm
#from IPython import embed

from scipy.misc import imsave
import tensorflow as tf

from lucid.modelzoo import vision_models
from lucid.misc.io import save, load
from lucid.optvis import objectives
from lucid.optvis import render
from lucid.misc.tfutil import create_session
from lucid.optvis.param import cppn

print ("Loading model")
model = vision_models.InceptionV1()
model.load_graphdef()

size_n = 200

n_frames = 5
starting_training_steps = 2**10
additional_training_steps = 2**9
prior_threshold = 0.4


optimizer = tf.train.AdamOptimizer(0.005)
transforms=[]

save_image_dest = "results/smooth_images"
save_model_dest = "results/smooth_models"
os.system(f'mkdir -p {save_model_dest}')
os.system(f'mkdir -p {save_image_dest}')


def render_set(
        channel, n_iter,
        prefix, starting_pos=None,
        force=False,
        objective=None,
):

    f_model = os.path.join(save_model_dest, channel + f"_{prefix}.npy")
    f_image = os.path.join(save_image_dest, channel + f"_{prefix}.png")
    if os.path.exists(f_model) and not force:
        return True

    print ("Starting", channel, prefix)
    obj = objective

    # Add this to "sharpen" the image... too much and it gets crazy
    #obj += 0.001*objectives.total_variation()

    sess = create_session()
    t_size = tf.placeholder_with_default(size_n, [])

    param_f = lambda: cppn(t_size)

    T = render.make_vis_T(
        model, obj,
        param_f=param_f,
        transforms=[],
        optimizer=optimizer, 
    )
    tf.global_variables_initializer().run()

    # Assign the starting weights
    if starting_pos is not None:
        for v,x in zip(tf.trainable_variables(), starting_pos):
            sess.run(tf.assign(v,x))
    
    for i in tqdm(range(n_iter)):
      _, loss = sess.run([T("vis_op"), T("loss"), ])

    # Save trained variables
    train_vars = sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    params = np.array(sess.run(train_vars), object)

    save(params, f_model)
  
    # Save final image
    images = T("input").eval({t_size: 600})
    img = images[0]
    sess.close()
    
    imsave(f_image, img)


channel = 'mixed4a_3x3_pre_relu'

CN = [25, 17, 1]


C = [objectives.channel(channel, cn) for cn in CN]

#C0 = objectives.channel(channel, cn0)
#C1 = objectives.channel(channel, cn1)

# Render the fixed points

for kn, obj in enumerate(C):
    render_set(channel, starting_training_steps,
               f'A{kn}', objective=obj)

#render_set(channel, starting_training_steps, 'A0', objective=C0)
#render_set(channel, starting_training_steps, 'A1', objective=C1)

MODELS = [
    load(os.path.join(save_model_dest, channel + f"_A{kn}.npy"))
    for kn in range(len(C))
]


#f_M0 = os.path.join(save_model_dest, channel + f"_A0.npy")
#f_M1 = os.path.join(save_model_dest, channel + f"_A1.npy")
#assert(os.path.exists(f_M0))
#assert(os.path.exists(f_M1))
#M0 = load(f_M0)
#M1 = load(f_M1)


model_n = 0
for i in range(len(C)-1):
    for p in np.linspace(0, 1, n_frames):
        print(f"Starting {i}, {p}")

        M0 = MODELS[i]
        M1 = MODELS[i+1]

        obj0 = C[i]
        obj1 = C[i+1]

        pos = p*M1 + (1-p)*M0
        obj = p*obj0 + (1-p)*obj1

        label = f"{model_n:08d}"
        prior_label = f"{model_n-1:08d}"

        if model_n>0:
            f_MX = os.path.join(save_model_dest, channel +
                                f"_{prior_label}.npy")
            MX = load(f_MX)
            pos = (pos + prior_threshold*MX) / (1+prior_threshold)


        render_set(
            channel,
            additional_training_steps,
            label,
            starting_pos=pos,
            force=True,
            objective=obj,
        )
        
        model_n += 1
        

'''
for k,
    print ("Starting", k,p)
    pos = p*M1 + (1-p)*M0
    obj = p*C1 + (1-p)*C0

    label = f"{k:08d}"
    prior_label = f"{k-1:08d}"

    if k>0:
        f_MX = os.path.join(save_model_dest, channel +
                            f"_{prior_label}.npy")
        MX = load(f_MX)

        pos = (pos + prior_threshold*MX) / (1+prior_threshold)


    render_set(
        channel,
        additional_training_steps,
        label,
        starting_pos = pos,
        force=True,
        objective=obj,
    )
'''
