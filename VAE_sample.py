from VAE_model import build_model, pack, unpack
import glob
import numpy as np
from tqdm import tqdm
from lucid.misc.io import load
from lucid.optvis.param import cppn
from lucid.optvis import objectives
from lucid.optvis import render
from lucid.misc.tfutil import create_session
import tensorflow as tf
from lucid.modelzoo import vision_models


cutoff = 2**20
image_size = 400
size_n = 200
latent_n = 200

F_MODELS = glob.glob("results/VAE_base_models/*.npy")[:cutoff]

# Load the save model params
raw_params = [load(f_model) for f_model in tqdm(F_MODELS)]
shapes = list(map(lambda x:x.shape, raw_params[0]))
X = np.array([unpack(p) for p in raw_params])


#import pylab as plt
#import seaborn as sns
#print (X.mean(axis=1))
#sns.distplot(X.mean(axis=1))
#plt.figure()
#sns.distplot(X.std(axis=1))
#plt.show()
#exit()

class render_model():
    def __init__(self):

        print ("Loading Inception model")
        self.model = vision_models.InceptionV1()
        self.model.load_graphdef()

        sess = create_session()

        self.t_size = tf.placeholder_with_default(size_n, [])
        obj = objectives.channel("mixed4a", 0)
        self.T = render.make_vis_T(
            self.model, obj,
            param_f=lambda: cppn(self.t_size),
        )

        tf.global_variables_initializer().run()
        self.train_vars = sess.graph.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES)

    def __call__(self, params):

        feed_dict = dict(zip(self.train_vars, params))
        feed_dict[self.t_size] = image_size
        return self.T("input").eval(feed_dict)[0]

#####################################################################
intermediate_n = 512
input_n = 8451
f_h5 = 'results/VAE_weights.h5'

VAE, encoder, decoder = build_model(
    input_n, intermediate_n, latent_n)
VAE = VAE.load_weights(f_h5)

z_mean, z_std_log, z = encoder.predict(X, batch_size=128)
XR = decoder.predict(z, batch_size=128)


import pylab as plt
import seaborn as sns
sns.distplot(X.mean(axis=1), label="Org")
sns.distplot(XR.mean(axis=1), label="Reconstructed")
plt.legend()
plt.figure()

sns.distplot(X.std(axis=1), label="Org")
sns.distplot(XR.std(axis=1), label="Reconstructed")
plt.legend()
plt.show()
exit()



print (z_mean)


params0 = pack(X[0], shapes)
params1 = pack(xp[0], shapes)
#######################################################################
#from IPython import embed; embed()

R = render_model()
img0 = R(params0)
img1 = R(params1)

#print(img0 - img1)

import pylab as plt
plt.imshow(img0)
plt.figure()
plt.imshow(img1)

plt.show()
exit()

