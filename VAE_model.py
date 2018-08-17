import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import numpy as np

# See this site for more information
#https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py



def unpack(X):
    return np.hstack([x.ravel() for x in X])

def pack(X, shapes):
    data, i = [], 0
    for size in shapes:
        n = np.prod(size)
        
        data.append(X[i:i+n].reshape(size))
        i+=n
    return np.array(data)



def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def build_model(input_n, intermediate_n, latent_n):

    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=[input_n,], name='encoder_input')
    Y = Dense(intermediate_n, activation='relu')(inputs)

    z_mean = Dense(latent_n, name='z_mean')(Y)
    z_log_var = Dense(latent_n, name='z_log_var')(Y)

    # use reparameterization trick to push the sampling out as input
    z = Lambda(sampling, output_shape=(latent_n,), name='z')(
        [z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    #print (encoder.summary())

    # build decoder model
    latent_inputs = Input(shape=(latent_n,), name='z_sampling')
    Zp = Dense(intermediate_n, activation='relu')(latent_inputs)
    outputs = Dense(input_n, activation='sigmoid')(Zp)
    decoder = Model(latent_inputs, outputs, name='decoder')

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    VAE = Model(inputs, outputs, name='vae_mlp')

    # Add loss functions and compile
    reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss *= input_n
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = K.mean(reconstruction_loss + kl_loss)
    VAE.add_loss(vae_loss)
    VAE.compile(optimizer='adam')
    
    return VAE, encoder, decoder
