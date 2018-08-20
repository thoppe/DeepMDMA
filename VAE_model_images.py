import tensorflow as tf
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import numpy as np

# See this site for more information
# https://blog.keras.io/building-autoencoders-in-keras.html

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D

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

    input_img = Input(shape=(input_n, input_n, 3))

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (75, 75, 8) i.e. 128-dimensional
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    print(autoencoder.summary())
    exit()

    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=[input_n,], name='encoder_input')
    #Y = Dense(intermediate_n, activation='relu')(inputs)
    Y = Dense(intermediate_n, activation=None)(inputs)

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

    #outputs = Dense(input_n, activation='sigmoid')(Zp)
    outputs = Dense(input_n, activation=None)(Zp)
    
    decoder = Model(latent_inputs, outputs, name='decoder')

    # instantiate VAE model
    reconstructed_input = decoder(encoder(inputs)[2])
    VAE = Model(inputs, reconstructed_input, name='vae_mlp')

    # Add loss functions and compile
    reconstruction_loss = mse(inputs, reconstructed_input)
    #reconstruction_loss =binary_crossentropy(inputs, reconstructed_input)
    
    reconstruction_loss *= input_n
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = K.mean(reconstruction_loss + kl_loss)
    VAE.add_loss(vae_loss)
    VAE.compile(optimizer='adam')
    
    return VAE, encoder, decoder
