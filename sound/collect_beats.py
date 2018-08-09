import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import librosa.display
import os

f_wav = "secret_crates.wav"
y, sr = librosa.load(f_wav, duration=10**10)

f_beats = f_wav + '_beats.npy'
f_onset = f_wav + '_onset.npy'

bpm, beats = librosa.beat.beat_track(y, sr=sr, units='time')

n_fft = 1024
lag = 2
n_mels = 138
fmin = 27.5
fmax = 16000.
max_size = 3
hop_length = int(librosa.time_to_samples(1./200, sr=sr))

################################################################

odf_default = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
onset_default = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length,
                                           units='time')


S = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft,
                                   hop_length=hop_length,
                                   fmin=fmin,
                                   fmax=fmax,
                                   n_mels=n_mels)

odf_sf = librosa.onset.onset_strength(S=librosa.power_to_db(S, ref=np.max),
                                      sr=sr,
                                      hop_length=hop_length,
                                      lag=lag, max_size=max_size)

onset_sf = librosa.onset.onset_detect(onset_envelope=odf_sf,
                                      sr=sr,
                                      hop_length=hop_length,
                                      units='time')

print(beats)
print(onset_sf)

np.save(f_beats, beats)
np.save(f_onset, onset_sf)



################################################################

plt.figure(figsize=(6, 4))
librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                         y_axis='mel', x_axis='time', sr=sr,
                         hop_length=hop_length, fmin=fmin, fmax=fmax)

plt.vlines(beats, 0, 2**12, label='beats',color='w',lw=3,alpha=0.75)
plt.vlines(onset_sf, 0, 2**11, label='beats',color='k')

plt.tight_layout()
plt.show()
