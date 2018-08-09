from __future__ import print_function
import numpy as np
import scipy
import matplotlib.pyplot as plt
import librosa
import librosa.display

f_wav = "Ratatat_Loud_Pipes_clip.wav"
y, sr = librosa.load(f_wav, duration=25)

#f_wav = "Ratatat_Cream_clip.wav"
#y, sr = librosa.load(f_wav, duration=25, offset=0)

bpm, beats = librosa.beat.beat_track(y, sr=sr, units='time')

n_fft = 1024
lag = 2
n_mels = 138
fmin = 27.5
fmax = 16000.
max_size = 3
hop_length = int(librosa.time_to_samples(1./200, sr=sr))

S = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft,
                                   hop_length=hop_length,
                                   fmin=fmin,
                                   fmax=fmax,
                                   n_mels=n_mels)


################################################################

odf_default = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
onset_default = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length,
                                           units='time')

odf_sf = librosa.onset.onset_strength(S=librosa.power_to_db(S, ref=np.max),
                                      sr=sr,
                                      hop_length=hop_length,
                                      lag=lag, max_size=max_size)

onset_sf = librosa.onset.onset_detect(onset_envelope=odf_sf,
                                      sr=sr,
                                      hop_length=hop_length,
                                      units='time')


################################################################


plt.figure(figsize=(6, 4))
librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                         y_axis='mel', x_axis='time', sr=sr,
                         hop_length=hop_length, fmin=fmin, fmax=fmax)

plt.vlines(beats, 0, 2**12, label='beats',color='w',lw=3,alpha=0.75)
plt.vlines(onset_sf, 0, 2**11, label='beats',color='k')

plt.tight_layout()
plt.show()
exit()

################################################################


# sphinx_gallery_thumbnail_number = 2
plt.figure(figsize=(6, 6))

frame_time = librosa.frames_to_time(np.arange(len(odf_default)),
                                    sr=sr,
                                    hop_length=hop_length)

ax = plt.subplot(2, 1, 2)
librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                         y_axis='mel', x_axis='time', sr=sr,
                         hop_length=hop_length, fmin=fmin, fmax=fmax)
#plt.xlim([0, 5.0])
plt.axis('tight')


plt.subplot(4, 1, 1, sharex=ax)
plt.plot(frame_time, odf_default, label='Spectral flux')
plt.vlines(onset_default, 0, odf_default.max(), label='Onsets')
#plt.xlim([0, 5.0])
plt.legend()


plt.subplot(4, 1, 2, sharex=ax)
plt.plot(frame_time, odf_sf, color='g', label='Superflux')
plt.vlines(onset_sf, 0, odf_sf.max(), label='Onsets')
#plt.xlim([0, 5.0])
plt.legend()

plt.tight_layout()
plt.show()
plt.show()
print (S)

