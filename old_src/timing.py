import numpy as np
import os
import pylab as plt
import librosa

f_wav = "sound/secret_crates.wav"

WAV,sr = librosa.load(f_wav,duration=8.2)
total_seconds = librosa.get_duration(WAV, sr)

f_beats = f_wav + '_beats.npy'
f_onset = f_wav + '_onset.npy'
beats = np.load(f_beats)
onsets = np.load(f_onset)

# If our duration is shorter than the time, truncate
beats = beats[beats<=total_seconds]
onsets = onsets[onsets<=total_seconds]
N_frames = 15

beats_per_frame = 4
sigma_weight = 1/2.5
exageration_weight = 0.10
exageration_sigma = 1/5.0
fps = 30


#seconds_per_mark =  beats_per_frame/bps
#total_seconds = seconds_per_mark*N_frames
N_frame = 4

T = np.linspace(0, total_seconds, fps*total_seconds)
WEIGHTS = np.zeros(shape=(N_frames, len(T)))

for k in range(N_frames):
    mu = beats[k]*beats_per_frame
    WEIGHTS[k] = np.exp(-(T-mu)**2/sigma_weight)

WEIGHTS /= WEIGHTS.sum(axis=0)
    
for mu in onsets:
    X = exageration_weight*np.exp(-(T-mu)**2/exageration_sigma**2)
    
    print ((X*WEIGHTS).sum())
    WEIGHTS += WEIGHTS*X
  
for k,Y in enumerate(WEIGHTS):
    plt.plot(T,Y,label=k)
plt.vlines(onsets, 0, .1, label='beats',color='k')

#plt.legend()
plt.show()
exit()





phase = np.random.uniform(0, 2*pi, dim)
period = np.random.uniform(0, pi/10, dim)
T = np.linspace(0,2*pi,N).reshape(-1,1)
Y = np.cos(period*T+phase)

#print (Y.shape)
#exit()

plt.plot(Y)
plt.show()
print (Y)


'''
N = 30
mu = np.zeros(shape=(N,))
r = 0.01*np.random.uniform(-1,1,size=(N,N)) + np.ones(shape=(N,N))
y = np.random.multivariate_normal(mu, r, size=N)
plt.plot(y[0])
plt.plot(y.T[0])
plt.ylim(-3,3)
plt.show()
'''


exit()


print (dim)


'''
bps = 120

fps = 30

pi = np.pi

period = 2*pi
duration = 60.0/bps
n_frames = (fps*duration)

T = np.linspace(0, duration, n_frames).reshape(-1,1)
period = [period, period]
phase  = [0, np.pi]

Y = np.cos(period*T+phase)
Y = np.sin(Y*(pi/2))
Y = np.sin(Y*(pi/2))
#Y = np.sin(Y*(pi/2))
#Y = np.sin(Y*(pi/2))
Y = (Y+1)/2

for y in Y.T:
    print(y.shape, T.shape)
    plt.plot(T, y)
plt.show()
'''
