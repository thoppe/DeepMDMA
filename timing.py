import numpy as np
import pylab as plt

beats_per_frame = 4
sigma_weight = 1/1.5
N_frames = 5

bpm = 80
fps = 30

bps = bpm/60.0

seconds_per_mark =  beats_per_frame/bps
total_seconds = seconds_per_mark*N_frames

T = np.linspace(0, total_seconds, fps*total_seconds)

WEIGHTS = np.zeros(shape=(N_frames, len(T)))

for k in range(N_frames):
    
    s = seconds_per_mark
    WEIGHTS[k] = np.exp(-(T-k*s)**2/(s*sigma_weight))
    
    if k==0:
        WEIGHTS[k] += np.exp(-(T-N_frames*s)**2/(s*sigma_weight))

WEIGHTS /= WEIGHTS.sum(axis=0)

WEIGHTS += 0.25*np.cos((np.pi/seconds_per_mark*beats_per_frame)*T.reshape(1,-1))**2

    
for k,Y in enumerate(WEIGHTS):
    plt.plot(T,Y,label=k)

plt.legend()
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
