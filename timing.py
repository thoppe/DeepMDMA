import numpy as np
import pylab as plt
pi = np.pi

dim = 208
N = 30
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
