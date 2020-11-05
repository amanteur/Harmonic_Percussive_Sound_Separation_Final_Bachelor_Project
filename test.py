from numpy import sin, linspace, pi, abs, angle, diff, unwrap

from numpy.random import randn
import numpy as np
# from scipy.signal.spectral import _spectral_helper
from scipy.signal import spectrogram
from matplotlib.mlab import _spectral_helper
import matplotlib.pyplot as plt



fs = 100000

t = linspace(0, 1, fs)

f = 15000



# Slight frequency sweep, some noise to break up the phase in the quiet parts

sig = sin(2*pi*(f+100*t)*t) + randn(fs)*0.001



f, t, R = spectrogram(sig, fs=fs, nperseg=512, noverlap=300, mode='complex')
print(np.shape(R))



plt.subplot(2, 1, 1)

plt.imshow(angle(R), cmap='hsv',

           aspect='auto', origin='lower', extent=[t[0], t[-1], 0, fs/2])

plt.xlabel('Time [s]')

plt.ylabel('Frequency [Hz]')

plt.title('Angle')



plt.subplot(2, 1, 2)

plt.imshow(unwrap(angle(R), axis=1), cmap='hsv',

           aspect='auto', origin='lower', extent=[t[0], t[-1], 0, fs/2])

plt.xlabel('Time [s]')

plt.ylabel('Frequency [Hz]')

plt.title('Time derivative of angle')
plt.show()