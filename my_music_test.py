import matplotlib.pyplot as plt
import librosa.display
import os

plt.rcParams['figure.figsize'] = (14, 5)
filename = 'samples_for_3_glava/original__mixed/1.wav'
x, sr = librosa.core.load(filename,sr=None)

# information about wav
print(len(x))
print(sr)

# short-time fourier transform
X = librosa.stft(x)
#  log-amplitude
Xmag = librosa.amplitude_to_db(X)

# show harm-perc spectrogram
librosa.display.specshow(Xmag, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
plt.show()

# harmonic percussion source separation
H, P = librosa.decompose.hpss(X,mask = True)
Hmag = librosa.amplitude_to_db(H)
Pmag = librosa.amplitude_to_db(P)

# show harmonic part spectrogram
librosa.display.specshow(Hmag, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
plt.show()

# show percussion part spectrogram
librosa.display.specshow(Pmag, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
plt.show()

# istft
h = librosa.istft(H)
p = librosa.istft(P)

# saving
librosa.output.write_wav(os.path.splitext(filename)[0]+'_H_med.wav',h,sr)
librosa.output.write_wav(os.path.splitext(filename)[0]+'_P_med.wav',p,sr)

