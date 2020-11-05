import matplotlib.pyplot as plt
import librosa.display
import scipy


# plt.rcParams['figure.figsize'] = (14, 5)

xh, sr_h = librosa.load('samples_for_3_glava/original_harmonic/1_H.wav', sr=44100)
xp, sr_p = librosa.load('samples_for_3_glava/original_percussive/1_P.wav', sr=44100)
winlen = 1024

# Шаг 1. Создать оконное преобразование Фурье
h, i, H = scipy.signal.stft(x=xh, fs=sr_h, window='hann', nperseg=winlen, noverlap=int(winlen / 2),
               nfft=winlen, detrend=False, return_onesided=True, padded=True, axis=-1)
h, i, P = scipy.signal.stft(x=xp, fs=sr_p, window='hann', nperseg=winlen, noverlap=int(winlen / 2),
               nfft=winlen, detrend=False, return_onesided=True, padded=True, axis=-1)
print(len(xh))
_, p = scipy.signal.istft(P, fs=sr_p, window='hann', nperseg=winlen,
             noverlap=int(winlen / 2), nfft=winlen, input_onesided=True)
_, h = scipy.signal.istft(H, fs=sr_h, window='hann', nperseg=winlen,
             noverlap=int(winlen / 2), nfft=winlen, input_onesided=True)
print(len(h))
# information about wavs
# print(xh.shape)
# print(len(xh), len(xp))
# print(sr_h, sr_p)

# mixing wavs
# x = xh/xh.max() + xp/xp.max()
# x = 0.5 * x/x.max()
# librosa.output.write_wav('samples_for_3_glava/original__mixed/8.wav', x, sr_h)
librosa.output.write_wav('samples_for_3_glava/original_harmonic/1_H.wav', h, sr_h)
librosa.output.write_wav('samples_for_3_glava/original_percussive/1_P.wav', p, sr_p)
print('reducing_done')
# short-time fourier transform
# X = librosa.stft(x)
#  log-amplitude
# Xmag = librosa.amplitude_to_db(X)
#
# # show mix harm-perc spectrogram
# librosa.display.specshow(Xmag, sr=sr_h, x_axis='time', y_axis='log')
# plt.colorbar()
# plt.show()
#
# # harmonic percussion source separation
# H, P = librosa.decompose.hpss(X)
# Hmag = librosa.amplitude_to_db(H)
# Pmag = librosa.amplitude_to_db(P)
#
# # show harmonic part spectrogram
# librosa.display.specshow(Hmag, sr=sr_h, x_axis='time', y_axis='log')
# plt.colorbar()
# plt.show()
#
# # show percussion part spectrogram
# librosa.display.specshow(Pmag, sr=sr_p, x_axis='time', y_axis='log')
# plt.colorbar()
# plt.show()

# # save wavs
# h = librosa.istft(H)
# librosa.output.write_wav('modified audio/prelude_cmaj_mod.wav', h, sr_h)
#
# p = librosa.istft(P)
# librosa.output.write_wav('modified audio/bounce_mod.wav', p, sr_p)








