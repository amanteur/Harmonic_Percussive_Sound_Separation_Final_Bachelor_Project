import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from scipy.ndimage import median_filter
from librosa import core
from librosa import util
from scipy.signal import stft, istft, spectrogram
import os


def separate_instruments(file_path):
    plt.rcParams['figure.figsize'] = (14, 5)

    x, sr = librosa.load(file_path, sr=None)
    winlen = 1024
    h, i, X = stft(x=x, fs=sr, window='hann', nperseg=winlen, noverlap=int(winlen / 2),
                   nfft=winlen, detrend=False, return_onesided=True, padded=True, axis=-1)
    # information about wav
    print(len(x))
    # short-time fourier transform
    # X = librosa.stft(x)
    #  log-amplitude
    Xmag = librosa.amplitude_to_db(X)

    # show harm-perc spectrogram
    librosa.display.specshow(Xmag, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.show()
    #############
    S = X
    kernel_size = 31
    power = 2.0
    mask = False
    margin = 1.0

    if np.iscomplexobj(S):
        S, phase = core.magphase(S)
    else:
        phase = 1

    if np.isscalar(kernel_size):
        win_harm = kernel_size
        win_perc = kernel_size
    else:
        win_harm = kernel_size[0]
        win_perc = kernel_size[1]

    if np.isscalar(margin):
        margin_harm = margin
        margin_perc = margin
    else:
        margin_harm = margin[0]
        margin_perc = margin[1]

    split_zeros = (margin_harm == 1 and margin_perc == 1)
    # Compute median filters. Pre-allocation here preserves memory layout.
    harm = np.empty_like(S)
    harm[:] = median_filter(S, size=(1, win_harm), mode='reflect')

    perc = np.empty_like(S)
    perc[:] = median_filter(S, size=(win_perc, 1), mode='reflect')

    Hmag = librosa.amplitude_to_db(harm)
    # librosa.display.specshow(harm, sr=sr, x_axis='time', y_axis='log')
    # plt.colorbar()
    # plt.show()

    Pmag = librosa.amplitude_to_db(perc)
    # librosa.display.specshow(perc, sr=sr, x_axis='time', y_axis='log')
    # plt.colorbar()
    # plt.show()

    mask_harm_soft = util.softmask(harm, perc * margin_harm,
                              power=power,
                              split_zeros=split_zeros)
    mask_perc_soft = util.softmask(perc, harm * margin_perc,
                              power=power,
                              split_zeros=split_zeros)
    soft_mask_X_harm = (S * mask_harm_soft) * phase
    Xmag_harm_soft = librosa.amplitude_to_db(soft_mask_X_harm)
    soft_mask_X_perc = (S * mask_perc_soft) * phase
    Xmag_perc_soft = librosa.amplitude_to_db(soft_mask_X_perc)

    # mask_harm_hard = harm > perc * margin_harm
    # mask_perc_hard = perc > harm * margin_perc
    # hard_mask_X_harm = (S * mask_harm_hard) * phase
    # Xmag_harm_hard = librosa.amplitude_to_db(hard_mask_X_harm)
    # hard_mask_X_perc = (S * mask_perc_hard) * phase
    # Xmag_perc_hard = librosa.amplitude_to_db(hard_mask_X_perc)


    librosa.display.specshow(Xmag_harm_soft, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.show()

    librosa.display.specshow(Xmag_perc_soft, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.show()


    # x_h, sr_h = librosa.load('my_audio_mod/01_AF_NM_h.wav', duration=6, sr=None)
    # x_p, sr_p = librosa.load('my_audio_mod/01_AF_NM_p.wav', duration=6, sr=None)
    # librosa.display.waveplot( x_h, sr=sr_h)
    # plt.show()
    # librosa.display.waveplot( x_p, sr=sr_p)
    # plt.show()
    H = (S * mask_harm_soft) * phase
    P = (S * mask_perc_soft) * phase

    Hmag = librosa.amplitude_to_db(H)
    Pmag = librosa.amplitude_to_db(P)

    librosa.display.specshow(Hmag, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.show()

    librosa.display.specshow(Pmag, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.show()

    # h = librosa.istft(H)
    # p = librosa.istft(P)
    _, h = istft(H, fs=sr, window='hann', nperseg=winlen,
                 noverlap=int(winlen / 2), nfft=winlen, input_onesided=True)
    _, p = istft(P, fs=sr, window='hann', nperseg=winlen,
                 noverlap=int(winlen / 2), nfft=winlen, input_onesided=True)

    # saving
    librosa.output.write_wav(os.path.splitext(file_path)[0] + '_H_med.wav', h, sr)
    librosa.output.write_wav(os.path.splitext(file_path)[0] + '_P_med.wav', p, sr)

if __name__ == '__main__':
    print("Начало")

    separate_instruments(file_path = 'ideal_and_not/ock-rack-tom-2.wav')

    print("Конец")

