import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from scipy.signal import spectrogram,stft
from librosa import core
from librosa import util
#(129,1181) (1025,517) shapes
def main():
    # step 1. Preprocessing and generating STFT

    # plot size
    plt.rcParams['figure.figsize'] = (14, 5)

    # getting wav sr - sampling frequency
    x, sr = librosa.load('audio\indie_rock_AF_NM.wav', duration=6, sr=None)

    # short-time fourier transform
    X = librosa.stft(x)
    #  log-amplitude
    Xmag = librosa.amplitude_to_db(X)
    # show harm-perc spectrogram
    # librosa.display.specshow(Xmag, sr=sr, x_axis='time', y_axis='log')
    # plt.colorbar()
    # plt.show()

    # step 2. Calculate module of stft and spectrum

    # f, t, R = _spectral_helper(x,x, fs=sr, nperseg=512, noverlap=300, mode='stft',return_onesided = False)
    f,t,R = spectrogram(x,fs= sr, mode= 'complex',nfft=2048,window=('hann'),nperseg=583)
    plt.imshow(1/(2*np.pi)*np.diff(np.unwrap(np.angle(R))), cmap='hsv' #'hsv' for phase
              ,aspect='auto', origin='lower', extent=[t[0], t[-1], 0, sr / 2])
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Unwrapped phase spectrogram')
    plt.tight_layout()
    plt.show()
    #phase spectrum
    unwr_ph_spec = np.unwrap(np.angle(R))
    dif_ph_spec = np.diff(unwr_ph_spec)
    #magnitude_spectrum
    mag_spec = abs(R)

    # step 3. Calculate delta k low and delta k high

    #frame length, hop size, sampling freq
    wl = 2048
    hs = 2048/4
    fs = 1000*sr/93
    print(sr)
    rows = X.shape[0]
    cols = X.shape[1]
    # computing centrals of k-bin
    f_tuple = librosa.feature.spectral_centroid(y=x, sr = sr)
    fk = f_tuple[0]
    plt.plot(fk)

    # initiating frequencies and deltas
    f_k_low = np.empty_like(fk)
    f_k_high = np.empty_like(fk)
    delta_k_low = np.empty_like(fk)
    delta_k_high = np.empty_like(fk)
    #compputing deltas
    for col in range(1, np.shape(X)[1]):
        # print(X[col,:])
        # computing k-frequencies low and high
        f_k_low[col] = fk[col] - fs / (2 * wl)
        f_k_high[col] = fk[col] + fs / (2 * wl)
        # deltas
        delta_k_low[col] = 2 * np.pi * f_k_low[col] * hs / fs
        delta_k_high[col] = 2 * np.pi * f_k_high[col] * hs / fs
    plt.plot(f_k_low)
    plt.plot(f_k_high)
    plt.show()
    # plt.semilogy(fk,label = 'centroid')

    # step 4. peak detection

    # computing power spectrogram
    W = np.power(np.abs(X),2)
    Wmag = librosa.amplitude_to_db(W)
    # librosa.display.specshow(Wmag, sr=sr, x_axis='time', y_axis='log')
    # plt.colorbar()
    # plt.show()
    onset_env = librosa.onset.onset_strength(y=x, sr=sr,hop_length=512,aggregate=np.median)
    peaks = librosa.util.peak_pick(onset_env, 3, 3, 3, 5, 0.5, 10)
    print(np.shape(X))
    print(np.shape(unwr_ph_spec))
    # step 5. creating spectral mask

    # create mask
    M = np.empty_like(X)
    for peak in peaks:
        for col in range(1, np.shape(M)[1]):
            if col == peak:
                M[:,col] = 1

    # step 6. masked phase

    masked_phase = np.multiply(unwr_ph_spec,M)
    librosa.display.specshow(masked_phase, sr=sr, x_axis='time', y_axis='linear')
    plt.show()
    print(np.shape(masked_phase))

    # step 7. computing derivative

    IFD = np.zeros(shape = (1025,516))
    for row in range(1, np.shape(X)[0]):
        IFD[row,:] = np.diff(masked_phase[row,:])*1/(2*np.pi)
    IFD_new = np.zeros((1025,517))
    IFD_new[:, :-1] = IFD
    print(IFD_new)
    librosa.display.specshow(IFD_new, sr=sr, x_axis='time', y_axis='linear')
    plt.colorbar()
    plt.show()

    # step 8. computing another mask
    print(np.shape(IFD))
    H_mask = np.empty_like(X)
    for col in range(1, np.shape(dif_ph_spec)[1]):
        for row in range(1, np.shape(dif_ph_spec)[0]):
            if dif_ph_spec[row,col] > delta_k_low[col] and dif_ph_spec[row,col] < delta_k_high[col]:
                H_mask[row,col] = 1
    P_mask = 1 - H_mask
    H_mask_mag = librosa.amplitude_to_db(H_mask)
    librosa.display.specshow(H_mask_mag, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.show()
    P_mask_mag = librosa.amplitude_to_db(P_mask)
    librosa.display.specshow(P_mask_mag, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.show()


main()