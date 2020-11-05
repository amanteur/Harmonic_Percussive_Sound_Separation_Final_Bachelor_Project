import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import stft, istft, spectrogram
import librosa.display
from scipy.io.wavfile import read, write


def separate_instruments(file_path):
    # Считывание файла

    # fs, x = read("./audio/" + file_path)
    x, fs = librosa.load(file_path, sr=None)
    # f, t, Sxx = spectrogram(x, fs)
    X = librosa.stft(x)
    #  log-amplitude
    Xmag = librosa.amplitude_to_db(X)

    # показать изначальную спектрограмму
    librosa.display.specshow(Xmag, sr=fs, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.title('Spectrogram of x(t)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    winlen = 1024

    # Шаг 1. Создать оконное преобразование Фурье
    h, i, F = stft(x=x, fs=fs, window='hann', nperseg=winlen, noverlap=int(winlen / 2),
                   nfft=winlen, detrend=False, return_onesided=True, padded=True, axis=-1)

    # Шаг 2: Амплитудная спектрограмма
    gamma = 0.3
    W = np.power(np.abs(F), 2 * gamma)

    # Шаг 3: Инициализация
    k_max = 100
    H = 0.5 * W
    P = 0.5 * W
    alpha = 0.3

    for k in range(k_max):
        # Шаг 4. Рассчитать дельту
        term_1 = np.zeros_like(H)
        term_2 = np.zeros_like(H)

        for i_iter in range(1, np.shape(H)[1] - 1):
            term_1[:, i_iter] = alpha * ((H[:, i_iter - 1] + H[:, i_iter + 1] - (2 * H[:, i_iter])) / 4)

        term_1[:, 0] = alpha * ((H[:, 1] - H[:, 0]) / 2)
        term_1[:, -1] = alpha * ((H[:, -2] - H[:, -1]) / 2)

        for h_iter in range(1, np.shape(H)[0] - 1):
            term_2[h_iter, :] = (1 - alpha) * ((P[h_iter - 1, :] + P[h_iter + 1, :] - (2 * P[h_iter, :])) / 4)

        term_2[0, :] = (1 - alpha) * ((P[1, :] - P[0, :]) / 2)
        term_2[-1, :] = (1 - alpha) * ((P[-2, :] - P[-1, :]) / 2)

        delta = term_1 - term_2

        # уменьшить шаг
        delta = delta * 0.9

        # Шаг 5. Обновить матрицы компонент
        H = np.minimum(np.maximum(H + delta, 0), W)
        P = W - H
        # Шаг 6: Автоматически увеличивать к

    # Шаг 7: Разделить амлпитудную спектрограмму
    H = np.where(np.less(H, P), 0, W)
    P = np.where(np.greater_equal(H, P), 0, W)

    # Шаг 8. Обратное оконное преобразование Фурье
    H_temp = np.power(H, (1 / (2 * gamma))) * np.exp(1j * np.angle(F))  # ISTFT is taken first on this, with H
    P_temp = np.power(P, (1 / (2 * gamma))) * np.exp(1j * np.angle(F))  # ISTFT is taken second on this, with P
    _, h = istft(H_temp, fs=fs, window='hann', nperseg=winlen,
                 noverlap=int(winlen / 2), nfft=winlen, input_onesided=True)
    _, p = istft(P_temp, fs=fs, window='hann', nperseg=winlen,
                 noverlap=int(winlen / 2), nfft=winlen, input_onesided=True)
    #####################################################################################################

    plt.figure()
    plt.subplot(2, 1, 1)
    t_scale = np.linspace(0, len(h) / fs, len(h))
    plt.plot(t_scale, h)
    plt.title('Форма волны h(t)')
    plt.axis('tight')
    plt.grid('on')
    plt.ylabel('Амплитуда')
    plt.xlabel('Время (с)')

    plt.subplot(2, 1, 2)
    t_scale = np.linspace(0, len(p) / fs, len(p))
    plt.plot(t_scale, p)
    plt.title('Форма волны p(t)')
    plt.axis('tight')
    plt.grid('on')
    plt.ylabel('Амплитуда')
    plt.xlabel('Время (с)')
    plt.show()

    plt.figure()

    plt.subplot(2, 1, 1)
    H = librosa.stft(h)
    Hmag = librosa.amplitude_to_db(H)
    librosa.display.specshow(Hmag, sr=fs, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.subplot(2, 1, 2)
    P = librosa.stft(p)
    Pmag = librosa.amplitude_to_db(P)
    librosa.display.specshow(Pmag, sr=fs, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    # plt.figure(3)
    # ha, = plt.plot((h + p)[10000:10500], label='reconstructed')
    # hb, = plt.plot(x[10000:10500], 'k--', label='original')
    # plt.legend(handles=[ha, hb])
    # plt.title('Original and separated(10000:105000 samples) waveforms')
    # plt.show()
    #
    original_power = 10 * np.log10(np.sum(np.power(x, 2)))
    error = x - (h + p)[0:len(x)]
    noise_power = 10 * np.log10(np.sum(np.power(error, 2)))
    print("SNR is: %.2fdB" % (original_power - noise_power,))

    # saving
    # librosa.output.write_wav(os.path.splitext(file_path)[0] + '_H_cd.wav', h, fs)
    # librosa.output.write_wav(os.path.splitext(file_path)[0] + '_P_cd.wav', p, fs)


if __name__ == '__main__':
    print("Начало")

    separate_instruments(file_path = 'samples_for_3_glava/original__mixed/8.wav')

    print("Конец")