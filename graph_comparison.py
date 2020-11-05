import mir_eval.display as dsp
import mir_eval.sonify as sn
from librosa import display
import librosa
import scipy
import matplotlib.pyplot as plt
import mass_ts as mts
import numpy as np
from scipy.spatial import distance
import matplotlib.axes as ax

def get_wav(file_path):
    x, sr = librosa.load(file_path, sr=None)
    return x

def get_chords(x):
    sn.chord(x)

if __name__ == '__main__':
    print("Начало")
    filepath = 'samples_for_3_glava/'
    alg_path = ('original','ono','median')
    type_path = ('_harmonic/','_percussive/')
    number_path = 1
    end_path = '.wav'
    sr = 44100
    file_path_true_h = filepath+alg_path[0]+type_path[0]+str(8)+end_path
    file_path_true_p = filepath+alg_path[0]+type_path[1]+str(8)+end_path
    file_path_est_md_h = filepath + alg_path[2] + type_path[0] + str(8) + end_path
    file_path_est_md_p = filepath + alg_path[2] + type_path[1] + str(8) + end_path
    file_path_est_ono_h = filepath + alg_path[1] + type_path[0] + str(8) + end_path
    file_path_est_ono_p = filepath + alg_path[1] + type_path[1] + str(8) + end_path
    h_true = get_wav(file_path_true_h)
    p_true = get_wav(file_path_true_p)
    h_md = get_wav(file_path_est_md_h)
    p_md = get_wav(file_path_est_md_p)
    h_ono = get_wav(file_path_est_ono_h)
    p_ono = get_wav(file_path_est_ono_p)
    # plt.figure(figsize=(12, 4))
    # dsp.separation([p_true, h_true], sr, labels=['percussive', 'harmonic'])
    # plt.legend()
    # plt.title('Original')
    # plt.show()
    plt.figure(figsize=(12, 4))
    plt.plot(h_ono)
    plt.legend()
    plt.title('Ono Harmonic')
    plt.show()
    plt.figure(figsize=(12, 4))
    plt.plot(h_md)
    plt.title('Median Harmoic')
    plt.show()

