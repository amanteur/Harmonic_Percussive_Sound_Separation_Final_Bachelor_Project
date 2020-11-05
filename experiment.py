import mir_eval
from librosa import display
import librosa
import scipy
import matplotlib.pyplot as plt
import mass_ts as mts
import numpy as np
from scipy.spatial import distance


def get_wavs(file_path_true, file_path_est):
    x_est, sr = librosa.load(file_path_est, sr=None)
    x_true, sr = librosa.load(file_path_true, sr=None)
    return x_true,x_est

def onset_eval(x_true,x_est,sr = 44100):
    est_onsets = librosa.onset.onset_detect(y=x_est, sr=sr, units='time')
    true_onsets = librosa.onset.onset_detect(y=x_true, sr=sr, units='time')
    # print(est_onsets)
    # print(true_onsets)
    dict = mir_eval.onset.evaluate(true_onsets, est_onsets)
    return dict['F-measure'],dict['Precision'],dict['Recall']


if __name__ == '__main__':
    print("Начало")
    filepath = 'samples_for_3_glava/'
    alg_path = ('original','original_mixed/','median')
    type_path = ('_harmonic/','_percussive/')
    number_path = 1
    end_path = '.wav'

    md_h_f = []
    md_h_pr = []
    md_h_rec = []
    md_p_f = []
    md_p_pr = []
    md_p_rec = []

    mix_h_f = []
    mix_h_pr = []
    mix_h_rec = []
    mix_p_f = []
    mix_p_pr = []
    mix_p_rec = []

    f,Prec,Rec = (0,0,0)

    for i in range(1,3):
        for j in range (0,2):
            for m in range (1,9):
                file_path_true = filepath + alg_path[0] + type_path[j] + str(m) + end_path
                if i == 1:
                    file_path_est = filepath + alg_path[i] + str(m) + end_path
                if i == 2:
                    file_path_est = filepath + alg_path[i] + type_path[j] + str(m) + end_path
                x_true, x_est = get_wavs(file_path_true = file_path_true,
                                                file_path_est = file_path_est)
                f,Prec,Rec = onset_eval(x_true,x_est)
                if alg_path[i]== 'original_mixed/':
                    if type_path[j] == '_harmonic/':
                        mix_h_f = np.append(mix_h_f,f)
                        mix_h_pr = np.append(mix_h_pr, Prec)
                        mix_h_rec = np.append(mix_h_rec, Rec)
                    if type_path[j] == '_percussive/':
                        mix_p_f = np.append(mix_p_f,f)
                        mix_p_pr = np.append(mix_p_pr, Prec)
                        mix_p_rec = np.append(mix_p_rec, Rec)
                if alg_path[i]== 'median':
                    if type_path[j] == '_harmonic/':
                        md_h_f = np.append(md_h_f,f)
                        md_h_pr = np.append(md_h_pr, Prec)
                        md_h_rec = np.append(md_h_rec, Rec)
                    if type_path[j] == '_percussive/':
                        md_p_f = np.append(md_p_f,f)
                        md_p_pr = np.append(md_p_pr, Prec)
                        md_p_rec = np.append(md_p_rec, Rec)

    data_f = [md_h_f,mix_h_f]
    data_pr = [md_h_pr,mix_h_pr]
    data_rec = [md_h_rec,mix_h_rec]

    # plotting f-measure, precision, recall
    my_xticks = ['Med Harm', 'Mix Harm']
    fig1, ax1 = plt.subplots()
    fig1.canvas.draw()
    ax1.set_xticklabels(my_xticks)
    ax1.set_ylim(0, 1)
    ax1.set_yticks(np.arange(0, 1.1, step=0.1))
    ax1.boxplot(data_f, showfliers=False)
    ax1.set_title('F-Measure')

    fig2, ax2  = plt.subplots()
    fig1.canvas.draw()
    ax2.set_ylim(0, 1)
    ax2.set_xticklabels(my_xticks)
    ax2.set_yticks(np.arange(0, 1.1, step=0.1))
    ax2.boxplot(data_pr, showfliers=False)
    ax2.set_title('Precision')

    fig3, ax3 = plt.subplots()
    fig1.canvas.draw()
    ax3.set_ylim(0, 1)
    ax3.set_xticklabels(my_xticks)
    ax3.set_yticks(np.arange(0, 1.1, step=0.1))
    ax3.boxplot(data_rec, showfliers = False)
    ax3.set_title('Recall')
    plt.show()

    print(np.mean(md_h_f))
    print(np.mean(mix_h_f))
    # print(np.mean(md_p_f))
    # print(np.mean(mix_p_f))

    print(np.mean(md_h_pr))
    print(np.mean(mix_h_pr))
    # print(np.mean(md_p_pr))
    # print(np.mean(mix_p_pr))

    print(np.mean(md_h_rec))
    print(np.mean(mix_h_rec))
    # print(np.mean(md_p_rec))
    # print(np.mean(mix_p_rec))

