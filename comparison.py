import mir_eval
from librosa import display
import librosa
import scipy
import matplotlib.pyplot as plt
import mass_ts as mts
import numpy as np
from scipy.spatial import distance
import matplotlib.axes as ax


def get_wavs(file_path_true, file_path_est):
    x_est, sr = librosa.load(file_path_est, sr=None)
    x_true, sr = librosa.load(file_path_true, sr=None)
    return x_true,x_est

# def simple(A,B,m):
#     length = len(A)-m+1
#     P = np.ones(length)*np.inf
#     I = np.zeros(length)
#     for idx in range(1,length):
#         D = mts.mass2(B,A[idx:idx+m-1])
#         P[idx] = D.min()
#         # I[idx] = np.where(D == D.min())
#     print('done')
#     return P

def comparison(x_true,x_est,sr=44100):
    hop_length = 1024
    mfcc_est = librosa.feature.mfcc(y=x_est, sr=sr, hop_length=hop_length)
    mfcc_true = librosa.feature.mfcc(y=x_true, sr=sr, hop_length=hop_length)
    xsim = librosa.segment.cross_similarity(mfcc_est, mfcc_true)
    xsim_cosine = librosa.segment.cross_similarity(mfcc_est, mfcc_true, metric='cosine')
    xsim_aff = librosa.segment.cross_similarity(mfcc_est, mfcc_true, mode='affinity')
    # P = simple(A=x_est, B= x_true,m=100)
    # corr = scipy.signal.correlate(x_true,x_est)

    norm = np.linalg.norm(x_true-x_est)
    mse = (np.square(x_true - x_est)).mean()
    corr = distance.cdist(abs(librosa.stft(x_true)), abs(librosa.stft(x_est)), 'chebyshev')
    time_static = np.max(x_true*x_est)
    print(time_static)
    print(mse)
    print(norm)
    # print(distances)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    librosa.display.specshow(xsim_aff,cmap='magma_r', x_axis='time', y_axis='time', hop_length=hop_length)
    plt.title('Binary recurrence (symmetric)')
    plt.subplot(1, 2, 2)
    plt.plot(corr)
    plt.title('Cross-correlated')
    plt.tight_layout()
    plt.show()

def bss_eval(x_true,x_est):
    (sdr, isr, sir, sar, perm) = mir_eval.separation.bss_eval_images(x_true,x_est)
    original_power = 10 * np.log10(np.sum(np.power(x_true, 2)))
    error = x_true - x_est[0:len(x_true)]
    noise_power = 10 * np.log10(np.sum(np.power(error, 2)))
    snr = original_power - noise_power

    return sdr,isr,snr,sar

def norm_euc_and_mse(x_true,x_est):
    norm = np.linalg.norm(x_true - x_est)
    rmse = np.sqrt((np.square(x_true - x_est)).mean())

    return norm,rmse

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
    alg_path = ('original','ono','median')
    type_path = ('_harmonic/','_percussive/')
    number_path = 1
    end_path = '.wav'

    ono_h_sdr = []
    ono_h_isr = []
    ono_h_sar = []
    ono_h_snr = []
    ono_p_sdr = []
    ono_p_isr = []
    ono_p_sar = []
    ono_p_snr = []

    md_h_sdr = []
    md_h_isr = []
    md_h_sar = []
    md_h_snr = []
    md_p_sdr = []
    md_p_isr = []
    md_p_sar = []
    md_p_snr = []

    md_h_norm = []
    md_p_norm = []
    ono_h_norm = []
    ono_p_norm = []

    md_h_rmse = []
    md_p_rmse = []
    ono_h_rmse = []
    ono_p_rmse = []

    md_h_f = []
    md_h_pr = []
    md_h_rec = []
    md_p_f = []
    md_p_pr = []
    md_p_rec = []

    ono_h_f = []
    ono_h_pr = []
    ono_h_rec = []
    ono_p_f = []
    ono_p_pr = []
    ono_p_rec = []

    sdr,isr,snr,sar = (0,0,0,0)
    rmse,norm = (0,0)
    f,Prec,Rec = (0,0,0)

    for i in range(1,3):
        for j in range (0,2):
            for m in range (1,9):
                file_path_true = filepath+alg_path[0]+type_path[j]+str(m)+end_path
                file_path_est = filepath + alg_path[i] + type_path[j] + str(m) + end_path
                x_true, x_est = get_wavs(file_path_true = file_path_true,
                                         file_path_est = file_path_est)
                # comparison(x_true,x_est)
                # sdr,isr,snr,sar = bss_eval(x_true, x_est)
                norm,rmse = norm_euc_and_mse(x_true,x_est)
                # f,Prec,Rec = onset_eval(x_true,x_est)
                if alg_path[i]== 'ono':
                    if type_path[j] == '_harmonic/':
                        ono_h_sdr = np.append(ono_h_sdr,sdr)
                        ono_h_isr = np.append(ono_h_isr, isr)
                        ono_h_sar = np.append(ono_h_sar, sar)
                        ono_h_snr = np.append(ono_h_snr,snr)
                        ono_h_norm = np.append(ono_h_norm, norm)
                        ono_h_rmse = np.append(ono_h_rmse, rmse)
                        ono_h_f = np.append(ono_h_f,f)
                        ono_h_pr = np.append(ono_h_pr, Prec)
                        ono_h_rec = np.append(ono_h_rec, Rec)
                    if type_path[j] == '_percussive/':
                        ono_p_sdr = np.append(ono_p_sdr,sdr)
                        ono_p_isr = np.append(ono_p_isr,isr)
                        ono_p_sar = np.append(ono_p_sar,sar)
                        ono_p_snr = np.append(ono_p_snr, snr)
                        ono_p_norm = np.append(ono_p_norm, norm)
                        ono_p_rmse = np.append(ono_p_rmse, rmse)
                        ono_p_f = np.append(ono_p_f,f)
                        ono_p_pr = np.append(ono_p_pr, Prec)
                        ono_p_rec = np.append(ono_p_rec, Rec)
                if alg_path[i]== 'median':
                    if type_path[j] == '_harmonic/':
                        md_h_sdr = np.append(md_h_sdr,sdr)
                        md_h_isr = np.append(md_h_isr, isr)
                        md_h_sar = np.append(md_h_sar, sar)
                        md_h_snr = np.append(md_h_snr, snr)
                        md_h_norm = np.append(md_h_norm, norm)
                        md_h_rmse = np.append(md_h_rmse, rmse)
                        md_h_f = np.append(md_h_f,f)
                        md_h_pr = np.append(md_h_pr, Prec)
                        md_h_rec = np.append(md_h_rec, Rec)
                    if type_path[j] == '_percussive/':
                        md_p_sdr = np.append(md_p_sdr,sdr)
                        md_p_isr = np.append(md_p_isr,isr)
                        md_p_sar = np.append(md_p_sar,sar)
                        md_p_snr = np.append(md_p_snr, snr)
                        md_p_norm = np.append(md_p_norm, norm)
                        md_p_rmse = np.append(md_p_rmse, rmse)
                        md_p_f = np.append(md_p_f,f)
                        md_p_pr = np.append(md_p_pr, Prec)
                        md_p_rec = np.append(md_p_rec, Rec)
    data_md_h = [ md_h_sdr, md_h_isr, md_h_sar,md_h_snr]
    data_md_p = [ md_p_sdr, md_p_isr, md_p_sar,md_p_snr]
    data_ono_h = [ ono_h_sdr, ono_h_isr, ono_h_sar,ono_h_snr]
    data_ono_p = [ ono_p_sdr, ono_p_isr, ono_p_sar,ono_p_snr]

    data_h_norm = [md_h_norm,ono_h_norm]
    data_p_norm = [md_p_norm,ono_p_norm]
    data_h_rmse = [md_h_rmse,ono_h_rmse]
    data_p_rmse = [md_p_rmse,ono_p_rmse]

    data_f = [md_h_f,ono_h_f,md_p_f,ono_p_f]
    data_pr = [md_h_pr,ono_h_pr,md_p_pr,ono_p_pr]
    data_rec = [md_h_rec,ono_h_rec,md_p_rec,ono_p_rec]

    print(np.mean(md_h_norm))
    print(np.mean(ono_h_norm))
    print(np.mean(md_p_norm))
    print(np.mean(ono_p_norm))

    print(np.mean(md_h_rmse))
    print(np.mean(ono_h_rmse))
    print(np.mean(md_p_rmse))
    print(np.mean(ono_p_rmse))

    #plotting bss_eval
    # my_xticks = ['SDR', 'ISR', 'SAR', 'SNR']
    # fig1, (ax1,ax2) = plt.subplots(1, 2)
    # fig1.canvas.draw()
    # ax1.set_xticklabels(my_xticks)
    # ax1.set_ylim(-10,16)
    # ax1.set_yticks(np.arange(-10,18,step = 2))
    # ax1.boxplot(data_md_h,showfliers=False)
    # ax1.set_title('Median Harmonic')
    # ax1.set_xlabel('dB')
    # ax2.set_ylim(-10,16)
    # ax2.set_xticklabels(my_xticks)
    # ax2.set_yticks(np.arange(-10,18,step = 2))
    # ax2.boxplot(data_ono_h,showfliers=False)
    # ax2.set_title('Ono Harmonic')
    # ax2.set_xlabel('dB')
    # plt.show()
    #
    # fig2, (ax3, ax4) = plt.subplots(1, 2)
    # fig2.canvas.draw()
    # ax3.set_xticklabels(my_xticks)
    # ax3.set_ylim(-8, 10)
    # ax3.set_yticks(np.arange(-8, 10, step=2))
    # ax3.boxplot(data_md_p, showfliers=False)
    # ax3.set_title('Median Percussive')
    # ax3.set_xlabel('dB')
    # ax4.set_ylim(-8, 10)
    # ax4.set_xticklabels(my_xticks)
    # ax4.set_yticks(np.arange(-8, 10, step=2))
    # ax4.boxplot(data_ono_p, showfliers=False)
    # ax4.set_title('Ono Percussive')
    # ax4.set_xlabel('dB')
    # plt.show()

    # plotting norm and rmse
    # my_xticks = ['Median', 'Ono']
    # fig1, (ax1, ax2) = plt.subplots(1, 2)
    # fig1.canvas.draw()
    # ax1.set_xticklabels(my_xticks)
    # ax1.set_ylim(0, 100)
    # ax1.set_yticks(np.arange(0, 100, step=10))
    # ax1.boxplot(data_h_norm, showfliers=False)
    # ax1.set_title('Harmonic')
    # ax2.set_ylim(0, 100)
    # ax2.set_xticklabels(my_xticks)
    # ax2.set_yticks(np.arange(0, 100, step=10))
    # ax2.boxplot(data_p_norm, showfliers=False)
    # ax2.set_title('Percussive')
    # plt.show()
    #
    # my_xticks = ['Median', 'Ono']
    # fig1, (ax1, ax2) = plt.subplots(1, 2)
    # fig1.canvas.draw()
    # ax1.set_xticklabels(my_xticks)
    # ax1.set_ylim(0, 0.1)
    # ax1.set_yticks(np.arange(0, 0.1, step=0.01))
    # ax1.boxplot(data_h_rmse, showfliers=False)
    # ax1.set_title('Harmonic')
    # ax2.set_ylim(0, 0.25)
    # ax2.set_xticklabels(my_xticks)
    # ax2.set_yticks(np.arange(0, 0.25, step=0.01))
    # ax2.boxplot(data_p_rmse, showfliers=False)
    # ax2.set_title('Percussive')
    # plt.show()

    # plotting f-measure, precision, recall
    # my_xticks = ['Med Harm', 'Ono Harm','Med Perc', 'Ono Perc']
    # fig1, ax1 = plt.subplots()
    # fig1.canvas.draw()
    # ax1.set_xticklabels(my_xticks)
    # ax1.set_ylim(0, 1)
    # ax1.set_yticks(np.arange(0, 1.1, step=0.1))
    # ax1.boxplot(data_f, showfliers=False)
    # ax1.set_title('F-Measure')
    #
    # fig2, ax2  = plt.subplots()
    # fig1.canvas.draw()
    # ax2.set_ylim(0, 1)
    # ax2.set_xticklabels(my_xticks)
    # ax2.set_yticks(np.arange(0, 1.1, step=0.1))
    # ax2.boxplot(data_pr, showfliers=False)
    # ax2.set_title('Precision')
    #
    # fig3, ax3 = plt.subplots()
    # fig1.canvas.draw()
    # ax3.set_ylim(0, 1)
    # ax3.set_xticklabels(my_xticks)
    # ax3.set_yticks(np.arange(0, 1.1, step=0.1))
    # ax3.boxplot(data_rec, showfliers = False)
    # ax3.set_title('Recall')
    # plt.show()


    print("Конец")