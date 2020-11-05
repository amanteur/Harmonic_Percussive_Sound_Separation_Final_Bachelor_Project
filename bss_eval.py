import mir_eval
import librosa
import scipy

def test(file_path_est, file_path_true):
    x_est, fs_est = librosa.load(file_path_est, sr=None)
    x, fs = librosa.load(file_path_true, sr=None)
    (sdr, sir, sar, perm)=mir_eval.separation.bss_eval_sources(x_est,x)
    print(sdr,sir,sar,perm)


if __name__ == '__main__':
    print("Начало")

    test(file_path_true='samples_for_3_glava/original_harmonic/7_H.wav',
         file_path_est='samples_for_3_glava/median_harmonic/7_H_med.wav')

    print("Конец")
    # 14.97499061
    # 17.76837021