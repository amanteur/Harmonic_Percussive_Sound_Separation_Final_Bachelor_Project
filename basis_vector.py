import os
import numpy as np
import scipy
import librosa
import matplotlib.pyplot as plt
import librosa.display
from scipy.signal import stft, istft, spectrogram


def NMF(X,Kh=500,Kp=250,max_iter=100):

	Nf = X.shape[0]
	Nt = X.shape[1]
	W_esti = np.random.rand(Nf,Kh+Kp)
	H_esti = np.random.rand(Kh+Kp,Nt)

	# Flat initialization
	W_esti[:,Kh:Kh+Kp]=np.ones((Nf,Kp))

	for i in range(0,max_iter):

		H_esti = H_esti*np.matmul(W_esti.T,X/np.matmul(W_esti,H_esti))
		Denominator = np.matmul(np.sum(W_esti,axis=0)[:,None],np.ones((1,Nt)))
		H_esti = H_esti/Denominator

		# Dirichlet prior (generalized to NMF framework)

		if i<(max_iter-1):
			# Harmonic
			neighbors = np.concatenate((H_esti[0:Kh,1:Nt],H_esti[0:Kh,Nt-1:Nt]),axis=1)
			delta = H_esti[0:Kh,:]-neighbors
			delta = 0.7*delta
			H_esti[0:Kh,:] = neighbors+delta
			H_esti[0:Kh,-1] = neighbors[:,-1]

			# Percussive
			neighbors = np.concatenate((H_esti[Kh:Kh+Kp,1:Nt],H_esti[Kh:Kh+Kp,Nt-1:Nt]),axis=1)
			delta = H_esti[Kh:Kh+Kp,:]-neighbors
			delta = 1.05*delta
			H_esti[Kh:Kh+Kp,:] = np.maximum(neighbors+delta,0.00000001)
			H_esti[Kh:Kh+Kp,-1] = neighbors[:,-1]

		W_esti = W_esti*np.matmul(X/np.matmul(W_esti,H_esti),H_esti.T)
		Denominator2 = np.matmul(np.ones((Nf,1)),np.sum(H_esti,axis=1)[None,:])
		W_esti = W_esti/Denominator2

		# Dirichlet prior (generalized to NMF framework)
		if i<(max_iter-1):
			# Harmonic
			neighbors = np.concatenate((W_esti[1:Nf,0:Kh],W_esti[Nf-1:Nf,0:Kh]),axis=0)
			delta = W_esti[:,0:Kh]-neighbors
			delta = 1.05*delta
			W_esti[:,0:Kh] = np.maximum(neighbors+delta,0.00000001)
			W_esti[-1,0:Kh] = neighbors[-1,:]

			# Percussive
			neighbors = np.concatenate((W_esti[1:Nf,Kh:Kh+Kp],W_esti[Nf-1:Nf,Kh:Kh+Kp]),axis=0)
			delta = W_esti[:,Kh:Kh+Kp]-neighbors
			delta = 0.95*delta
			W_esti[:,Kh:Kh+Kp] = neighbors+delta
			W_esti[-1,Kh:Kh+Kp] = neighbors[-1,:]

	return (W_esti,H_esti)


def HPSS(x,sr=44100,wiener_filt=0):
	# (2049, 345)
	# Audio signals are assumed to be mono
	winlen = 1024
	# h, i, X = stft(x=x, fs=sr, window='hamming', nperseg=4096, noverlap=int(winlen / 2),
	# 			 return_onesided=True, padded=True)
	X = librosa.core.stft(x,n_fft=4096,hop_length=1024,window='hamming')

	X_abs = np.abs(X)+0.0000000001

	phase = X/X_abs

	Kh = 500
	Kp = 250
	max_iter = 100
	W_esti,H_esti = NMF(X_abs,Kh,Kp,max_iter)
	# print(np.shape(H_esti))
	# Post-processing
	X_harmonic = np.matmul(W_esti[:,0:Kh],H_esti[0:Kh,:])
	X_percussive = np.matmul(W_esti[:,Kh:Kh+Kp],H_esti[Kh:Kh+Kp,:])

	if (wiener_filt==1):
		X_harmonic2 = np.square(X_harmonic)/(np.square(X_harmonic)+np.square(X_percussive))*X_abs
		X_percussive = np.square(X_percussive)/(np.square(X_harmonic)+np.square(X_percussive))*X_abs
		X_harmonic = X_harmonic2

	harmonic = librosa.core.istft(X_harmonic*phase,hop_length=1024,window='hamming')
	percussive = librosa.core.istft(X_percussive*phase,hop_length=1024,window='hamming')
	# _, harmonic = istft(X_harmonic, fs=sr, window='hamming', nperseg=winlen,
	# 			 noverlap=int(winlen / 2), nfft=winlen, input_onesided=True)
	# _, percussive = istft(X_percussive, fs=sr, window='hamming', nperseg=winlen,
	# 			 noverlap=int(winlen / 2), nfft=winlen, input_onesided=True)

	return (harmonic,percussive)


def main(file_path):

	### Example
	x, fs = librosa.load(file_path, sr=None)
	print(fs)
	harmonic,percussive = HPSS(x,sr=fs,wiener_filt=1)

	print(len(harmonic))
	print(len(x))
	# Save
	librosa.output.write_wav(os.path.splitext(file_path)[0]+'_H_nmf.wav',harmonic,fs)
	librosa.output.write_wav(os.path.splitext(file_path)[0]+'_P_nmf.wav',percussive,fs)


	# Plotting
	#waves
	plt.figure(1)
	plt.subplot(2, 1, 1)
	t_scale = np.linspace(0, len(harmonic) / fs, len(harmonic))
	plt.plot(t_scale, harmonic)
	plt.title('Time domain visualization of h(t)')
	plt.axis('tight')
	plt.grid('on')
	plt.ylabel('Amplitude')
	plt.xlabel('Time (s)')

	plt.subplot(2, 1, 2)
	t_scale = np.linspace(0, len(percussive) / fs, len(percussive))
	plt.plot(t_scale, percussive)
	plt.title('Time domain visualization of p(t)')
	plt.axis('tight')
	plt.grid('on')
	plt.ylabel('Amplitude')
	plt.xlabel('Time (s)')
	plt.show()

	#spectrograms
	X = librosa.stft(x)
	H = librosa.stft(harmonic)
	P = librosa.stft(percussive)
	print(np.shape(H))
	print(np.shape(X))
	plt.figure()
	plt.subplot(3, 1, 1)
	Xmag = librosa.amplitude_to_db(X)
	librosa.display.specshow(Xmag, sr=fs, x_axis='time', y_axis='log')
	plt.colorbar()
	plt.title('Spectrogram of h(t)')
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')

	plt.subplot(3, 1, 2)
	Hmag = librosa.amplitude_to_db(H)
	librosa.display.specshow(Hmag, sr=fs, x_axis='time', y_axis='log')
	plt.colorbar()
	plt.title('Spectrogram of h(t)')
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')

	plt.subplot(3, 1, 3)
	Pmag = librosa.amplitude_to_db(P)
	librosa.display.specshow(Pmag, sr=fs, x_axis='time', y_axis='log')
	plt.colorbar()
	plt.title('Spectrogram of p(t)')
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')
	plt.show()

if __name__ == "__main__":
	print("Начало")

	main(file_path='samples_for_3_glava/original__mixed/1.wav')

	print("Конец")
	# main_toy()