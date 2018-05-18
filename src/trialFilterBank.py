#http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import signal as sig


def log_specgram(audio, sample_rate, window_size=25,
                 step_size=10, nfft=512, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = sig.spectrogram(audio,
    								nfft=nfft,
                                    fs=sample_rate,
                                    window='hamming',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


def preEmph(signal, pre_emphasis=0.97):
	#Pre-emphasis
	emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
	return emphasized_signal

def framing(signal, sample_rate, frame_size=0.025, frame_stride=0.01):
	#Framing

	frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
	signal_length = len(signal)
	frame_length = int(round(frame_length))
	frame_step = int(round(frame_step))
	num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

	pad_signal_length = num_frames * frame_step + frame_length
	z = np.zeros((pad_signal_length - signal_length))
	pad_signal = np.append(signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

	indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
	frames = pad_signal[indices.astype(np.int32, copy=False)]
	return frames, frame_length

def windowing(frames, frame_length):
	#Windowing
	frames *= np.hamming(frame_length)
	return frames

def fftPs(frames, NFFT=512):
	#FT and power spectrum
	NFFT = 512
	mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
	pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
	return pow_frames

def filterBank(pow_frames, sample_rate, nfilt=40, NFFT=512):
	#Filter banks
	low_freq_mel = 0
	high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
	mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
	hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
	bin = np.floor((NFFT + 1) * hz_points / sample_rate)

	fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
	for m in range(1, nfilt + 1):
	    f_m_minus = int(bin[m - 1])   # left
	    f_m = int(bin[m])             # center
	    f_m_plus = int(bin[m + 1])    # right

	    for k in range(f_m_minus, f_m):
	        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
	    for k in range(f_m, f_m_plus):
	        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
	filter_banks = np.dot(pow_frames, fbank.T)
	filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
	filter_banks = 20 * np.log10(filter_banks)  # dB
	return filter_banks

def meanNormalization(spec):
	#Mean normalization
	return spec - (np.mean(spec, axis=0) + 1e-8)

def cleanNaN(matrix):
	indexNaNs = np.isnan(matrix)
	matrix[indexNaNs] = np.nanmin(matrix)
	return matrix


def runplot(file):
	#MAIN
	sample_rate, signal = scipy.io.wavfile.read('../data/train/audio/' + file)
	print(file)
	print("Sample rate:", sample_rate)
	print("Signal:", signal.shape)

	#Running log_spec from lightweight. No other pre-processing
	freqs, times, logSpec = log_specgram(signal, sample_rate)

	#Running log_spec from lightweight. Pre-emphasized
	signalP = preEmph(signal)
	freqsP, timesP, logSpecP = log_specgram(signalP, sample_rate)

	#Running log_spec from lightweight. Pre-emphasized and mean normalized
	logSpecPM = meanNormalization(logSpecP)

	#Running custom spec. No other pre-processing
	frames, frameLength = framing(signal, sample_rate)
	frames = windowing(frames, frameLength)
	powFrames = fftPs(frames)
	custSpec = np.log(filterBank(powFrames, sample_rate) + 1e-10)

	#RUnning custom spec. Pre-emphasized
	framesP, frameLengthP = framing(signalP, sample_rate)
	framesP = windowing(framesP, frameLengthP)
	powFramesP = fftPs(framesP)
	custSpecP = np.log(filterBank(powFramesP, sample_rate) + 1e-10)

	#Running custom spec. Pre-emphasized and mean normalized
	#custSpecPM = meanNormalization(filterBank(powFramesP))
	custSpecPM = np.log(meanNormalization(filterBank(powFramesP, sample_rate)) + 1e-10)
	custSpecPM = cleanNaN(custSpecPM)




	#Plot emphasized wave
	fig = plt.figure(figsize=(14, 8))
	ax1 = fig.add_subplot(211)
	ax1.set_title('Raw wave')
	ax1.set_ylabel('Amplitude')
	ax1.plot(np.linspace(0, sample_rate/len(signal), sample_rate), signal)

	ax2 = fig.add_subplot(212)
	ax2.set_title('Emphasized wave')
	ax2.set_ylabel('Amplitude')
	ax2.plot(np.linspace(0, sample_rate/len(signalP), sample_rate), signalP)

	plt.plot()
	plt.show()

	#Print shapes
	print('logSpec:', logSpec.shape)
	print('custSpec:', custSpec.shape)

	#Plot spectrograms
	fig = plt.figure(figsize=(14, 8))
	ax1 = fig.add_subplot(321)
	ax1.set_title('LogSpec')
	ax1.set_ylabel('Hz')
	ax1.set_xlabel('Sec')
	im1 = ax1.imshow(logSpec.T, aspect='auto', origin='lower', cmap=cm.jet, 
	           extent=[times.min(), times.max(), freqs.min(), freqs.max()])
	fig.colorbar(im1)

	ax3 = fig.add_subplot(323)
	ax3.set_title('LogSpecP')
	ax3.set_ylabel('Hz')
	ax3.set_xlabel('Sec')
	im3 = ax3.imshow(logSpecP.T, aspect='auto', origin='lower', cmap=cm.jet, 
	           extent=[timesP.min(), timesP.max(), freqsP.min(), freqsP.max()])
	fig.colorbar(im3)

	ax5 = fig.add_subplot(325)
	ax5.set_title('LogSpecPM')
	ax5.set_ylabel('Hz')
	ax5.set_xlabel('Sec')
	im5 = ax5.imshow(logSpecPM.T, aspect='auto', origin='lower', cmap=cm.jet, 
	           extent=[timesP.min(), timesP.max(), freqsP.min(), freqsP.max()])
	fig.colorbar(im5)

	ax2 = fig.add_subplot(322)
	ax2.set_title('CustSpec')
	ax2.set_ylabel('Hz')
	ax2.set_xlabel('Sec')
	im2 = ax2.imshow(np.flipud(custSpec.T), cmap=cm.jet, aspect='auto', 
			   extent=[times.min(), times.max(), freqs.min(), freqs.max()])
	fig.colorbar(im2)

	ax4 = fig.add_subplot(324)
	ax4.set_title('CustSpecP')
	ax4.set_ylabel('Hz')
	ax4.set_xlabel('Sec')
	im4 = ax4.imshow(np.flipud(custSpecP.T), cmap=cm.jet, aspect='auto', 
			   extent=[timesP.min(), timesP.max(), freqsP.min(), freqsP.max()])
	fig.colorbar(im4)

	ax6 = fig.add_subplot(326)
	ax6.set_title('CustSpecPM')
	ax6.set_ylabel('Hz')
	ax6.set_xlabel('Sec')
	im6 = ax6.imshow(np.flipud(custSpecPM.T), cmap=cm.jet, aspect='auto', 
			   extent=[timesP.min(), timesP.max(), freqsP.min(), freqsP.max()])
	fig.colorbar(im6)

	plt.plot()
	plt.show()



files = ['happy/0ac15fe9_nohash_0.wav', 
		'bed/0c40e715_nohash_0.wav', 
		'dog/a60a09cf_nohash_0.wav', 
		'down/8056e897_nohash_0.wav', 
		'eight/37dca74f_nohash_0.wav']

files2 = ['marvin/1b4c9b89_nohash_0.wav',
		'marvin/1f3bece8_nohash_0.wav',
		'marvin/1f653d27_nohash_0.wav',
		'marvin/1fe4c891_nohash_0.wav',
		'marvin/20d3f11f_nohash_0.wav',
		'marvin/24ad3ebe_nohash_0.wav',
		'marvin/24ad3ebe_nohash_1.wav',
		'marvin/24ad3ebe_nohash_2.wav',
		'marvin/24ad3ebe_nohash_3.wav',
		'marvin/27c30960_nohash_0.wav',
		'marvin/2aa787cf_nohash_0.wav']
#for file in files2:
	#runplot(file) 
