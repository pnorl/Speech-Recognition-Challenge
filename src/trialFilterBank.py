#http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import signal as sig


def log_specgram(audio, sampleRate, window_size=25,
                 step_size=10, nfft=512, eps=1e-10):
	#Old spec-maker from internet
	#Not used anymore
    nperseg = int(round(window_size * sampleRate / 1e3))
    noverlap = int(round(step_size * sampleRate / 1e3))
    freqs, times, spec = sig.spectrogram(audio,
    								nfft=nfft,
                                    fs=sampleRate,
                                    window='hamming',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


def preEmphasis(signal, emphasisLevel=0.97):
	#Pre-emphasis
	emphasizedSignal = np.append(signal[0], signal[1:] - emphasisLevel * signal[:-1])
	return emphasizedSignal

def framing(signal, sampleRate, frameSize=0.025, frameStride=0.01):
	#Framing

	frameLength, frameStep = frameSize * sampleRate, frameStride * sampleRate  # Convert from seconds to samples
	signalLength = len(signal)
	frameLength = int(round(frameLength))
	frameStep = int(round(frameStep))
	noFrames = int(np.ceil(float(np.abs(signalLength - frameLength)) / frameStep))  # Make sure that we have at least 1 frame

	padSignalLength = noFrames * frameStep + frameLength
	z = np.zeros((padSignalLength - signalLength))
	padSignal = np.append(signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

	indices = np.tile(np.arange(0, frameLength), (noFrames, 1)) + np.tile(np.arange(0, noFrames * frameStep, frameStep), (frameLength, 1)).T
	frames = padSignal[indices.astype(np.int32, copy=False)]
	return frames, frameLength

def windowing(frames, frameLength):
	#Windowing
	frames *= np.hamming(frameLength)
	return frames

def fftPs(frames, NFFT=512):
	#FT and power spectrum
	NFFT = 512
	magFrames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
	powFrames = ((1.0 / NFFT) * ((magFrames) ** 2))  # Power Spectrum
	return powFrames

def filterBank(powFrames, sampleRate, nfilt=40, NFFT=512):
	#Filter banks
	lowFreqMel = 0
	highFreqMel = (2595 * np.log10(1 + (sampleRate / 2) / 700))  # Convert Hz to Mel
	melPoints = np.linspace(lowFreqMel, highFreqMel, nfilt + 2)  # Equally spaced in Mel scale
	hzPoints = (700 * (10**(melPoints / 2595) - 1))  # Convert Mel to Hz
	bin = np.floor((NFFT + 1) * hzPoints / sampleRate)

	fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
	for m in range(1, nfilt + 1):
	    fmMinus = int(bin[m - 1])   # left
	    fm = int(bin[m])             # center
	    fmPlus = int(bin[m + 1])    # right

	    for k in range(fmMinus, fm):
	        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
	    for k in range(fm, fmPlus):
	        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
	filterBanks = np.dot(powFrames, fbank.T)
	filterBanks = np.where(filterBanks == 0, np.finfo(float).eps, filterBanks)  # Numerical Stability
	filterBanks = 20 * np.log10(filterBanks)  # dB
	return filterBanks

def meanNormalization(spec):
	#Mean normalization
	return spec - (np.mean(spec, axis=0) + 1e-8)

def cleanNaN(matrix):
	indexNaNs = np.isnan(matrix)
	matrix[indexNaNs] = np.nanmin(matrix)
	return matrix



###HERE ARE THE FUNCTIONS WE REACH FROM OUTSIDE
def filBank(signal, sampleRate, frameSize=0.025, frameStride=0.01, nfilt=40, NFFT=512):
	##Filter bank without any other preprocessing
	frames, frameLength = framing(signal, sampleRate, frameSize, frameStride)
	frames = windowing(frames, frameLength)
	powFrames = fftPs(frames, NFFT)
	spectrogram = filterBank(powFrames, sampleRate, nfilt, NFFT)
	spectrogram = np.log(spectrogram)
	return cleanNaN(spectrogram)

def filBankP(signal, sampleRate, emphasisLevel=0.97, frameSize=0.025, frameStride=0.01, nfilt=40, NFFT=512):
	##Filter bank with pre-emphasis
	signal = preEmphasis(signal, emphasisLevel)
	frames, frameLength = framing(signal, sampleRate, frameSize, frameStride)
	frames = windowing(frames, frameLength)
	powFrames = fftPs(frames, NFFT)
	spectrogram = filterBank(powFrames, sampleRate, nfilt, NFFT)
	spectrogram = np.log(spectrogram)
	return cleanNaN(spectrogram)

def filBankM(signal, sampleRate, frameSize=0.025, frameStride=0.01, nfilt=40, NFFT=512):
	##Filter bank with mean normalization
	frames, frameLength = framing(signal, sampleRate, frameSize, frameStride)
	frames = windowing(frames, frameLength)
	powFrames = fftPs(frames, NFFT)
	spectrogram = filterBank(powFrames, sampleRate, nfilt, NFFT)
	spectrogram = meanNormalization(spectrogram)
	spectrogram = np.log(spectrogram)
	return cleanNaN(spectrogram)

def filBankPM(signal, sampleRate, emphasisLevel=0.97, frameSize=0.025, frameStride=0.01, nfilt=40, NFFT=512):
	##Filter bank with pre-emphasis and mean normalization
	signal = preEmphasis(signal, emphasisLevel)
	frames, frameLength = framing(signal, sampleRate, frameSize, frameStride)
	frames = windowing(frames, frameLength)
	powFrames = fftPs(frames, NFFT)
	spectrogram = filterBank(powFrames, sampleRate, nfilt, NFFT)
	spectrogram = meanNormalization(spectrogram)
	spectrogram = np.log(spectrogram)
	return cleanNaN(spectrogram)




def runplot(file):
	#Used to visualize
	#Not used anymore

	sampleRate, signal = scipy.io.wavfile.read('../data/train/audio/' + file)
	print(file)
	print("Sample rate:", sampleRate)
	print("Signal:", signal.shape)

	#Running log_spec from lightweight. No other pre-processing
	freqs, times, logSpec = log_specgram(signal, sampleRate)

	#Running log_spec from lightweight. Pre-emphasized
	signalP = preEmphasis(signal)
	freqsP, timesP, logSpecP = log_specgram(signalP, sampleRate)

	#Running log_spec from lightweight. Pre-emphasized and mean normalized
	logSpecPM = meanNormalization(logSpecP)

	#Running custom spec. No other pre-processing
	frames, frameLength = framing(signal, sampleRate)
	frames = windowing(frames, frameLength)
	powFrames = fftPs(frames)
	custSpec = np.log(filterBank(powFrames, sampleRate) + 1e-10)

	#RUnning custom spec. Pre-emphasized
	framesP, frameLengthP = framing(signalP, sampleRate)
	framesP = windowing(framesP, frameLengthP)
	powFramesP = fftPs(framesP)
	custSpecP = np.log(filterBank(powFramesP, sampleRate) + 1e-10)

	#Running custom spec. Pre-emphasized and mean normalized
	#custSpecPM = meanNormalization(filterBank(powFramesP))
	custSpecPM = np.log(meanNormalization(filterBank(powFramesP, sampleRate)) + 1e-10)
	custSpecPM = cleanNaN(custSpecPM)




	#Plot emphasized wave
	fig = plt.figure(figsize=(14, 8))
	ax1 = fig.add_subplot(211)
	ax1.set_title('Raw wave')
	ax1.set_ylabel('Amplitude')
	ax1.plot(np.linspace(0, sampleRate/len(signal), sampleRate), signal)

	ax2 = fig.add_subplot(212)
	ax2.set_title('Emphasized wave')
	ax2.set_ylabel('Amplitude')
	ax2.plot(np.linspace(0, sampleRate/len(signalP), sampleRate), signalP)

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
