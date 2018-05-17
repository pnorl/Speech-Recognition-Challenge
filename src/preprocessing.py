import os
from glob import glob
import re
import scipy
from scipy.io import wavfile
from scipy.signal import resample as sp_resample

import numpy as np

######### Reading in data from file #########

#Returns lists of file names (fnames) and their corresponding labels (labels)
def list_wavs_fname(dirpath, ext='wav'):
    print("Fetching data from: "+dirpath)
    fpaths = glob(os.path.join(dirpath, r'*/*' + ext))
    pat = r'.+/(\w+)/\w+\.' + ext + '$' #./typeOfSound/filename.wav
    labels = []
    for fpath in fpaths:
        r = re.match(pat, fpath)
        if r:
            labels.append(r.group(1))
    
    pat = r'.+/(\w+\.' + ext + ')$' #./filename.wav
    fnames = []
    for fpath in fpaths:
        r = re.match(pat, fpath)
        if r:
            fnames.append(r.group(1))
    return labels, fnames

def read_data():
	# Data folders
	root_path = r'..'

	#Join paths
	train_data_path = os.path.join(root_path, 'data', 'train','audio')

	#Create representation of directory
	labels, fnames = list_wavs_fname(train_data_path)

	#Read wav files and put in arrays
	sample_rate_array,samples_array = [],[] 
	for label, fname in zip(labels, fnames):
		sample_rate, samples = wavfile.read(os.path.join(train_data_path, label, fname))
		sample_rate_array.append(sample_rate)
		samples_array.append(samples)

	return sample_rate_array, samples_array,labels

######### Preprocessing #########

#Pads audio with (zeros?) if signal is shorter than custom length
def pad_audio(samples,L=16000):
    if len(samples) >= L: return samples
    else: return np.pad(samples, pad_width=(L - len(samples), 0), mode='constant', constant_values=(0, 0))

#Cuts samples into n sequences if they're longer than L (16000 = 1s)
def chop_audio(samples, L=16000, n=20):
    for i in range(n):
        start = np.random.randint(0, len(samples) - L)
        yield samples[start: start + L]

#Resamples 
def resample(sample_rate,samples,L=16000,new_sample_rate=8000): 
    return sp_resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))

#This or mean normalization?
def pre_emphasis(signal,pre_emphasis = 0.97):
	return np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])


#Transforms label to its valid form (and one-hot-encodes it?)
def label_transform(labels):
    nlabels = []
    for label in labels:
        if label == '_background_noise_':
            nlabels.append('silence')
        elif label not in legal_labels:
            nlabels.append('unknown')
        else:
            nlabels.append(label)
    return pd.get_dummies(pd.Series(nlabels))

''' 
Pre-processing:
1. Choping/padding data
2. Resample data
3. Pre-emphasize data
4. Calculate spectrogram
5. Apppend to feature- and label arrays
'''

def preprocess():
	#Read data from file
	print("***Reading data from file***")
	sample_rates,signals,labels = read_data()

	x_train = [] #Array of spectrograms
	y_train = [] #Array of labels, later to be one-hot-encoded

	L = 16000 #Threshold of clips
	new_sample_rate = 8000 #Want to resample from L to new_sample_rate

	print("***Starts preprocessing***")
	#For each clip of sound 
	for sample_rate,samples,label in zip(sample_rates,signals,labels):
		samples = pad_audio(samples) #Custom function (pads samples < 16000)
		if(len(samples) > L):
			n_samples = chop_audio(samples) #Custom function (chops samples > 16000)
		else:
			n_samples = [samples] 
	    
	    #For each clip of sound (equals to one if clip not longer than 16000) 
		for samples in n_samples:
			samples = resample(sample_rate,samples) #Resample from 16000 to 8000
			samples = pre_emphasis(samples)


			#CALCULATE SPECTROGRAM
			#_, _, specgram = log_specgram(samples, sample_rate=new_sample_rate)

			#x_train-APPEND(spectrogram)
			#y_train-APPEND(label)
		print("klar")
		break


preprocess()