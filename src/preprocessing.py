import os
from glob import glob
import re
import scipy
from scipy.io import wavfile
from scipy.signal import resample as sp_resample
from trialFilterBank import *
import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np

L = 16000
######### Reading in data from file #########

class SoundClip(object):
	def __init__(self, signal, sample_rate, label, fname):
		self.signal = signal
		self.sample_rate = sample_rate
		self.label = label
		self.fname = fname

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

	#Join paths to get audio folder
	train_data_path = os.path.join(root_path, 'data', 'train','audio')

	#Create representation of directory
	labels, fnames = list_wavs_fname(train_data_path)

	#Create 3 sets of examples for test train and val
	val_examples, test_examples, train_examples = [], [], []

	# Read the validation and testing files
	with open(os.path.join(root_path, 'data', 'train', 'validation_list.txt'), 'r') as f:
		val_fnames = f.readlines()
	with open(os.path.join(root_path, 'data', 'train', 'testing_list.txt'), 'r') as f:
		test_fnames = f.readlines()

	for label, fname in zip(labels, fnames):
		try:
			sample_rate, signal = wavfile.read(os.path.join(train_data_path, label, fname))
			clip = SoundClip(signal, sample_rate, label, fname)
			clipped_name = os.path.join(label, fname)
			if (clipped_name in val_fnames):
				val_examples.append(clip)
			elif (clipped_name in test_fnames):
				test_examples.append(clip)
			else:
				train_examples.append(clip)
		except:
			print("Failed to read wavfile", os.path.join(train_data_path, label, fname))

	return train_examples, test_examples, val_examples

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
	legal_labels = 'yes no up down left right on off stop go silence unknown'.split()
	nlabels = []
	for label in labels:
		if label == '_background_noise_':
			nlabels.append('silence')
		elif label not in legal_labels:
			nlabels.append('unknown')
		else:
			nlabels.append(label)
	return pd.get_dummies(pd.Series(nlabels))

def preprocess_one_clip(clip):
	# Zero pads clip where L < 16000
	signal = pad_audio(clip.signal) # Zero pads clip where L < 16000

	# Chop audio to chunks if L > 16000
	# This is valid for silence sound
	if(len(signal) > L):
		n_samples = chop_audio(signal) #Custom function (chops samples > 16000)
	else:
		n_samples = [signal]

	x = []
	y = []
	#For each clip of sound (equals to one if clip not longer than 16000)
	for samples in n_samples:
		samples = preEmph(samples)
		frames, framesLength = framing(samples, L)
		frames = windowing(frames, framesLength)
		powFrames = fftPs(frames)
		specgram = filterBank(powFrames, L)
		specgram = np.log(meanNormalization(specgram) + 1e-10)
		specgram = cleanNaN(specgram)
		if np.isnan(np.nanmin(specgram)):
			continue

		x.append(specgram)
		y.append(clip.label)

	return x,y

def transform_arrays(x, y):
	x = np.array(x)
	x = x.reshape(tuple(list(x.shape) + [1]))
	y = label_transform(y)
	label_index = y.columns.values
	y = y.values
	y = np.array(y)
	return x, y

def preprocess():
	'''
	Pre-processing:
	1. Choping/padding data
	2. Resample data
	3. Pre-emphasize data
	4. Calculate spectrogram
	5. Apppend to feature- and label arrays
	'''
	train_examples, test_examples, val_examples = read_data()
	x_train, y_train, x_test, y_test, x_valid, y_valid = [],[],[],[],[],[]
	for clip in train_examples:
		x,y = preprocess_one_clip(clip)
		x_train.extend(x)
		y_train.extend(y)

	for clip in test_examples:
		x,y = preprocess_one_clip(clip)
		x_test.extend(x)
		y_test.extend(y)

	for clip in val_examples:
		x,y = preprocess_one_clip(clip)
		x_val.extend(x)
		y_val.extend(y)

	x_val, y_val = transform_array(x_val, y_val)
	x_train, y_train = transform_array(x_train, y_train)
	x_test, y_test = transform_array(x_test, y_test)

	return x_train, y_train, x_test, y_test, x_valid, y_valid

if __name__ == "__main__":
	preprocess()
