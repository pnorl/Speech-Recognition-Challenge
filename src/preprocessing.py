import os
from glob import glob
import re
import scipy

import numpy as np


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

def resample(labels,fnames,L=16000):
	for label, fname in zip(labels, fnames):
	    sample_rate, samples = scipy.io.wavfile.read(os.path.join(train_data_path, label, fname))
	    samples = pad_audio(samples) #Custom function (pads samples < 16000)
	    if len(samples) > L:
	        n_samples = chop_audio(samples) #Custom function (chops samples > 16000)
	    else: n_samples = [samples] 
	    for samples in n_samples:
	        resampled = scipy.signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))
	        _, _, specgram = log_specgram(resampled, sample_rate=new_sample_rate)
	        y_train.append(label)
	        x_train.append(specgram)



def main():
	# Data folders
	root_path = r'..'

	#Join paths
	train_data_path = os.path.join(root_path, 'data', 'train','audio')


	labels, fnames = list_wavs_fname(train_data_path)


main()