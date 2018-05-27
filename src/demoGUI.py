import sounddevice as sd
import keras
from scipy.io import wavfile
import time

import numpy as np
from preprocessing import SoundClip,preprocess_one_clip

from tkinter import *

def playback(recording):
    sd.play(recording)
    sd.wait()

def recordPredict():
    # Record audio
    sample_rate = 16000
    sd.default.samplerate = sample_rate #Set default samplerate to 16000
    duration = 1.0  # seconds
    plain_myrecording = sd.rec(int(duration * sample_rate), channels=1).reshape(sample_rate,)
    sd.wait()

    #Amplify signal to match training data
    myrecording=plain_myrecording*10000

    #Make into object
    clip = SoundClip(myrecording, sample_rate, None, None)

    # Preprocess
    x,_ = preprocess_one_clip(clip)
    x = np.array(x)
    x_model = x.reshape(1,98,40,1)

    # Make prediction
    pred = model.predict(x_model)

    # Turn prediction into word
    pred_idx = np.argmax(pred,axis=1)
    prediction = idxToLabel[pred_idx[0]]

    # Alter graphics
    txt = "You said: " + prediction
    lbl.configure(text = txt)

# Load the deep learning model
filepath = r'../model/'+'filbank_only1526901329.969079'
model = keras.models.load_model(filepath)

# Turn prediction into word
legal_labels = 'yes no up down left right on off stop go silence unknown'.split()
idxToLabel={index:label for index,label in enumerate(legal_labels)}

# Graphics
window = Tk()
window.title('Speech Recognition Demo')
window.geometry('350x200')
lbl = Label(window, text='Say something...', font =("Arial Bold", 20))
lbl.grid(column = 1, row = 0)
btn = Button(window, text = 'Record', font =("Arial Bold", 16), command=recordPredict)
btn.grid(column = 0, row = 0)
window.mainloop()
