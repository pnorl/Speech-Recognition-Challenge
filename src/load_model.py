import keras
from preprocessing import preprocess2
from sklearn.metric import classification_report

print("***Reading data***")
path = r'../data/train_preprocessed/'
filename = r'raw_wav'
npzfile = np.load(path+filename+'.npz')

x_test, y_test = npzfile['x_test'],npzfile['y_test']

print("***Loading model***")
#Filepath+filename
filepath = r'../model/'+'raw_wav.model'
model = keras.models.load_model(filepath)

print("***Evalute model***")
pred = model.predict(x_test, verbose=1)


#Define mappings for transforming predictions to labels
legal_labels = 'yes no up down left right on off stop go silence unknown'.split()
idxToLabel={index:label for index,label in enumerate(legal_labels)}

#Transform predictions to labels
labels_idx = np.argmax(pred,axis=1)

#labels = [idxToLabel[i] for i in np.argmax(pred,axis=1)]

freq = np.bincount(labels_idx) #Frequency of predictions
for label,no in zip(legal_labels,freq):
    print(label,no)


true_idx = np.argmax(y_test,axis=1)
print(classification_report(y_test,labels_idx))