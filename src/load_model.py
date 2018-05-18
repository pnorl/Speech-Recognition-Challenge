import keras
from preprocessing import preprocess2

print("***Reading data***")
x_train, y_train, x_test, y_test, x_valid, y_valid=preprocess2()

print("***Loading model***")
#Filepath+filename
filepath = r'../model/'+'cnn.model'
model = keras.models.load_model(filepath)

print("***Evalute model***")
score = model.evaluate(x_test, y_test, verbose=1)
print(score)