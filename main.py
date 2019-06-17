import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from sklearn import datasets

iris = datasets.load_iris()

X = iris['data']
Y = np_utils.to_categorical(iris['target'])

model = Sequential()

model.add(Dense(15, input_dim=4, activation = 'relu'))
model.add(Dense(15, input_dim=4, activation = 'relu'))

model.add(Dense(3, activation = 'softmax'))

model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X, Y, epochs = 1000)

model.save('iris_model.hdf5')

print("\n\n\n")

scores = model.evaluate(X, Y, verbose=0)
print("Acuracia =", (round(scores[1], 5)*100))


'''Teste 1'''
test = np.array([[4.4, 2.9, 1.4, 0.2]])
result = model.predict(test,verbose=0)
print('Entrada =', test[0], ' Saida =',[round(result[0][0], 5), round(result[0][1], 5), round(result[0][2], 5)])


'''Teste 2'''
test = np.array([[5.2,2.7,3.9,1.4]])
result = model.predict(test,verbose=0)
print('Entrada =', test[0], ' Saida =',[round(result[0][0]*100, 5), round(result[0][1], 5), round(result[0][2], 5)])

'''Teste 3'''
test = np.array([[6.8,3.0,5.5,2.1]])
result = model.predict(test,verbose=0)
print('Entrada =', test[0], ' Saida =',[round(result[0][0], 5), round(result[0][1], 5), round(result[0][2], 5)])
