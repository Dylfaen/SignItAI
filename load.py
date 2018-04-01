import data_utils
import numpy as np
import copy
from keras.utils import np_utils



#A present, nous allons cherger les donnees sur lesquelles
#nous entrainerons puis testerons notre modele. Pour cela,
#nous allons utiliser les fonctions de la bibliotheque keras
#pour charger les donnees mnist.

(X_data, Y_data) = data_utils.load_data("data", "data.txt")

(X_data, Y_data) = data_utils.randomize(X_data, Y_data)

old_Y_data = data_utils.transform_data_Y(Y_data)
Y_data = copy.deepcopy(old_Y_data)

first_edge = round(len(X_data)*0.6)
second_edge = first_edge + round(len(X_data)*0.2)


X_train = np.array(X_data)[:first_edge, :]
Y_train = np.array(Y_data)[:first_edge, :]

X_test = np.array(X_data)[first_edge:second_edge,:]
Y_test = np.array(Y_data)[first_edge:second_edge,:]

X_predict = np.array(X_data)[second_edge:,:]
Y_predict = np.array(Y_data)[second_edge:,:]





X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_predict = X_predict.astype('float32')


Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
Y_predict = np_utils.to_categorical(Y_predict)
nb_classes = Y_test.shape[1]


np.savez("data.npz", (X_train, Y_train, X_test, Y_test, X_predict, Y_predict, X_data, Y_data, old_Y_data))
