#Avant de commencer, n'oublier pas d'activer tensorflow :
#source ~/tensorflow/bin.activate
import data_utils
#On charge les donnees mnist :
from keras.datasets import mnist

#On charge quelques classes utiles pour la suite
#Pas specialement pour faire du deep mais plutot
#pour des calculs

import numpy as np
from keras.utils import np_utils

#Notre reseau aura une structure de chaine (feedword),
#On importe donc lo modeles correspondant

from keras.models import Sequential

#Pour ce premier tp, nous aurons besoin de definir des
#couches completement connextees. On importe la couche
#qui permet de le faire

from keras.layers import Dense


#A present, nous allons cherger les donnees sur lesquelles
#nous entrainerons puis testerons notre modele. Pour cela,
#nous allons utiliser les fonctions de la bibliotheque keras
#pour charger les donnees mnist.

(X_data, Y_data) = data_utils.load_data("data", "data.txt")
Y_data = data_utils.transform_data_Y(Y_data)

(X_data, Y_data) = data_utils.randomize(X_data, Y_data)




if(len(X_data) > 0):
    nb_features = len(X_data[0])



edge = round(len(X_data)*0.6)


X_test = np.array(X_data)[edge:,:]
Y_test = np.array(Y_data)[edge:,:]

X_train = np.array(X_data)[:edge, :]
Y_train = np.array(Y_data)[:edge, :]




# (X_train, Y_train), (X_test, Y_test) = mnist.load_data()



#Les sorties sont des entiers entre 0 et 9. Nous les transformons
#en categories :


nb_classes = len(data_utils.get_result_tabs(Y_data))


nb_features =  X_train.shape[1]

total_features = X_train.shape[0] * X_train.shape[1]


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#On normalise les valeurs d'entree de 0-255 a 0-1
#
# X_train = X_train / 255
# X_test = X_test / 255

#Les sorties sont des entiers entre 0 et 9. Nous les transformons
#en categories :
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
nb_classes = Y_test.shape[1]


np.savez("data.npz", (X_train, Y_train, X_test, Y_test))

#Nous avons donc a notre disposition tous les elements
#necessaires pour faire construire notre reseau, l'entrainer
#le tester etc.
#Syntexes (voir http://keras.io):
#Dense()


def baseline_model():
    model = Sequential()
    model.add(Dense(nb_features,input_dim=nb_features, kernel_initializer='normal', activation='relu'))
    model.add(Dense(nb_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = baseline_model()
model.summary()

#On entraine a present le modele :

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=10)

#On affiche l'evaluation de notre modele :

scores = model.evaluate(X_test, Y_test)

model.save("model.ke")
print("Baseline accuracy: %.2f%%" % (scores[1]*100))
