import numpy as np

import data_utils

from keras.models import Sequential

from keras.layers import Dense
from keras.layers import Dropout


(X_train, Y_train, X_test, Y_test, X_predict, Y_predict, X_data, Y_data, old_Y_data) = np.load('data.npz')['arr_0']


nb_classes = len(data_utils.get_result_tabs(Y_data))

nb_features =  X_train.shape[1]

total_features = X_train.shape[0] * X_train.shape[1]

def baseline_model():
    model = Sequential()
    model.add(Dense(nb_features,input_dim=nb_features, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = baseline_model()
model.summary()

#On entraine a present le modele :
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=5)

#On affiche l'evaluation de notre modele :
scores = model.evaluate(X_test, Y_test)
model.save("model.ke")
print("Baseline accuracy: %.2f%%" % (scores[1]*100))
