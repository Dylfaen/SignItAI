from keras import *
import numpy as np


model = models.load_model("model.ke")

(X_train, Y_train, X_test, Y_test) = np.load('data.npz')['arr_0']


p = model.predict(X_test[0:5])
print(list(p[0]).index(max(p[0])))
print(Y_test[0])
