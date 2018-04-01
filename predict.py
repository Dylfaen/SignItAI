from keras import *
import numpy as np
import data_utils


model = models.load_model("model.ke")

(X_train, Y_train, X_test, Y_test, X_predict, Y_predict, X_data, Y_data, old_Y_data) = np.load('data.npz')['arr_0']




p = model.predict_classes(X_predict)

truths = []
for i, v in enumerate(p):
    test = p[i]
    print(data_utils.get_result_tabs(old_Y_data)[test]) #On affiche le label de la prédiction
    print('-')
    solution = np.argmax(Y_predict[i])
    print(data_utils.get_result_tabs(old_Y_data)[solution]) #On affiche le label de la solution
    print('--')
    truth = test == solution
    print(truth) # On affiche si le résultat est cohérent
    truths.append(truth)
    print('===========')

trues = 0
falses = 0
for t in truths:
    if(t):
        trues+=1
    else:
        falses+=1

print("verdict : %.2f%%" % (trues/len(truths)*100)) #On calcule la proportion de réussite
