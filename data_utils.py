import json
import numpy as np
from PIL import Image
import random
import copy


# Retourne la valeur en nuance de gris du pixel en paramètre
def average(pixel):
    return (pixel[0] + pixel[1] + pixel[2]) / 3

# Retourne le tableau de pixels grisés et normalisé à partir du nom de fichier en paramètre
def getImage(filename):
    image = Image.open(filename) #On ouvre l'image
    image = image.resize((50, 50), Image.ANTIALIAS) #On la réduit
    pix = image.load() #On récupère le tableau de pixels
    (i,j) = image.size
    res = []
    for x in range(i):
        for z in range(j):
            res.append((255-average(pix[x,z]))/255) #On transforme chaque pixel en nuance de gris et on le normalise
    return res

# On transforme les données json en un tableau aplati pour fonctionner avec Keras
def flatten(workspace, raw_json) :
    flattened_data_X = [];
    flattened_data_Y = [];
    tupled_data = []
    # On parcoure les profils
    for profile  in raw_json['trainingProfiles']:
        #On parcour les signatures de chaque profil
        for signature in profile['signatures']:

            temp_array = [] #La ligne représentant une signature

            total_time = 0
            strokes_len = 0
            if('stroke' in signature): #On vérifie que la signature n'est pas vide
                strokes_len = len(signature['strokes'])
                if(strokes_len>0) :
                    total_time = signature['strokes'][strokes_len-1]['stopTime'] - signature['strokes'][0]['startTime'] #On calcule le temps de la signature
            image = getImage(workspace + "/" + signature['filename']) #On récupère le tableau de l'image
            temp_array.append(strokes_len) #On ajoute le critère du nombre de traits
            temp_array.append(total_time) #On ajoute le critère de la durée de la signature

            flattened_data_Y.append([profile['title']]) #On ajoute le titre comme label de la signature

            for composante in image: #Pour chaque pixel
                temp_array.append(composante) #On ajoute le pixel au tableau
            flattened_data_X.append(temp_array) #On ajoute la signature aux données
    return (flattened_data_X, flattened_data_Y)

# Retourne le tableau des labels
def get_result_tabs(data_Y) :
    titles_found = []
    result_tab = []
    for data in data_Y:

        title = data[0]
        if (title not in titles_found) :
            titles_found.append(title)

    i = 0;
    for title in titles_found:
        result_tab.append((i, title))
        i+= 1

    return result_tab

# On transforme les labels
def transform_data_Y(Y_data):

    data_Y = copy.deepcopy(Y_data)
    result_tab = get_result_tabs(data_Y)
    new_data_Y = []

    for data in data_Y:
        for ref in result_tab:
            if(ref[1] == data[0]):
                data = [ref[0]]

        new_data_Y.append(data)
    return new_data_Y

# On charge le json et on aplati les données
def load_data(workspace, filename) :
    file = open(workspace + "/" + filename);
    json_data = json.load(file);
    return flatten(workspace, json_data)

# On mélange les signatures au sein des données
def randomize(data_X, data_Y):
    tuples = []
    new_data_X = []
    new_data_Y = []
    for index, value in enumerate(data_X): #On transforme le tuples de deux listes en liste de tuples pour garder la correspondance critères / label
        tuples.append((data_X[index], data_Y[index]))
    random.shuffle(tuples) # On mélange la liste de tuples
    for tuple in tuples: #On retransforme la liste de tuples en tuple de listes
        new_data_X.append(tuple[0])
        new_data_Y.append(tuple[1])

    return(new_data_X, new_data_Y) #On retourne les données mélangées
