import json
import numpy as np
from PIL import Image
import random


def average(pixel):
    return (pixel[0] + pixel[1] + pixel[2]) / 3

def getImage(filename):
    image = Image.open(filename)
    image = image.resize((100, 100))
    image.save('test.png')
    pix = image.load()
    (i,j) = image.size
    res = []
    for x in range(i):
        for z in range(j):
            res.append((255-average(pix[x,z]))/255)

    # print(len(res))
    return res

def flatten(workspace, raw_json) :
    flattened_data_X = [];
    flattened_data_Y = [];
    tupled_data = []
    #print(raw_json['trainingProfiles'])
    for profile  in raw_json['trainingProfiles']:
        for signature in profile['signatures']:

            temp_array = []

            total_time = 0
            strokes_len = 0
            if('stroke' in signature):
                strokes_len = len(signature['strokes'])
                if(strokes_len>0) :
                    total_time = signature['strokes'][strokes_len-1]['stopTime'] - signature['strokes'][0]['startTime']
            image = getImage(workspace + "/" + signature['filename'])
            temp_array.append(strokes_len)
            temp_array.append(total_time)

            flattened_data_Y.append([profile['title']])

            for composante in image:
                temp_array.append(composante)
            flattened_data_X.append(temp_array)
    return (flattened_data_X, flattened_data_Y)

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

def transform_data_Y(data_Y):

    result_tab = get_result_tabs(data_Y)
    new_data_Y = []

    for data in data_Y:
        for ref in result_tab:
            if(ref[1] == data[0]):
                data = [ref[0]]

        new_data_Y.append(data)
    return new_data_Y

def load_data(workspace, filename) :
    file = open(workspace + "/" + filename);
    json_data = json.load(file);
    return flatten(workspace, json_data)

def randomize(data_X, data_Y):
    tuples = []
    new_data_X = []
    new_data_Y = []
    for index, value in enumerate(data_X):
        tuples.append((data_X[index], data_Y[index]))
    random.shuffle(tuples)
    for tuple in tuples:
        new_data_X.append(tuple[0])
        new_data_Y.append(tuple[1])

    return(new_data_X, new_data_Y)
