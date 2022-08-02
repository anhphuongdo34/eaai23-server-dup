from torch import abs_
from helpers.face_detection import face_detection
from helpers.load_model import Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helpers.recommendation import Recommendation
import os
import joblib
import base64
import gc

abs_path = os.getcwd()

def load_songs(dataset='spotify') :
    '''
    Args:
        dataset (str): default 'spotify.' Others include 'users' and 'both'
            - 'spotify': recommend songs from the Spotify 1m dataset
            - 'users': recommend songs from the user's Spotify playlist
            - 'both': recommend songs from both datasets
    Return a pandas dataframe containing songId, valence, and arousal
    '''
    spotify_df = pd.read_csv('helpers/models/spotifyData.csv').loc[:, ['id', 'valence', 'energy']]
    try :
        user_df = pd.read_csv('helpers/models/usersData.csv').loc[:, ['id', 'valence', 'energy']]
    except :
        print('user did not provide a personal playlist')

    if (dataset == 'spotify') :
        return spotify_df
    elif (dataset == 'users') :
        return user_df
    elif (dataset == 'both') :
        return pd.concat([spotify_df, user_df], axis=0, ignore_index=True)
    else :
        print('Incorrect dataset.')

def get_emotion(idx) :
    emotions = [
        'neutral',
        'happy',
        'sad',
        'surprise',
        'fear',
        'disgust',
        'anger',
        'contempts'
    ]

    return emotions[idx]

def plot_emotion(valence, arousal) :

    saved_graph_path = abs_path + '/helpers/models/user_graph.png'

    plt.rcParams["figure.figsize"] = [10, 10]
    plt.rcParams["figure.autolayout"] = True
    img = plt.imread(abs_path + "/helpers/models/affect_space.png")
    fig, axis = plt.subplots()
    img = axis.imshow(img, extent=[-1.25, 1.25, -1.25, 1.25])

    x = [valence]
    y = [arousal]

    axis.plot(x, y, 'or', markersize=20)
    axis.annotate('Your emotion', (valence, arousal), fontsize=15)

    plt.savefig(saved_graph_path)
    
    # convert the graph img to base64 to send to client
    return str(base64.b64encode(open(saved_graph_path, 'rb').read())).split("'")[1]

def get_results():
    '''
    Return valence (double), arousal (double), and the tracks ids of \
        the five songs with shortest distance to the predicted values \
        (List) 
    '''

    image_path = abs_path + '/helpers/models/face_img.png'
    reg_path = abs_path + '/helpers/models/InvNet50_best_weights.pt'
    class_path = abs_path + '/helpers/models/decision_tree.pkl'

    # detect all the faces in the image
    faces = face_detection(image_path)
    # load the trained model
    regression = Model(reg_path)
    classification = joblib.load(class_path)

    # getting the song recommendation
    song_df = load_songs()
    dist = Recommendation(song_df)

    # get the predicted valence and arousal
    predicted = regression.predict(faces)
    avg_predicted = np.mean(np.asarray(predicted), dtype=float, axis=0)
    rescale = lambda x : (x+1)/2
    scaled_predicted = rescale(avg_predicted)

    # get classification results
    emotion_idx = classification.predict([avg_predicted])[0]
    valence = avg_predicted[0]
    arousal = avg_predicted[1]

    # return best matched song IDs
    top5_songs = dist.emotion_to_songs(np.asarray(scaled_predicted))
    gc.collect()
    return valence, arousal, plot_emotion(valence, arousal), top5_songs.tolist(), get_emotion(emotion_idx)
