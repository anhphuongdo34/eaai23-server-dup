from flask import Flask, request
import pandas as pd
from flask_cors import CORS 
import base64
import numpy as np
import cv2
import gc
from helpers.results import get_results

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_geek():
    return '<h1>Hello from Flask & Docker</h2>'

# save audio_features to models/audio_features.csv
@app.route('/data', methods=["POST"], strict_slashes=False)
def audio_features():
    audio_features = request.json['audio_features']
    res = [i for i in audio_features if i]

    df = pd.DataFrame(res)
    df.drop_duplicates()
    df.to_csv('helpers/models/usersData.csv')
    return "data saved to models/usersData.csv"


# receive base64 image from client
@app.route('/webcam_capture', methods=["POST"], strict_slashes=False)
def webcam_capture():
    # base64 image
    imageSrc = request.json['imageSrc'] 

    # handle imageSrc
    header, data = imageSrc.split(',', 1)
    r = base64.b64decode(data)
    imageArr = np.frombuffer(r, dtype=np.uint8)
    image = cv2.imdecode(imageArr, cv2.IMREAD_UNCHANGED)

    cv2.imwrite('./helpers/models/face_img.png', image)

    gc.collect()
    return "image saved to models/face_img.png"

# send prediction result
@app.route('/recommend_song')
def recommend_song():
    valence, arousal, plot, track_ids, emotion = get_results()

    result = {
        'track_ids': track_ids,
        'valence' : valence,
        'arousal' : arousal,
        'plot' : plot,
        'emotion': emotion
    }
    
    gc.collect()
    return result

# receive feedback on the song recommendation
@app.route('/song_feedback', methods=["POST"], strict_slashes=False)
def song_feedback() :
    track_id = request.json['trackId']
    val = request.json['valence']
    ars = request.json['arousal']
    rate = request.json['rate']

    with open('feedback.txt', 'a') as f:
        f.write(str(track_id)+','+str(val)+','+str(ars)+','+rate+"\n")

    gc.collect()
    return 'feedback was saved to the file feedback.txt'


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)