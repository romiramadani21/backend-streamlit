import subprocess
import sys
import streamlit as st
import os
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename
import flask
from flask import request

model = tf.keras.models.load_model('cnn_walet.h5')

def preprocess_image(image):
    image = image.resize((150, 150))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(image):
    image = preprocess_image(image)
    prediction = model.predict(image)

    labels = ['Grade A', 'Grade B', 'Grade C']
    class_index = np.argmax(prediction)
    class_label = labels[class_index]
    confidence = prediction[0][class_index] * 100

    return class_label, confidence

app = flask.Flask(__name__)
run_with_ngrok(app)

@app.route('/', methods=['GET'])
def index():
    return "<center><div><image src='https://thumbs.gfycat.com/InfiniteRemarkableDesertpupfish-size_restricted.gif'></image></div></center>"

@app.route('/predict', methods=['POST'])
def upload():
    data = {"success": False}
    namaFile = ''
    if request.method == 'POST':
        file = request.files['file']

        if file.filename == '':
            print('Tidak ada file')
        else:
            print('File berhasil di simpan')
            filename = secure_filename(file.filename)
            file.save('data_test/' + file.filename)
            namaFile = 'data_test/' + file.filename

            image = Image.open(namaFile)
            class_label, confidence = predict(image)

            data['success'] = True
            data['label'] = class_label
            data['acc'] = confidence
            data['name'] = class_label
            data['all_label'] = [confidence]

        print(data)
        return flask.jsonify(data)
    else:
        return '<h1>Method Salah</h1>'

def install_packages():
    packages = [
        'pyngrok',
        'gevent',
        'flask',
        'keras',
        'numpy',
        'pandas',
        'flask-ngrok',
        'tensorflow',
        'pillow'
    ]

    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == '__main__':
    install_packages()
    app.run()
