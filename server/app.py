from flask import Flask
from model import encode, decode, validate
from flask import request, url_for, send_file, json
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
CORS(app)


@app.get('/')
def index():
    # encode()
    return '<h1>Hi from Flask!!!</h1>'


@app.post('/image/encode')
def imageEncode():
    image = request.files['image']
    filename = secure_filename(f'{datetime.now()}-{image.filename}')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)
    encoded_filename = encode(filepath, request.form['text'])
    return {'url': url_for('download_encoded_image', name=encoded_filename)}


@app.get('/image/encoded/<name>')
def download_encoded_image(name):
    return send_file(os.path.join('encoded_images', name))


@app.post('/image/decode')
def imageDecode():
    image = request.files['image']
    filename = secure_filename(f'{datetime.now()}-{image.filename}')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)
    text = decode(filepath)
    return { 'text': text }


app.run(debug=True, port=3030)
