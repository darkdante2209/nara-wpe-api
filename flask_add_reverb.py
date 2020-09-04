import os
import requests
from flask import Flask, send_file, escape, request, render_template, jsonify, make_response, session
import random
from flask_cors import CORS
from AudioLib.AudioProcessing import AudioProcessing
import pathlib

project_root = pathlib.Path(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
))

app = Flask(__name__)
CORS(app)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST','GET'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        print(filename)
        if  os.path.splitext(filename)[1][1:].strip() not in ['mp3','wav','flac']:
            return render_template('index.html', filename='{} file not support. select mp3, wav or flac!'.format(filename))
        file_path = 'static/upload/' + filename
        file.save(file_path)
        print('saved file: {}'.format(file_path))
        res = make_response(jsonify({"file_path": file_path, "message": "Đã lưu file : {} lên server".format(filename)}))
        return res
    return render_template('index.html')

@app.route('/add_reverb/<file_path>')
def add_reverb(file_path):
    print(file_path)
    file_path = file_path.replace('=', '/')
    file_name = '.'.join(file_path.split('.')[0:-1])
    sound1 = AudioProcessing(str(file_path))
    sound1.set_reverb(0.08, 0.7)
    out_file_path = file_name + '_added_reverb.wav'
    sound1.save_to_file(str(out_file_path))
    print('add reverb done!!')
    print (out_file_path)
    res = make_response(jsonify({"out_file_path":out_file_path, "message": "Thêm tiếng vọng hoàn tất!"}))
    return res, 200

if __name__ == "__main__":
    app.debug = True
    app.run(host='127.0.0.1', port='9002')
    #app.run(port='8080')