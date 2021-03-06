import os
import io
import requests
from flask import Flask, send_file, escape, request, render_template, jsonify, make_response, session,send_from_directory
#from utils import cvtToWavMono16, split
import random
from flask_cors import CORS
from infer import get_model_new
UPLOAD_DIRECTORY = "./upload"

app = Flask(__name__)
CORS(app)
infer = get_model_new()
 
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
        #if  os.path.splitext(file_path)[1][1:].strip() not in ['wav']:
        #    out_file_path =u'"%s.wav"' %(file_path[:-4])
        #    print ('out_file_path:',out_file_path)
        #    str =u'ffmpeg -i "%s" -acodec pcm_u8 -ar 16000 -ac 1 "%s"' %(file_path,out_file_path)
        #    print (str)
        #    os.system(str)
        #    file_path =out_file_path

        print('saved file: {}'.format(file_path))
        res = make_response(jsonify({"file_path": file_path, "message": "Đã lưu file : {} lên server".format(filename)}))
        return res
    return render_template('index.html')

@app.route('/predict/<file_path>')
def predict(file_path):
    print(file_path)
    file_path = file_path.replace('=','/')
    out_file_path = infer(file_path)
    print('predict done!!')
    print('out_file_path:', out_file_path)
    res = make_response(jsonify({"out_file_path":out_file_path, "message": "Khử nhiễu hoàn tất!"}))
    return res

@app.route("/audio", methods=["POST"])
def process_audio():
    file = request.files['audio']
    file_path = os.path.join('./upload/', file.filename)
    file.save(file_path)
    out_file_path = infer(file_path)
    with open(out_file_path, 'rb') as bites:
        return send_file(io.BytesIO(bites.read()),attachment_filename='out.wav',mimetype='audio/wav')
@app.route("/audio_url", methods=["POST"])
def process_audio_url():
    file = request.files['audio']
    file_path = os.path.join('./upload/', file.filename)
    file.save(file_path)
    out_file_path = infer(file_path)
    return jsonify({'url':     out_file_path})

if __name__ == "__main__":
    app.debug = True
    app.secret_key = 'dangvansam'
    app.run(host='192.168.1.254', port='9002')