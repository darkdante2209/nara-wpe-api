import os
from flask import Flask, request, render_template, jsonify, make_response
from flask_cors import CORS
import soundfile as sf
import pathlib
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
from nara_wpe.utils import stft, istft
from nara_wpe.tf_wpe import wpe


project_root = pathlib.Path(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
))

#Parameters
stft_options = dict(size=512, shift=128)
channels = 1
delay = 3
alpha=0.99
taps = 10
frequency_bins = stft_options['size'] // 2 + 1
iterations=5

app = Flask(__name__)
CORS(app)
@app.route('/')
def index():
    return render_template('index_remove_reverb.html')

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

@app.route('/remove_reverb/<file_path>')
def remove_reverb(file_path):
    print(file_path)
    file_path = file_path.replace('=', '/')
    file_name = '.'.join(file_path.split('.')[0:-1])
    signal_list, sampling_rate = sf.read(str(file_path))
    y = np.stack(signal_list, axis=0)
    y = np.reshape(y, (1, y.shape[0]))
    Y = stft(y, **stft_options).transpose(2, 0, 1)
    Z = wpe(
        Y,
        taps=taps,
        delay=delay,
        iterations=iterations,
        statistics_mode='full'
    ).transpose(1, 2, 0)
    z = istft(Z, size=stft_options['size'], shift=stft_options['shift'])
    # save the dereverbration sound file
    z_save = np.reshape(z, (z.shape[1],))
    out_file_path = file_name + '_removed_reverb.wav'
    sf.write(str(out_file_path), z_save, sampling_rate)
    print("Finish Offline Algorithm")
    print('add reverb done!!')
    print (out_file_path)
    res = make_response(jsonify({"out_file_path":out_file_path, "message": "Khử tiếng vọng hoàn tất!"}))
    return res, 200

if __name__ == "__main__":
    app.debug = True
    app.run(host='127.0.0.1', port='9002')
    #app.run(port='8080')