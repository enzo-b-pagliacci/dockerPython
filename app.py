from flask import Flask, jsonify, request
from flask_restful import Api
from flask_cors import CORS
import os
import shutil
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import models



app = Flask("ChoroAPI")
api = Api(app)

cors = CORS(app, resources={r"/choro/*": {"origins": "*"}})

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit(
        '.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/choro')
def upload_form():
    return jsonify('hello World')


@app.route('/choro', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        #métodos que pegam o arquivo da requisição post enviada através do formData do react native e salva na pasta uploads
        #com nome de choro
        file = request.files['file']
        filename = "choro"
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        #método que transforma o arquivo de áudio para choro.wav
        main_folder = r'./uploads'


        #método que irá ler o arquivo e mandar a I.A. predizer
        model = tf.keras.models.load_model('./content/modelo.h5')

        #necessário para transformar o audio em waveform
        def decode_audio(audio_binary):
          # Decode WAV-encoded audio files to `float32` tensors, normalized
          # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
          audio, _ = tf.audio.decode_wav(contents=audio_binary)
          # Since all the data is single channel (mono), drop the `channels`
          # axis from the array.
          return tf.squeeze(audio, axis=-1)

        #transforma o audio em waveform
        def get_waveform(file_path):
          audio_binary = tf.io.read_file(file_path)
          waveform = decode_audio(audio_binary)
          return waveform

        #função utilitária para converter formas de onda em espectrogramas:
        def get_spectrogram(waveform):
          input_len = 16000
          waveform = waveform[:input_len]
          zero_padding = tf.zeros(
              [16000] - tf.shape(waveform),
              dtype=tf.float32)
          # Cast the waveform tensors' dtype to float32.
          waveform = tf.cast(waveform, dtype=tf.float32)
          # Concatenate the waveform with `zero_padding`, which ensures all audio
          # clips are of the same length.
          equal_length = tf.concat([waveform, zero_padding], 0)
          # Convert the waveform to a spectrogram via a STFT.
          spectrogram = tf.signal.stft(
              equal_length, frame_length=255, frame_step=128)
          # Obtain the magnitude of the STFT.
          spectrogram = tf.abs(spectrogram)
          # Add a `channels` dimension, so that the spectrogram can be used
          # as image-like input data with convolution layers (which expect
          # shape (`batch_size`, `height`, `width`, `channels`).
          spectrogram = spectrogram[..., tf.newaxis]
          return spectrogram



        # renomear o arquivo 
        def rename_file(file):
          file_name, file_extension = os.path.splitext(file)
          file_extension = ".wav"
          return f'{file_name}{file_extension}'

        def file_loop(root, dirs, xfiles):
            for file in xfiles:
              new_file_name = rename_file(file)
              old_file_full_path = os.path.join(root, file)
              new_file_full_path = os.path.join(root, new_file_name)
              shutil.move(old_file_full_path, new_file_full_path)
              waveform = get_waveform(new_file_full_path)
              return waveform
             
        try:
          for root, dirs, xfiles in os.walk(main_folder):
              waveform = file_loop(root, dirs, xfiles)
        except:
          return jsonify('Não transformou o áudio em .wav')

        

        try:
          pass
          #Onde passamos o path do arquivo .wav
          # try:
          #   find = False
          #   contador = 0
            
          #   while find == False:
          #     for root, dirs, xfiles in os.walk(main_folder):
          #       if xfiles == ['choro.wav']:
          #         find = True
          #         for x in xfiles:
                        
                  
          #       else:
          #         contador = contador + 1
          #         if contador > 70000:
          #           return jsonify('não recebeu o áudio')
          #         else:
          #           break
          # except:
          #   return jsonify('Não pegou o audio e passou o waveform')

          try:
            spectrogram = get_spectrogram(waveform)
          except:
            return jsonify('Não pegou o espectrograma')

          try:
            audio_ds = []

            audio_ds.append(spectrogram.numpy())
          
            audio_ds = np.array(audio_ds)

          except:
            return jsonify('não mudou o shape')


          try:
            y_pred = np.argmax(model.predict(audio_ds), axis=1)

            indice = 0

          
            predicao = y_pred[indice]
          
          except:
            return jsonify('não conseguiu predizer')


          try:
            predicao = predicao.item()

            resposta = {
                "Fome": predicao
            }

            retornoAPI = json.dumps(resposta)
            

            # retorna a resposta da I.A. para o front end
            return retornoAPI
          
          except:
            return jsonify('não conseguiu converter e retornar')
            
        except:
          return jsonify('não foi')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
