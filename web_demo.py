# -*- coding: utf-8 -*-
from os import listdir
import os
from app import app
import urllib.request
from flask import Flask, render_template, request, redirect, flash
from wtforms import Form, TextField, validators, SubmitField, DecimalField, IntegerField
from werkzeug.utils import secure_filename
from keras import backend as K
from PIL import Image
import torch
from torch.autograd import Variable
import craft

import dataset
import utils
import models.VGG_BiLSTM_CTC as crnn
import models.ResNet_BiLSTM_CTC as crnn


"""
config
"""
model_path = './trained_weights/VGG_BiLSTM_CTC_cs.pth'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\';:.-! )"$\\#%,@&/?([]{}+-=*^|'
nclass = len(alphabet)  + 1
nc = 1
imgH = 32
hidden_size = 256
converter = utils.strLabelConverter(alphabet)

model = crnn.CRNN(imgH, nc, nclass, hidden_size)
if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load(model_path))


def show_predict(image_directory, filename):
    output_dir = 'static/images/cropped_craft'
    prediction_result = craft.detect_text(image_directory, output_dir, crop_type='polly',
                                          export_extra=True, refiner=False, cuda=True)
    cropped_dir = output_dir + "/" + filename[:-4] + "_crops"
    transformer = dataset.resizeNormalize((100, 32))
    predicted_text = ""
    
    for cropped in listdir(cropped_dir):
        img = cropped_dir + "/" + cropped
        image = Image.open(img).convert("L")
        image = transformer(image)
    
        if torch.cuda.is_available():
            image = image.cuda()
    
        image = image.view(1, *image.size())
        image = Variable(image)
        model.eval()
        preds = model(image)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        predicted_text = predicted_text + sim_pred[2:len(predicted_text) - 1] + "\n"
    
    return predicted_text, output_dir


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/images')
MODEL_DIR = os.path.join(BASE_DIR, 'model')


# Create app
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route("/", methods=['GET', 'POST'])
def home():
    K.clear_session()
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath=os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            result, output_dir = show_predict(filepath, filename)
            filename = filename[:-4] + ".png"

            response = {}
            response['path'] = output_dir + "/" + filename
            response['text'] = result 
            return render_template('index.html', response=response)

        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(request.url)
        
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))

    app.run(host="0.0.0.0", port=8081, debug=True )


