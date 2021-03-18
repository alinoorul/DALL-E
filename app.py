import io
import sys
import requests
import PIL

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from dall_e          import map_pixels, unmap_pixels, load_model
from IPython.display import display, display_markdown

import os

from flask import Flask, flash, request, redirect, url_for
import torch.nn.functional as F

def preprocess(img):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return map_pixels(img)

# This can be changed to a GPU, e.g. 'cuda:0'.
dev = torch.device('cpu')

enc = load_model("./dalle_files/encoder.pkl", dev)
dec = load_model("./dalle_files/decoder.pkl", dev)

target_image_size = 256

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        UPLOAD_FOLDER = 'static'
    )

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/show', methods=['GET'])
    def show_file():
    	x = preprocess(PIL.Image.open('./static/img.jpg'))
    	T.ToPILImage(mode='RGB')(x[0]).save('./static/preprocessed.jpg')
    	z_logits = enc(x)
    	z = torch.argmax(z_logits, axis=1)
    	z = F.one_hot(z, num_classes=enc.vocab_size).permute(0, 3, 1, 2).float()
    	x_stats = dec(z).float()
    	x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
    	x_rec = T.ToPILImage(mode='RGB')(x_rec[0])
    	x_rec.save("./static/reconstructed.jpg")

    	x_rec.show()
    	return '''<!doctype html>
	    <title>Reconstructed Image</title>
	    <h1>Original Image</h1>
	    <img src="static/img.jpg" alt="Original Image">
	    <h1>Preprocessed Image</h1>
	    <img src="static/preprocessed.jpg" alt="Preprocessed Image">
	    <h1>Reconstructed Image</h1>
	    <img src="static/reconstructed.jpg" alt="Reconstructed Image">'''

    @app.route('/', methods=['GET', 'POST'])
    def upload_file():
    	if request.method == 'POST':
    		if 'file' not in request.files:
    			flash("no file uploaded")
    			return redirect(request.url)
    		file = request.files['file']
    		if file.filename == '':
    			flash('no selected file')
    			return redirect(request.url)
    		if file:
    			filename='img.jpg'
    			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    			return redirect(url_for('show_file'))
    	
    	return '''<!doctype html>
	    <title>Upload new File</title>
	    <h1>Upload new File</h1>
	    <form method=post enctype=multipart/form-data>
	      <input type=file name=file>
	      <input type=submit value=Upload>
	    </form>
	    '''
	

    return app