from flask import Flask, render_template
from flask import request
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib # to access files inside folders eaasily
import scipy
from skimage.transform import resize, rescale
from PIL import Image
from sklearn import metrics 
import skimage.io
import skimage
from skimage import feature
from glob import glob
app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, "static/upload/")
MODEL_PATH = os.path.join(BASE_PATH, "static/models/")

## ---------------------Load Models ------------------------------
model_sgd_path = os.path.join(MODEL_PATH, "dsa_image_classification_sgd.pickle")
scaler_path = os.path.join(MODEL_PATH, "dsa_scaler.pickle")
model_sgd = pickle.load(open(model_sgd_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

@app.errorhandler(404)
def error404(error):
   message = "ERROR 404 OCCURRED. Page not found. Please go to home page and try again"
   return render_template("error.html", message = message)

@app.errorhandler(405)
def error405(error):
   message = "ERROR 405. Method not found" 
   return render_template("error.html", message = message)
    
@app.errorhandler(500)
def error500(error):
   message = "INTERNAL ERROR 500. Error occurs in the program"
   return render_template("error.html", message = message)

@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        upload_file = request.files["image_name"]
        filename = upload_file.filename
        print("The filename that has been uploaded =", filename)
        # know the extension of filename
        # only accept .jpg, .png, .jpeg, PNG
        ext = filename.split(".")[-1]
        print("The extension of the filename =", ext)
        if ext.lower() in ['jpg', 'png', 'jpeg']:
            # saving the file
            path_save = os.path.join(UPLOAD_PATH, filename)
            upload_file.save(path_save)
            print("file saved successfully")
            # send it to the model
            results = pipeline_model(path_save,scaler,model_sgd)
            hei = getheight(path_save)
            print(results)
            return render_template("upload.html", fileupload = True, extension = False, data = results, image_filename = filename, height = hei)
            
        else:
            print("Use only the extension with .jpg, .png, .jpeg")
            return render_template("upload.html", extension = True, fileupload = False)
    
    else:             
         return render_template("upload.html", fileupload = False, extension = False)

@app.route('/about/')
def about():
    return render_template('about.html')


def getheight(path):
    img = skimage.io.imread(path)
    h,w,_ = img.shape
    aspect = h/w
    given_width = 300
    height = given_width*aspect
    return height



def pipeline_model(path, scaler_transform, model_sgd):
    # pipeline model
    image = skimage.io.imread(path)
    # Resize the image
    image_resize = skimage.transform.resize(image, (80, 80))
    # After resizing the image gets normalised
    # So, rescaling it is important
    image_scale = 255*image_resize
    image_transform = image_scale.astype(np.uint8) # converting float values to int
    # converting rgb to gray
    gray = skimage.color.rgb2gray(image_transform)
    feature_vector = feature.hog(gray,
                                orientations = 10,
                                pixels_per_cell = (8,8),
                                cells_per_block = (2,2))
    
    #scaling
    scale = scaler_transform.transform(feature_vector.reshape(1,-1))
    result = model_sgd.predict(scale)

    # decision value
    decision_value = model_sgd.decision_function(scale).flatten()

    labels = model_sgd.classes_

    # Calculating confidence interval
    z = scipy.stats.zscore(decision_value)
    prob_value = scipy.special.softmax(z)

    # Getting top 5 labels
    top_5_prob_ind = prob_value.argsort()[::-1][:5]
    top_labels = labels[top_5_prob_ind]
    top_prob = prob_value[top_5_prob_ind]

    # Putting all this in dictionary
    top_dict = {}

    for key, val in zip(top_labels, top_prob):
      top_dict[key] = np.round(val, 2)

    return top_dict

if __name__ == "__main__":
    app.run(debug = True)