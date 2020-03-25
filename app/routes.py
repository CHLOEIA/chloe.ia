from flask import render_template, request, redirect
from app import app
import os
from slugify import slugify
from helpers import allowed_image, allowed_image_filesize
from werkzeug.utils import secure_filename
import time
import tensorflow as tf
import keras
from keras.models import load_model
import cv2
import numpy as np
from keras import backend as K

from io import BytesIO
from scipy import misc


app.config["IMAGE_UPLOADS"] = "/Users/valentinbeggi/Desktop/site_insecte/app/static/images_uploaded"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]
app.config["MAX_IMAGE_FILESIZE"] = 0.5 * 1024 * 1024

classes = ['AGELENIDAE','AMAUROBIIDAE','ANYPHAENIDAE','DICTYNIDAE','DYSDERIDAE','ERESIDAE','GNAPHOSIDAE','INDET','LINYPHIIDAE','LIOCRANIDAE','LYCOSIDAE','OONOPIDAE','PHILODROMIDAE','PHOLCIDAE','SALTICIDAE','SYNAPHRIDAE','THERIDIIDAE','THOMISIDAE','TITANOECIDAE','ZODARIIDAE','ZOROPSIDAE']

dirpath = os.getcwd()
print("current directory is : " + dirpath)
foldername = os.path.basename(dirpath)
print("Directory name is : " + foldername)
scriptpath = os.path.realpath(__file__)
print("Script path is : " + scriptpath)


graph1 = tf.Graph()
with graph1.as_default():
    session1 = tf.Session(graph=graph1)
    with session1.as_default():
        model_1 = load_model('./app/static/models/model.h5')


@app.route('/')

@app.route('/index')
def index():
    return render_template('index2.html', title='Home')


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":

        if request.files:
            if 'filesize' in request.cookies:
                if not allowed_image_filesize(request.cookies["filesize"]):
                    print("Filesize exceeded maximum limit")
                    return redirect(request.url)

                image = request.files["image"]

                if image.filename == "":
                    print("No filename")
                    return redirect(request.url)

                if allowed_image(image.filename):

                    #image_name = slugify(image.filename)+'.png'
                    filename = secure_filename(image.filename)
                    filepath = os.path.join(app.config["IMAGE_UPLOADS"], filename)
                    image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

                    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
                    if img.shape != (256, 256, 3):
                        img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
                    img = np.expand_dims(img, axis=0)

                    K.set_session(session1)
                    with graph1.as_default():
                        prediction = model_1.predict(img)
                        print(prediction)
                        score = str(round(100*np.amax(prediction), 2))
                        print(score)
                    return render_template('prediction2.html', \
                    filename='images_uploaded/'+filename, score = score, classe = classes[np.argmax(prediction)])

                else:
                    print("That file extension is not allowed")
                    return redirect(request.url)

    return render_template("index2.html")
