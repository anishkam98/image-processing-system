import os
from flask import render_template
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras.backend import set_session
import tensorflow as tf


breeds = ["Australian Shepherd", "Basenji","Bernese mountain","Borzoi","Bulldog","Cavalier King Charles Spaniel",
        "French bulldog","Glen of imaal terrier","Irish setter","Rhodesian ridgeback","Saluki",
        "Scottish deerhound","Shiba Inu","Shih Tzu","Soft coated wheaten terrier"]

sample_image_path = "/train/Australian shepherd/images (1).jpeg"

UPLOAD_FOLDER = 'static/uploads'

#Allowed files
ALLOWED_EXTENSIONS = {'png','jpg','jpeg','gif'}


def load_model_from_file():
    #set up the machine learning session
    mySession = tf.Session()
    set_session(mySession)
    #load the model
    myModel = load_model('saved_model.h5')
    myGraph = tf.get_default_graph()
    return (mySession,myModel,myGraph)

#create website 
app = Flask(__name__)
(mySession,myModel,myGraph) = load_model_from_file()
app.config['SECRET_KEY'] = 'super secret key'
app.config['SESSION'] = mySession
app.config['MODEL'] = myModel
app.config['GRAPH'] = myGraph    
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

#Try to allow only images
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#we are now defining the view for the top level page in our website
@app.route('/',methods=['GET','POST'])
def upload_file():
    #load the initial page
    if request.method == 'GET':
        return render_template('index.html',mybreeds=breeds,mysample_image_path=sample_image_path)
    else: # if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser may also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # If it doesn't look like an image file
        if not allowed_file(file.filename):
            flash('I only accept files of type'+str(ALLOWED_EXTENSIONS))
            return redirect(request.url)
        #When the user uploads a file with good parameters
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))
        
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    test_image = image.load_img(UPLOAD_FOLDER+"/"+filename,target_size=(150,150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    mySession = app.config['SESSION']
    myModel = app.config['MODEL']
    myGraph = app.config['GRAPH']
    with myGraph.as_default():
        set_session(mySession)
        result = myModel.predict(test_image)
        image_src = "/"+UPLOAD_FOLDER +"/"+filename
        if result[0] < 0.5 :
            answer = "<div class='col text-center'><img width='150' height='150' src='"+image_src+"' class='img-thumbnail' /><h4>guess:"+breeds+" "+str(result[0])+"</h4></div><div class='col'></div><div class='w-100'></div>"     
        results.append(answer)
        return render_template('index.html',mybreeds=breeds,mysample_image_path=sample_image_path,len=len(results),results=results)



# we will create a list of the results so far classified.
results = []

if __name__ == "__main__":
    app.run()

# launch the website
#main()
