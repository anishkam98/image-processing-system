import os
from flask import render_template
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import keras.utils as image
from keras.models import load_model
from keras.backend import set_session
import tensorflow as tf



breeds ={  0:'cocker_spaniel',
            1: 'german_shepherd',
            2:'golden_retriever',
            3:'labrador_retriever',
            4: 'standard_poodle'
}


sample_image_path = "static/n02099601_308.jpg"

UPLOAD_FOLDER = '/static/uploads'

#Allowed files
ALLOWED_EXTENSIONS = {'png','jpg','jpeg','gif'}


def load_model_from_file():
    #set up the machine learning session
    mySession = tf.compat.v1.Session()
    set_session(mySession)
    tf.compat.v1.disable_eager_execution()
    #load the model
    model_path = os.path.dirname(__file__) + '//saved_model.h5'
    myModel = load_model(model_path)
    #myGraph = tf.get_default_graph()
    myGraph = tf.compat.v1.get_default_graph()
    #return (mySession,myModel,myGraph)
    return (mySession, myModel,myGraph)

#create website 
app = Flask(__name__, static_folder='./static')
(mySession, myModel,myGraph) = load_model_from_file()
app.config['SECRET_KEY'] = 'super secret key'
app.config['SESSION'] = mySession
app.config['MODEL'] = myModel
app.config['GRAPH'] = myGraph    
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Folder to store uploaded images
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}  # Allowed file extensions for upload
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

#Try to allow only images
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#we are now defining the view for the top level page in our website
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # Load the initial page
    if request.method == 'GET':
        return render_template('index.html', mybreeds=breeds)
    else:  # if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser may also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # If it doesn't look like an image file
        if not allowed_file(file.filename):
            flash('I only accept files of type ' + str(ALLOWED_EXTENSIONS))
            return redirect(request.url)
        # When the user uploads a file with good parameters
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Process the uploaded image and obtain the results
            test_image = image.load_img(file_path, target_size=(150, 150))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            with app.config['GRAPH'].as_default():
                set_session(app.config['SESSION'])
                result = app.config['MODEL'].predict(test_image)
                max_index = np.argmax(result)
                breed_id = max_index 
                confidence = result[0][max_index]
                breed_name = breeds.get(breed_id, "Unknown") 
                # Store the results in the 'results' variable
                results = [{'predicted_breed': breed_name, 'confidence': confidence}]
            # Pass the 'results' variable to the template
            return render_template('index.html', mybreeds=breeds, results=results)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    path = os.path.dirname(__file__) + app.config['UPLOAD_FOLDER'] + '//' +filename
    test_image = image.load_img(path,target_size=(150,150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    

    mySession = app.config['SESSION']
    myModel = app.config['MODEL']
    myGraph = app.config['GRAPH']
    with myGraph.as_default():
        set_session(mySession)
        result = myModel.predict(test_image)
        pred = np.argmax(result[0], axis=-1)
        if pred < 0.5 :
            answer = "<div class='col text-center'><img width='150' height='150' src='"+url_for('static', filename='/uploads/'+filename)+"' class='img-thumbnail' /><h4>Confidence: " +str(pred)+"</h4></div><div class='col'></div><div class='w-100'></div>"     
            results.append(answer)
        print(results)
        return render_template('index.html',mybreeds=breeds,mysample_image_path=sample_image_path,len=len(results),results=results)



# we will create a list of the results so far classified.
results = []


if __name__ == "__main__":
    app.debug = True
    app.run()

# launch the website
#main()