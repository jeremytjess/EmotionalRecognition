from flask import render_template, redirect, url_for, request, send_from_directory, flash
from app import app
import os
from werkzeug import secure_filename
from app import predictor 
from PIL import Image

@app.route('/<filename>')
def get_file(filename):
    """ Takes in filename and uploads the file to the 
        static directory
    """
    return send_from_directory('static/img', filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """ This runs when a Get or POST request is sent to our 
        website at the / endpoint renders our displayResult.html
        file if a POST request is sent with an image of type jpg,
        else renders index.html
    """
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_to=(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file.save(save_to)
            pred_class=predictor.model_predict(save_to, '/home/ubuntu/cs121/app')
            return render_template('displayResult.html', filename=filename, prediction=pred_class)    
        else:
            return render_template('error.html')
    return render_template('index.html')

@app.route('/about')
def about():
    """ when requests are sent to the /about endpoint this 
        renders displayFile.html
    """
    return render_template('displayFile.html')

# allowed image types
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
app.config['ALLOWED_EXTENSIONS']=ALLOWED_EXTENSIONS

def allowed_file(filename):
    """ is file allowed to be uploaded?
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']
