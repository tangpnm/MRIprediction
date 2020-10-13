from flask import Flask, request, jsonify, render_template
from torch_utils import transforms_image, get_prediction
import matplotlib.image as mpimg
# import torch_utils

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'jpg'} 
def allowed_file(filename):
    # xxx.jpg -> it return TRUE if it look like ALLOWED_EXTENSIONS
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def predict():

    class_names = ['MildDemented', 'ModerateDemented',
                   'NonDemented', 'VeryMildDemented']
                   
    if request.method == 'GET':
        return render_template('home.html', title='home')
    if request.method == 'POST':
        file = request.files['image']
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        try:

            # 1 load image
            img_bytes = file.read()
            pic = mpimg.imread(file)

            # 2 convert image -> tensor
            tensor = transforms_image(pic)

            # 3 prediction
            prediction = get_prediction(tensor)
            className = class_names[prediction.item()]

            # 4 return the result to JSON
            # data = {'prediction': prediction.item(), 'class_name': str(className)}
            data = {'class_name': str(className)}
            return jsonify(data)
        except:
            return jsonify({'error': 'error during prediction'})

    return jsonify({'result': 1})

if __name__ == '__main__':
    app.run(debug=True)
