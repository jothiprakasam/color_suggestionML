import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
import numpy as np
from colorthief import ColorThief

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained TensorFlow model
model = tf.keras.applications.ResNet50(weights='imagenet', include_top=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    try:
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        preds = model.predict(image)
        label = imagenet_utils.decode_predictions(preds)[0][0][1]
        
        return label
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Predict the image content
        image_label = predict_image(file_path)

        if image_label:
            # Get dominant color
            color_thief = ColorThief(file_path)
            dominant_color = color_thief.get_color(quality=1)

            # Dummy logic to suggest pant color (inverse of the dominant color)
            pant_color = [255 - c for c in dominant_color]

            return jsonify({
                'label': image_label,
                'dominant_color': dominant_color,
                'pant_color': pant_color
            })
        else:
            return jsonify({'error': 'Failed to process image'})

    return jsonify({'error': 'File not allowed'})

if __name__ == '__main__':
    app.run(debug=True)
