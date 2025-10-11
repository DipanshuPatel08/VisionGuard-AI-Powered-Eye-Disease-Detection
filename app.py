import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, url_for, redirect
from werkzeug.utils import secure_filename
from PIL import Image
import io

# Initialize the Flask application
app = Flask(__name__)

# --- Configuration ---
# Load the pre-trained Keras model
# The model includes the necessary resizing and rescaling layers.
try:
    MODEL_PATH = 'my_model.keras'
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the class names based on the training notebook
CLASS_NAMES = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

# Configure the upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Helper Functions ---
def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_for_prediction(image_bytes):
    """
    Prepares the image for the model.
    1. Reads the image from bytes.
    2. Converts it to a NumPy array.
    3. Expands dimensions to create a batch of 1.
    The model itself will handle resizing and rescaling.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    return img_array

def predict_disease(image_bytes):
    """
    Makes a prediction on the preprocessed image.
    Returns the predicted class name and the confidence score.
    """
    if model is None:
        return "Error", "Model not loaded. Please check the server logs."

    # Preprocess the image without manual resizing/rescaling
    processed_image = preprocess_image_for_prediction(image_bytes)
    
    # The model will now apply its internal resizing and rescaling layers
    predictions = model.predict(processed_image)
    
    # Get the class with the highest probability
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_index]
    
    # Get the confidence score
    confidence = round(100 * np.max(predictions[0]), 2)
    
    return predicted_class, confidence

# --- Routes ---
@app.route('/', methods=['GET'])
def home():
    """Renders the home page."""
    return render_template('index.html')

@app.route('/detector', methods=['GET', 'POST'])
def detector():
    """
    Handles the image upload and prediction.
    - GET: Renders the upload page.
    - POST: Processes the uploaded file, makes a prediction, and renders the result.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            image_bytes = file.read()
            # It's good practice to save the file after reading bytes
            with open(filepath, 'wb') as f:
                f.write(image_bytes)

            predicted_class, confidence = predict_disease(image_bytes)
            
            return render_template('result.html', 
                                   predicted_class=predicted_class.replace('_', ' ').title(), 
                                   confidence=confidence, 
                                   image_url=url_for('static', filename='uploads/' + filename))

    return render_template('detector.html')

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True)