from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
#from tensorflow.keras.preprocessing import image
import numpy as np
import joblib
import cv2 
from skimage.feature import hog 

# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'  # Ensure this directory exists and is writable
app.secret_key = 'Sannidhi123'  # Use a strong, secure key for production

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Load the pre-trained ML model
model_path = '/home/abhayraj/Projects/personalProjects/leafDataAnalysis/code/my_flask_app/leaf_svm_model_c.pkl'  # Update to the correct path
label_encoder_path = '/home/abhayraj/Projects/personalProjects/leafDataAnalysis/code/my_flask_app/label_encoder_c.pkl'  # Update to the correct path for the label encoder


try:
    model = joblib.load(model_path)  # Corrected joblib model loading
    label_encoder = joblib.load(label_encoder_path)  # Load the label encoder
except Exception as e:
    print(f"Error loading model or label encoder: {e}")
    model = None  # Set model to None if loading fails
    label_encoder = None  # Set label encoder to None if loading fails

# Function to check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to extract HOG features
def extract_hog_features(image):
    hog_features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    return hog_features

# Function to predict image category
def predict_image(file_path):
    if model is None or label_encoder is None:
        print("Model or label encoder not loaded.")
        return None
    try:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        img = cv2.resize(img, (128, 128))  # Resize to match training images
        img = img.astype('float32') / 255.0  # Normalize pixel values
   
    # Extract HOG features
        hog_features = extract_hog_features(img).reshape(1, -1)  # Reshape for prediction
   
        predictions = model.predict(hog_features)
        prediction = model.predict(hog_features)
        print(f"Prediction: {prediction}")
        
        predicted_class = np.argmax(predictions[0])
        
        # Decode the numerical prediction back to the original label
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]# Get original label
        print(f"Predicted Label: {predicted_label}")
        

        return predicted_label
    except Exception as e:
        flash(f"Error in prediction: {e}")
        return None

# Route for the home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route for uploading and processing the image
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                # Ensure the upload directory exists
                if not os.path.exists(app.config['UPLOAD_FOLDER']):
                    os.makedirs(app.config['UPLOAD_FOLDER'])
                
                file.save(file_path)

                # Process and predict using the ML model
                prediction = predict_image(file_path)
                
                if prediction is not None:
                    return render_template('result.html', filename=filename, prediction=prediction)
                else:
                    flash("An error occurred while making the prediction. Please try again.")
            except Exception as e:
                flash(f"An error occurred during file upload or processing: {e}")
                print(f"Error during file upload or processing: {e}")  # Log the specific error
        else:
            flash("Invalid file type. Please upload an image file (png, jpg, jpeg, gif).")
        
    return redirect(url_for('index'))

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
