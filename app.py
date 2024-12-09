# app.py

from flask import Flask, request, render_template
import os
from model.deepfake_detection import detect_deepfake

app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file", 400
    
    # Save the uploaded file
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(video_path)

    # Path to your pre-trained model
    model_path = r'G:\dpfkd\dpfkd\model\dpfkd\dpfkd\deepfake_model.h5'  # Update this path

    # Detect deepfake
    result = detect_deepfake(video_path, model_path)

    return {
        'is_fake': result['is_fake'],
        'confidence': result['confidence']
    }

if __name__ == "__main__":
    app.run(debug=True)
