from flask import Flask, request, jsonify, render_template  # Added render_template import
from werkzeug.utils import secure_filename
import os

# Ensure the correct import path for detect_deepfake
try:
    from model.deepfake_detection import detect_deepfake
except ImportError as e:
    print(f"Error importing detect_deepfake: {e}")

app = Flask(__name__)  # Corrected _name_ to __name__
app.config['UPLOAD_FOLDER'] = 'uploads/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'})
    
    video_file = request.files['video']
    filename = secure_filename(video_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    video_file.save(filepath)

    # Detect deepfake
    result = detect_deepfake(filepath)
    
    return jsonify({
        'is_fake': result['is_fake'],
        'accuracy': result['accuracy']
    })

if __name__ == '__main__':  # Corrected _name_ to __name__
    app.run(debug=True)
