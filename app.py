from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from model.deepfake_detection import detect_deepfake

app = Flask(_name_)
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
    video_file.save(filepath)

    # Detect deepfake
    result = detect_deepfake(filepath)
    
    return jsonify({
        'is_fake': result['is_fake'],
        'accuracy': result['accuracy']
    })

if _name_ == '_main_':
    app.run(debug=True)