# model/deepfake_detection.py

import os
import cv2
import numpy as np
import tensorflow as tf

def load_model(model_path):
    """
    Load the pre-trained deepfake detection model.

    Args:
        model_path (str): The path to the model file.

    Returns:
        model: The loaded Keras model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file does not exist: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_video(video_path, target_size=(224, 224)):
    """
    Preprocess the video for analysis.

    Args:
        video_path (str): The path to the video file.
        target_size (tuple): The target size for the frames.

    Returns:
        np.array: Array of preprocessed frames.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Resize frame to target size
        frame = cv2.resize(frame, target_size)
        frames.append(frame)

    cap.release()
    return np.array(frames)

def detect_deepfake(video_path, model_path):
    """
    Analyze the video to determine if it is a deepfake.

    Args:
        video_path (str): The path to the video file.
        model_path (str): The path to the pre-trained model.

    Returns:
        dict: A dictionary containing the detection results.
    """
    # Load the model
    model = load_model(model_path)

    # Preprocess the video
    frames = preprocess_video(video_path)

    # Normalize the frames
    frames = frames / 255.0  # Scale pixel values to [0, 1]

    # Make predictions
    predictions = model.predict(frames)
    
    # Average the predictions
    average_prediction = np.mean(predictions)

    # Determine if the video is a deepfake based on a threshold
    is_fake = average_prediction > 0.5  # Assuming 0.5 is the threshold for binary classification

    return {
        'is_fake': is_fake,
        'confidence': average_prediction
    }