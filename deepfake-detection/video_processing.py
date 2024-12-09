import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

class DeepfakeDetector:
    def _init_(self, model_path):
        # Load the pre-trained model
        self.model = load_model(model_path)
        self.input_size = (224, 224)  # Input size for the model

    def preprocess_frame(self, frame):
        # Resize and normalize the frame
        frame = cv2.resize(frame, self.input_size)
        frame = frame.astype('float32') / 255.0  # Normalize to [0, 1]
        return np.expand_dims(frame, axis=0)  # Add batch dimension

    def predict_frame(self, frame):
        # Preprocess the frame and make a prediction
        processed_frame = self.preprocess_frame(frame)
        prediction = self.model.predict(processed_frame)
        return prediction[0][0]  # Return the probability

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        real_count = 0
        fake_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Predict if the frame is real or fake
            prediction = self.predict_frame(frame)
            frame_count += 1

            # Threshold for classification (0.5 for binary classification)
            if prediction >= 0.5:
                fake_count += 1
                label = "Fake"
            else:
                real_count += 1
                label = "Real"

            # Display the result on the frame
            cv2.putText(frame, f"{label}: {prediction:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if label == "Real" else (0, 0, 255), 2)
            cv2.imshow('Frame', frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Print summary of results
        print(f"Total Frames: {frame_count}, Real Frames: {real_count}, Fake Frames: {fake_count}")

if __name__ == "__main__":
    # Path to the trained model
    model_path = 'D:/model/deepfake_model.h5'
    # Path to the video to be processed
    video_folder = 'D:/dataset/'

    # Process all videos in the specified folder
    for video_file in os.listdir(video_folder):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(video_folder, video_file)
            print(f"Processing video: {video_path}")
            detector = DeepfakeDetector(model_path)
            detector.process_video(video_path)
