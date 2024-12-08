import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from sklearn.model_selection import train_test_split

def create_deepfake_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = create_deepfake_model()
    model.fit(X_train, y_train, epochs=10, validation_split=0.2)
    
    model.save('deepfake_model.h5')
    return model

def detect_deepfake(video_path):
    model = tf.keras.models.load_model('deepfake_model.h5')
    
    # Video preprocessing steps
    frames = extract_frames(video_path)
    predictions = model.predict(frames)
    
    is_fake = predictions.mean() > 0.5
    accuracy = predictions.mean() * 100
    
    return {
        'is_fake': is_fake,
        'accuracy': accuracy
    }