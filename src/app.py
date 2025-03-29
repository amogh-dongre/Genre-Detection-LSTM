import os
from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
import tensorflow as tf

app = Flask(__name__, template_folder='templates')

# Load your trained LSTM model
model = tf.keras.models.load_model('../model/gtzan_lstm_model.keras')

# Define your genre labels
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

def extract_features(audio_file, sr=22050, duration=30):
    """Extract features from an audio file for model prediction"""
    # Load audio file
    y, sr = librosa.load(audio_file, sr=sr, duration=duration)
    
    # Extract features (adjust based on your model's input requirements)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    
    # Combine features
    features = np.concatenate([
        np.mean(mfccs, axis=1),
        np.mean(chroma, axis=1),
        np.mean(mel, axis=1),
        np.mean(contrast, axis=1),
        np.mean(tonnetz, axis=1)
    ])
    
    # Reshape for LSTM input (time steps, features)
    # Adjust this based on your model's expected input shape
    features = features.reshape(1, -1, features.shape[0])
    
    return features

@app.route('/')
def home():
    return render_template('frontend.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Save the file temporarily
    temp_path = 'temp_audio.wav'
    file.save(temp_path)
    
    try:
        # Extract features
        features = extract_features(temp_path)
        
        # Make prediction
        prediction = model.predict(features)
        genre_index = np.argmax(prediction[0])
        genre = genres[genre_index]
        confidence = float(prediction[0][genre_index])
        
        # Return prediction
        result = {
            'genre': genre,
            'confidence': confidence,
            'all_probs': {g: float(p) for g, p in zip(genres, prediction[0])}
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})
    
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(host ='0.0.0.0' , port = 5000 , debug=True)
