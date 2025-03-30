
import os
import json
from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
import tensorflow as tf

app = Flask(__name__, template_folder='templates')

# Load model and genre mapping
try:
    model_path = os.environ.get('MODEL_PATH', '../model/genre_classification_fma.keras')
    model = tf.keras.models.load_model(model_path)
    
    # Load updated genre mapping for FMA Medium
    with open('./genre_mapping.json', 'r') as f:
        mapping = json.load(f)
    genres = [mapping[str(i)] for i in range(len(mapping))]
except Exception as e:
    print(f"Error loading model: {e}")
    genres = []
    model = None

def extract_features(audio_file, sr=22050, duration=30, offset=0):
    """Extract features from an audio file for model prediction"""
    y, sr = librosa.load(audio_file, sr=sr, duration=duration, offset=offset)
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # Increased MFCC components
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    
    features = np.concatenate([
        np.mean(mfccs, axis=1),
        np.mean(mfcc_delta, axis=1),
        np.mean(mfcc_delta2, axis=1),
        np.mean(chroma, axis=1),
        np.mean(mel, axis=1),
        np.mean(contrast, axis=1),
        np.mean(tonnetz, axis=1)
    ])
    
    features = features.reshape(1, 1, -1)
    
    return features

@app.route('/')
def home():
    return render_template('frontend.html')

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    temp_path = 'temp_audio.wav'
    file.save(temp_path)
    
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Please check server logs.'})
        
        features = extract_features(temp_path)
        prediction = model.predict(features)
        genre_index = np.argmax(prediction[0])
        genre = genres[genre_index] if genres else 'Unknown'
        confidence = float(prediction[0][genre_index])
        
        result = {
            'genre': genre,
            'confidence': confidence,
            'all_probs': {g: float(p) for g, p in zip(genres, prediction[0])}
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

