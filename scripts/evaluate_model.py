#!/usr/bin/env python
"""
Evaluate LSTM model on GTZAN test dataset and save evaluation metrics.
Used in CI/CD pipeline for automated model evaluation.
"""
import os
import argparse
import json
import numpy as np
import tensorflow as tf
import librosa
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define your genre labels
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

def extract_features(audio_file, sr=22050, duration=30):
    """Extract features from an audio file for model prediction"""
    # Load audio file with error handling
    try:
        y, sr = librosa.load(audio_file, sr=sr, duration=duration)
    except Exception as e:
        print(f"Error loading {audio_file}: {e}")
        return None
    
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
    
    return features

def evaluate_model(model_path, test_data_path):
    """Evaluate model on test dataset and generate metrics"""
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Create output directory
    os.makedirs('evaluation', exist_ok=True)
    
    # Process test files and get ground truth
    X_test = []
    y_true = []
    file_names = []
    
    for genre in genres:
        genre_dir = os.path.join(test_data_path, genre)
        if not os.path.exists(genre_dir):
            print(f"Warning: Directory {genre_dir} does not exist")
            continue
            
        for filename in os.listdir(genre_dir):
            if filename.endswith(('.wav', '.au')):
                file_path = os.path.join(genre_dir, filename)
                features = extract_features(file_path)
                
                if features is not None:
                    X_test.append(features)
                    y_true.append(genres.index(genre))
                    file_names.append(file_path)
    
    # Convert to numpy arrays
    X_test = np.array(X_test)
    
    # Reshape for LSTM input (samples, time steps, features)
    # This reshape depends on your model's expected input shape
    # Assuming the model expects a 3D input with a time dimension
    n_timesteps = 1  # Adjust based on your model
    n_features = X_test.shape[1]
    X_test = X_test.reshape(X_test.shape[0], n_timesteps, n_features)
    
    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=genres, output_dict=True)
    
    # Save classification report as JSON
    with open('evaluation/classification_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=genres, yticklabels=genres)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('evaluation/confusion_matrix.png')
    
    # Create a detailed results dataframe
    results_df = pd.DataFrame({
        'filename': file_names,
        'true_genre': [genres[i] for i in y_true],
        'predicted_genre': [genres[i] for i in y_pred],
        'correct': [y_true[i] == y_pred[i] for i in range(len(y_true))]
    })
    
    # Add probability columns for each genre
    for i, genre in enumerate(genres):
        results_df[f'prob_{genre}'] = y_pred_proba[:, i]
    
    # Save results to CSV
    results_df.to_csv('evaluation/prediction_results.csv', index=False)
    
    # Calculate overall metrics
    accuracy = report['accuracy']
    avg_precision = report['weighted avg']['precision']
    avg_recall = report['weighted avg']['recall']
    avg_f1 = report['weighted avg']['f1-score']
    
    # Save summary metrics
    summary = {
        'accuracy': accuracy,
        'weighted_precision': avg_precision,
        'weighted_recall': avg_recall,
        'weighted_f1': avg_f1,
        'total_samples': len(y_true),
        'confusion_matrix': cm.tolist()
    }
    
    with open('evaluation/summary_metrics.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Evaluation complete. Accuracy: {accuracy:.4f}")
    
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate GTZAN LSTM model')
    parser.add_argument('--model', required=True, help='Path to model file (.keras or .h5)')
    parser.add_argument('--data', required=True, help='Path to test data directory')
    args = parser.parse_args()
    
    evaluate_model(args.model, args.data)
