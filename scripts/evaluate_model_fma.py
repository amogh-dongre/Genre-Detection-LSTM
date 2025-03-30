#!/usr/bin/env python
"""
Evaluate LSTM model on FMA Medium dataset and save evaluation metrics.
Memory-optimized with better error handling to prevent segmentation faults.
"""
import os
import argparse
import json
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import gc
import traceback
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)

def load_fma_metadata(metadata_path):
    """Load FMA metadata including track IDs and genre information"""
    try:
        logging.info("Loading FMA metadata...")
        
        # Load tracks metadata - handling both CSV and parquet formats
        tracks_file = os.path.join(metadata_path, 'tracks.csv')
        if not os.path.exists(tracks_file):
            tracks_file = os.path.join(metadata_path, 'tracks.parquet')
            if not os.path.exists(tracks_file):
                raise FileNotFoundError(f"Could not find tracks metadata file at {metadata_path}")
        
        # Load the appropriate format
        if tracks_file.endswith('.csv'):
            tracks = pd.read_csv(tracks_file, index_col=0, header=[0, 1], low_memory=False)
        else:
            tracks = pd.read_parquet(tracks_file)
        
        # Filter for medium subset
        medium_filter = tracks.index[tracks[('set', 'subset')] == 'medium']
        
        # Get the genre information for the medium subset
        genre_top = tracks.loc[medium_filter, ('track', 'genre_top')]
        
        # Extract unique genres and sort them
        genres = sorted(genre_top.dropna().unique())
        
        # Create a mapping from track_id to genre
        track_to_genre = genre_top.to_dict()
        
        logging.info(f"Loaded metadata for {len(track_to_genre)} tracks with {len(genres)} genres")
        return track_to_genre, genres
    
    except Exception as e:
        logging.error(f"Error loading metadata: {e}")
        logging.error(traceback.format_exc())
        raise

def extract_features(audio_file, sr=22050, duration=30):
    """Extract features from an audio file for model prediction with memory optimization"""
    try:
        # Load audio file with a lower duration to save memory
        y, sr = librosa.load(audio_file, sr=sr, duration=duration)
        
        # Basic feature set to reduce memory usage
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Combine features using mean only to reduce dimensionality
        features = np.concatenate([
            np.mean(mfccs, axis=1),
            np.mean(spectral_centroid),
            np.mean(chroma, axis=1)
        ])
        
        # Clean up to free memory
        del y, mfccs, spectral_centroid, chroma
        gc.collect()
        
        return features
        
    except Exception as e:
        logging.error(f"Error extracting features from {audio_file}: {e}")
        return None

def batch_process_files(file_list, track_to_genre, genres, batch_size=100):
    """Process files in batches to avoid memory issues"""
    all_features = []
    all_labels = []
    all_files = []
    all_track_ids = []
    
    for i in range(0, len(file_list), batch_size):
        batch = file_list[i:i+batch_size]
        logging.info(f"Processing batch {i//batch_size + 1}/{(len(file_list)//batch_size) + 1} ({len(batch)} files)")
        
        batch_features = []
        batch_labels = []
        batch_files = []
        batch_track_ids = []
        
        for file_path in tqdm(batch):
            try:
                # Extract track ID from filename (remove .mp3)
                filename = os.path.basename(file_path)
                track_id = int(os.path.splitext(filename)[0])
                
                # Check if track is in medium subset and has genre info
                if track_id in track_to_genre and track_to_genre[track_id] in genres:
                    features = extract_features(file_path)
                    
                    if features is not None:
                        genre = track_to_genre[track_id]
                        genre_idx = genres.index(genre)
                        
                        batch_features.append(features)
                        batch_labels.append(genre_idx)
                        batch_files.append(file_path)
                        batch_track_ids.append(track_id)
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
        
        # Add batch results to main lists
        all_features.extend(batch_features)
        all_labels.extend(batch_labels)
        all_files.extend(batch_files)
        all_track_ids.extend(batch_track_ids)
        
        # Free memory
        del batch_features
        gc.collect()
        
    return np.array(all_features), all_labels, all_files, all_track_ids

def find_audio_files(audio_path):
    """Find all MP3 files in the directory recursively"""
    all_files = []
    for root, dirs, files in os.walk(audio_path):
        for file in files:
            if file.endswith('.mp3'):
                all_files.append(os.path.join(root, file))
    return all_files

def evaluate_model(model_path, audio_path, metadata_path=None, max_files=None):
    """Evaluate model on FMA medium dataset with memory optimization"""
    try:
        # Create output directory
        os.makedirs('evaluation', exist_ok=True)
        
        # Load the model with memory optimization
        logging.info(f"Loading model from {model_path}")
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Determine genres
        if metadata_path:
            track_to_genre, genres = load_fma_metadata(metadata_path)
        else:
            # If no metadata, we'll use the model's output layer size to determine number of genres
            # and assign generic names
            logging.warning("No metadata provided. Using generic genre names.")
            num_genres = model.output_shape[1]
            genres = [f"genre_{i}" for i in range(num_genres)]
            track_to_genre = {}
        
        # Find audio files
        logging.info(f"Finding audio files in {audio_path}")
        all_files = find_audio_files(audio_path)
        logging.info(f"Found {len(all_files)} audio files")
        
        if max_files and max_files < len(all_files):
            logging.info(f"Limiting to {max_files} files for evaluation")
            all_files = all_files[:max_files]
        
        # Process files in batches
        if metadata_path:
            X_test, y_true, file_names, track_ids = batch_process_files(all_files, track_to_genre, genres)
        else:
            # If no metadata, we need to make predictions without ground truth
            logging.info("Extracting features without ground truth")
            X_test = []
            file_names = []
            track_ids = []
            
            for file_path in tqdm(all_files[:max_files] if max_files else all_files):
                try:
                    filename = os.path.basename(file_path)
                    track_id = os.path.splitext(filename)[0]
                    
                    features = extract_features(file_path)
                    if features is not None:
                        X_test.append(features)
                        file_names.append(file_path)
                        track_ids.append(track_id)
                except Exception as e:
                    logging.error(f"Error processing {file_path}: {e}")
            
            X_test = np.array(X_test)
            y_true = None
        
        # Check if we have data to evaluate
        if len(X_test) == 0:
            logging.error("No valid features extracted. Cannot evaluate model.")
            return None
        
        logging.info(f"Extracted features for {len(X_test)} files")
        
        # Get model's expected input shape
        input_shape = model.input_shape
        logging.info(f"Model input shape: {input_shape}")
        
        # Reshape for model input
        try:
            if len(input_shape) == 3:  # (None, timesteps, features)
                n_timesteps = input_shape[1]
                if n_timesteps is None:
                    n_timesteps = 1
                
                # Dynamic reshape based on model's expected input
                X_test = X_test.reshape(X_test.shape[0], n_timesteps, -1)
                logging.info(f"Reshaped input to {X_test.shape}")
            elif len(input_shape) == 2:  # (None, features)
                # No reshaping needed for non-sequential models
                pass
        except Exception as e:
            logging.error(f"Error reshaping data: {e}")
            logging.error(f"Current shape: {X_test.shape}, Target shape: {input_shape}")
            # Fallback to simple reshape
            X_test = X_test.reshape(X_test.shape[0], 1, -1)
            logging.info(f"Fallback reshape to {X_test.shape}")
        
        # Make predictions
        logging.info("Running predictions")
        y_pred_proba = model.predict(X_test, batch_size=32)  # Use batch_size to manage memory
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Create a detailed results dataframe
        results_df = pd.DataFrame({
            'track_id': track_ids,
            'filename': file_names
        })
        
        # Add predicted genre
        results_df['predicted_genre'] = [genres[i] if i < len(genres) else f"unknown_{i}" for i in y_pred]
        
        # Add true genre and correctness if we have ground truth
        if y_true:
            results_df['true_genre'] = [genres[i] for i in y_true]
            results_df['correct'] = [y_true[i] == y_pred[i] for i in range(len(y_true))]
        
        # Add probability columns for each genre
        for i, genre in enumerate(genres):
            if i < y_pred_proba.shape[1]:
                results_df[f'prob_{genre}'] = y_pred_proba[:, i]
        
        # Save results to CSV
        results_df.to_csv('evaluation/prediction_results.csv', index=False)
        logging.info("Saved prediction results to evaluation/prediction_results.csv")
        
        # If we have ground truth, generate classification report and confusion matrix
        if y_true:
            report = classification_report(y_true, y_pred, target_names=genres, output_dict=True)
            
            # Save classification report as JSON
            with open('evaluation/classification_report.json', 'w') as f:
                json.dump(report, f, indent=4)
            
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Plot confusion matrix
            plt.figure(figsize=(14, 12))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=genres, yticklabels=genres)
            plt.title('Confusion Matrix - FMA Dataset')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('evaluation/confusion_matrix.png')
            
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
                'genres': genres,
                'confusion_matrix': cm.tolist()
            }
            
            with open('evaluation/summary_metrics.json', 'w') as f:
                json.dump(summary, f, indent=4)
            
            logging.info(f"Evaluation complete. Accuracy: {accuracy:.4f}")
            
            return summary
        else:
            logging.info("Evaluation complete (without ground truth)")
            return {"total_samples": len(X_test)}
        
    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        logging.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on FMA dataset')
    parser.add_argument('--model', required=True, help='Path to model file (.keras or .h5)')
    parser.add_argument('--audio', required=True, help='Path to FMA audio files directory')
    parser.add_argument('--metadata', required=False, help='Path to FMA metadata directory')
    parser.add_argument('--max_files', type=int, default=None, help='Maximum number of files to evaluate')
    args = parser.parse_args()
    
    evaluate_model(args.model, args.audio, args.metadata, args.max_files)
