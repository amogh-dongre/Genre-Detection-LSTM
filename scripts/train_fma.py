import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf
from tqdm import tqdm
import random
from scipy.signal import resample
import ast
import warnings
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import functools
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Define constants
SAMPLE_RATE = 22050
DURATION = 30  # in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
NUM_SEGMENTS = 10  # Split each track into segments
SAMPLES_PER_SEGMENT = SAMPLES_PER_TRACK // NUM_SEGMENTS
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
MFCC_PER_SEGMENT = (SAMPLES_PER_SEGMENT // HOP_LENGTH) + 1

# FMA dataset paths - update these paths as needed
FMA_METADATA_PATH = "../data/fma_metadata"
FMA_AUDIO_PATH = "../data/fma_medium"  # or fma_medium, fma_large
TRACKS_FILE = os.path.join(FMA_METADATA_PATH, "tracks.csv")
FEATURES_CACHE = "../data/fma_features.npz"

# Set cache directory for librosa
os.environ['LIBROSA_CACHE_DIR'] = os.path.join(os.path.dirname(FEATURES_CACHE), 'librosa_cache')
os.makedirs(os.environ['LIBROSA_CACHE_DIR'], exist_ok=True)

def load_fma_metadata():
    """Load and prepare FMA metadata"""
    print("Loading FMA metadata...")
    
    # Load the tracks metadata
    tracks = pd.read_csv(TRACKS_FILE, index_col=0, header=[0, 1])
    
    # Select only tracks from medium dataset
    small = tracks['set', 'subset'] == 'medium'
    tracks = tracks[small]
    
    # Get genre information
    genres = tracks['track', 'genre_top']
    genres = genres.dropna()
    
    # Create a mapping from track ID to file path
    def get_audio_path(track_id):
        tid_str = '{:06d}'.format(track_id)
        return os.path.join(FMA_AUDIO_PATH, tid_str[:3], tid_str + '.mp3')
    
    # Create a dataframe with track_id, genre, and file_path
    data = pd.DataFrame({
        'track_id': genres.index,
        'genre': genres.values,
        'file_path': genres.index.map(get_audio_path)
    })
    
    # Verify files exist (with optimized approach using list comprehension)
    file_exists = [os.path.exists(path) for path in data['file_path']]
    data['file_exists'] = file_exists
    valid_data = data[data['file_exists']]
    
    print(f"Total valid tracks: {len(valid_data)}")
    print(f"Genre distribution:\n{valid_data['genre'].value_counts()}")
    
    return valid_data

def extract_features_optimized(file_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    """Extract MFCCs from music file and split into segments (optimized version)"""
    try:
        # Load audio file with faster resampling
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, 
                            res_type='kaiser_fast', dtype=np.float32)
        
        # Early return if audio is too short
        if len(y) < SAMPLES_PER_TRACK / 2:
            return None
        
        # Ensure consistent length (efficiently)
        if len(y) < SAMPLES_PER_TRACK:
            y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)))
        else:
            y = y[:SAMPLES_PER_TRACK]
        
        # Pre-allocate features array
        num_features = n_mfcc * 3 + 1 + 12  # MFCCs + deltas + spectral centroid + chroma
        features = np.zeros((num_segments, num_features))
        
        # Extract MFCCs for all segments at once
        for s in range(num_segments):
            start_sample = SAMPLES_PER_SEGMENT * s
            end_sample = start_sample + SAMPLES_PER_SEGMENT
            
            segment = y[start_sample:end_sample]
            
            # Combined feature extraction (reduced function calls)
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # Extract additional features (only what's necessary)
            spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length)
            chroma = librosa.feature.chroma_stft(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length)
            
            # Efficient feature combination
            feature_vector = np.concatenate([
                np.mean(mfcc, axis=1),
                np.mean(mfcc_delta, axis=1),
                np.mean(mfcc_delta2, axis=1),
                np.mean(spectral_centroid, axis=1),
                np.mean(chroma, axis=1)
            ])
            
            features[s] = feature_vector
            
        return features
    except Exception as e:
        # print(f"Error processing {file_path}: {e}")
        return None

def process_track(row):
    """Process a single track (for parallel execution)"""
    try:
        features = extract_features_optimized(row['file_path'])
        if features is not None:
            return features, [row['genre']] * len(features)
        return None, None
    except Exception as e:
        # print(f"Error in process_track for {row['file_path']}: {e}")
        return None, None

def load_data_parallel():
    """Load and process the FMA dataset using parallel processing"""
    # Check if cached features exist
    if os.path.exists(FEATURES_CACHE):
        print("Loading preprocessed features from cache...")
        data = np.load(FEATURES_CACHE, allow_pickle=True)
        return data['features'], data['labels']
    
    # Load metadata
    metadata = load_fma_metadata()
    
    all_features = []
    all_labels = []
    
    # Get the number of CPU cores, but limit to a reasonable number
    num_workers = min(os.cpu_count(), 16)  # Using 16 as a reasonable upper limit
    print(f"Using {num_workers} workers for parallel processing")
    
    # Create batches of tracks for processing
    metadata_records = metadata.to_dict('records')
    batch_size = 100  # Process in batches to avoid memory issues
    
    total_processed = 0
    
    # Process in batches
    for i in range(0, len(metadata_records), batch_size):
        batch = metadata_records[i:i+batch_size]
        batch_features = []
        batch_labels = []
        
        print(f"Processing batch {i//batch_size + 1}/{(len(metadata_records) + batch_size - 1)//batch_size}...")
        
        # Use ProcessPoolExecutor for CPU-bound tasks (feature extraction)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_track, row) for row in batch]
            
            # Process results as they complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tracks"):
                try:
                    features, labels = future.result()
                    if features is not None and labels is not None:
                        batch_features.extend(features)
                        batch_labels.extend(labels)
                except Exception as e:
                    print(f"Error processing track: {e}")
        
        # Accumulate batch results
        if batch_features:
            all_features.extend(batch_features)
            all_labels.extend(batch_labels)
            
        total_processed += len(batch)
        print(f"Processed {total_processed}/{len(metadata_records)} tracks, extracted {len(all_features)} feature segments")
        
        # Save intermediate results periodically
        if total_processed % 500 == 0 and all_features:
            np.savez(FEATURES_CACHE + f".partial_{total_processed}", 
                    features=np.array(all_features), 
                    labels=np.array(all_labels))
    
    # Convert to numpy arrays
    all_features = np.array(all_features) if all_features else np.array([])
    all_labels = np.array(all_labels) if all_labels else np.array([])
    
    # Cache the features
    os.makedirs(os.path.dirname(FEATURES_CACHE), exist_ok=True)
    np.savez(FEATURES_CACHE, features=all_features, labels=all_labels)
    
    return all_features, all_labels

def data_augmentation(features, labels):
    """Apply data augmentation techniques (optimized version)"""
    print("Applying data augmentation...")
    
    # Count samples per class
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    max_count = max(counts)
    
    # Pre-allocate arrays (more efficient than extending lists)
    total_augmented = len(features)
    for label in unique_labels:
        label_count = label_counts[label]
        augmentation_factor = max(1, int(max_count / label_count))
        # For each under-represented class, we'll add (augmentation_factor - 1) * label_count * 2 samples
        # (2 because we have 2 augmentation methods)
        total_augmented += (augmentation_factor - 1) * label_count * 2
    
    augmented_features = np.zeros((total_augmented, features.shape[1]))
    augmented_labels = np.empty(total_augmented, dtype=labels.dtype)
    
    # Add original data
    augmented_features[:len(features)] = features
    augmented_labels[:len(labels)] = labels
    
    # Track position in output arrays
    current_pos = len(features)
    
    # Efficient augmentation using vectorized operations where possible
    for label in unique_labels:
        # Get indices for this class
        indices = np.where(labels == label)[0]
        label_features = features[indices]
        
        # Calculate how many more samples we need for balance
        augmentation_factor = max(1, int(max_count / label_counts[label]))
        
        if augmentation_factor > 1:
            # For noise augmentation
            noise_factor = 0.05
            for i in range(augmentation_factor - 1):
                # 1. Add noise (vectorized operation)
                noise = noise_factor * np.random.normal(0, 1, label_features.shape)
                noisy_features = label_features + noise
                
                end_pos_noise = current_pos + len(label_features)
                augmented_features[current_pos:end_pos_noise] = noisy_features
                augmented_labels[current_pos:end_pos_noise] = label
                current_pos = end_pos_noise
                
                # 2. Frequency masking
                mask_features = label_features.copy()
                mask_size = int(0.1 * label_features.shape[1])
                
                # Apply different masks to each feature
                for j, feat in enumerate(mask_features):
                    start = np.random.randint(0, len(feat) - mask_size)
                    mask_features[j, start:start+mask_size] = 0
                
                end_pos_mask = current_pos + len(mask_features)
                augmented_features[current_pos:end_pos_mask] = mask_features
                augmented_labels[current_pos:end_pos_mask] = label
                current_pos = end_pos_mask
    
    # Trim in case we overestimated
    return augmented_features[:current_pos], augmented_labels[:current_pos]

def prepare_datasets(features, labels, test_size=0.2, validation_size=0.2):
    """Prepare train, validation, and test sets (optimized version)"""
    # Convert string labels to integers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Convert to categorical (one-hot encoding)
    categorical_labels = to_categorical(encoded_labels)
    
    # Split into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, categorical_labels, test_size=test_size, random_state=42, stratify=encoded_labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation_size, random_state=42
    )
    
    # Reshape for LSTM [samples, time_steps, features]
    # Do this efficiently by creating new arrays with the right shape
    X_train = np.expand_dims(X_train, axis=1)
    X_val = np.expand_dims(X_val, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder

def build_lstm_model(input_shape, num_classes):
    """Build an LSTM model for genre classification (unchanged)"""
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    """Main function to run the entire pipeline"""
    print("Starting music genre classification with LSTM for FMA dataset...")
    
    # Load and process data using parallel processing
    features, labels = load_data_parallel()
    print(f"Loaded dataset with {len(features)} samples across {len(np.unique(labels))} genres")
    
    # Apply data augmentation to handle small dataset and class imbalance
    augmented_features, augmented_labels = data_augmentation(features, labels)
    print(f"Original dataset size: {len(features)}")
    print(f"Augmented dataset size: {len(augmented_features)}")
    
    # Prepare datasets
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = prepare_datasets(
        augmented_features, augmented_labels
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(label_encoder.classes_)
    model = build_lstm_model(input_shape, num_classes)
    model.summary()
    
    # Set up callbacks
    os.makedirs('models', exist_ok=True)
    checkpoint = ModelCheckpoint(
        'models/best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model (with optimization options for TensorFlow)
    # Configure TensorFlow to use all available CPU threads
    config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=os.cpu_count(),
        inter_op_parallelism_threads=os.cpu_count(),
        allow_soft_placement=True,
        device_count={'CPU': os.cpu_count()}
    )
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
    
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,  # Increased batch size for faster training
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    
    # Plot results
    plot_history(history)
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save model and summary
    model.save('models/fma_classifier_lstm.keras')
    # save_model_summary function remains unchanged
    
    print("Model saved as 'models/fma_classifier_lstm.keras'")

if __name__ == "__main__":
    main()
