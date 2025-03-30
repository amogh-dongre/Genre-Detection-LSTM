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
import tensorflow as tf
import multiprocessing
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define constants
SAMPLE_RATE = 22050
DURATION = 30  # in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
NUM_SEGMENTS = 10  # Split each track into segments
SAMPLES_PER_SEGMENT = SAMPLES_PER_TRACK // NUM_SEGMENTS
N_FFT = 2048
HOP_LENGTH = 512

# FMA dataset paths - update these paths as needed
FMA_METADATA_PATH = "../data/fma_metadata"
FMA_AUDIO_PATH = "../data/fma_medium"
TRACKS_FILE = os.path.join(FMA_METADATA_PATH, "tracks.csv")
FEATURES_CACHE = "../data/fma_features.npz"
BATCH_CACHE_DIR = "../data/batch_cache"

# Create cache directories
os.makedirs(BATCH_CACHE_DIR, exist_ok=True)

def load_fma_metadata():
    """Load and prepare FMA metadata"""
    print("Loading FMA metadata...")
    
    # Load the tracks metadata - more efficient loading with usecols
    tracks = pd.read_csv(TRACKS_FILE, index_col=0, header=[0, 1], 
                         usecols=lambda x: x.startswith('track') or x.startswith('set'))
    
    # Select only tracks from medium dataset
    small = tracks['set', 'subset'] == 'medium'
    tracks = tracks[small]
    
    # Get genre information
    genres = tracks['track', 'genre_top']
    genres = genres.dropna()
    
    # Create a mapping from track ID to file path
    # FMA organizes files in a specific directory structure
    def get_audio_path(track_id):
        tid_str = '{:06d}'.format(track_id)
        return os.path.join(FMA_AUDIO_PATH, tid_str[:3], tid_str + '.mp3')
    
    # Create a dataframe with track_id, genre, and file_path
    data = pd.DataFrame({
        'track_id': genres.index,
        'genre': genres.values,
        'file_path': genres.index.map(get_audio_path)
    })
    
    # More efficient file existence check
    file_exists = Parallel(n_jobs=-1)(delayed(os.path.exists)(path) for path in tqdm(data['file_path'], desc="Checking files"))
    data['file_exists'] = file_exists
    valid_data = data[data['file_exists']]
    
    print(f"Total valid tracks: {len(valid_data)}")
    print(f"Genre distribution:\n{valid_data['genre'].value_counts()}")
    
    return valid_data

def extract_minimal_features(file_path):
    """Extract only essential features from music file - highly optimized"""
    try:
        # Load audio file with minimal processing
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, 
                            res_type='kaiser_fast', mono=True)
        
        # Early return if audio is too short
        if len(y) < SAMPLES_PER_TRACK / 2:
            return None
        
        # Ensure consistent length
        if len(y) < SAMPLES_PER_TRACK:
            y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)))
        else:
            y = y[:SAMPLES_PER_TRACK]
        
        features = []
        
        # Extract only MFCCs (fewer features)
        for s in range(NUM_SEGMENTS):
            start_sample = SAMPLES_PER_SEGMENT * s
            end_sample = start_sample + SAMPLES_PER_SEGMENT
            
            segment = y[start_sample:end_sample]
            
            # Use reduced number of MFCCs and skip delta features
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=HOP_LENGTH)
            
            # Take mean across time only (simplified feature vector)
            feature_vector = np.mean(mfcc, axis=1)
            
            features.append(feature_vector)
            
        return np.array(features)
    except Exception as e:
        return None

def process_batch(batch_files, batch_id, genres):
    """Process a batch of files and save the results to a cache file"""
    batch_cache_file = os.path.join(BATCH_CACHE_DIR, f"batch_{batch_id}.npz")
    
    # Skip if this batch has already been processed
    if os.path.exists(batch_cache_file):
        print(f"Batch {batch_id} already processed, skipping...")
        # Load and return the shapes for logging
        data = np.load(batch_cache_file, allow_pickle=True)
        return data['features'].shape[0], data['labels'].shape[0]
    
    batch_features = []
    batch_labels = []
    
    # Process each file in the batch
    for file_path, genre in tqdm(zip(batch_files['file_path'], batch_files['genre']), 
                                total=len(batch_files), desc=f"Batch {batch_id}"):
        features = extract_minimal_features(file_path)
        
        if features is not None:
            batch_features.extend(features)
            batch_labels.extend([genre] * len(features))
    
    # Skip saving if no features were extracted
    if not batch_features:
        print(f"No features extracted for batch {batch_id}")
        return 0, 0
    
    # Convert to numpy arrays
    batch_features = np.array(batch_features)
    batch_labels = np.array(batch_labels)
    
    # Save batch results
    np.savez(batch_cache_file, features=batch_features, labels=batch_labels)
    
    return batch_features.shape[0], batch_labels.shape[0]

def load_data_batched():
    """Load and process the FMA dataset using optimized batched approach"""
    # Check if final cached features exist
    if os.path.exists(FEATURES_CACHE):
        print("Loading preprocessed features from cache...")
        data = np.load(FEATURES_CACHE, allow_pickle=True)
        return data['features'], data['labels']
    
    # Load metadata
    metadata = load_fma_metadata()
    
    # Process in smaller batches with file-based caching
    batch_size = 20  # Smaller batch size for more frequent progress
    num_batches = (len(metadata) + batch_size - 1) // batch_size
    
    # Process each batch
    total_features = 0
    total_labels = 0
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(metadata))
        batch_files = metadata.iloc[start_idx:end_idx]
        
        # Process this batch
        features_count, labels_count = process_batch(batch_files, i, metadata['genre'])
        
        total_features += features_count
        total_labels += labels_count
        
        print(f"Completed batch {i+1}/{num_batches}. Total features so far: {total_features}")
    
    # Combine all batch results
    print("Combining all batch results...")
    all_features = []
    all_labels = []
    
    for i in range(num_batches):
        batch_cache_file = os.path.join(BATCH_CACHE_DIR, f"batch_{i}.npz")
        if os.path.exists(batch_cache_file):
            batch_data = np.load(batch_cache_file, allow_pickle=True)
            if batch_data['features'].size > 0:  # Only add non-empty batches
                all_features.append(batch_data['features'])
                all_labels.append(batch_data['labels'])
    
    # Combine arrays
    if all_features:
        all_features = np.vstack(all_features)
        all_labels = np.concatenate(all_labels)
        
        # Cache the final combined features
        np.savez(FEATURES_CACHE, features=all_features, labels=all_labels)
        
        return all_features, all_labels
    else:
        raise ValueError("No features were successfully extracted from any audio files.")

def simplified_augmentation(features, labels):
    """Apply simplified data augmentation techniques"""
    print("Applying simplified data augmentation...")
    
    # Count samples per class
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    max_count = max(counts)
    
    # Start with original data
    augmented_features = features.copy()
    augmented_labels = labels.copy()
    
    # Only augment the classes that have fewer than 80% of the max count
    for label in unique_labels:
        if label_counts[label] < max_count * 0.8:
            # Get indices for this class
            indices = np.where(labels == label)[0]
            
            # Add noise to samples (just once per sample)
            noise_factor = 0.05
            noise = noise_factor * np.random.normal(0, 1, (len(indices), features.shape[1]))
            noisy_features = features[indices] + noise
            
            # Add augmented data
            augmented_features = np.vstack([augmented_features, noisy_features])
            augmented_labels = np.append(augmented_labels, [label] * len(indices))
    
    return augmented_features, augmented_labels

def prepare_datasets(features, labels, test_size=0.2, validation_size=0.2):
    """Prepare train, validation, and test sets"""
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
    X_train = np.expand_dims(X_train, axis=1)
    X_val = np.expand_dims(X_val, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder

def build_simplified_model(input_shape, num_classes):
    """Build a simpler model for faster training"""
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    """Main function to run the entire pipeline"""
    print("Starting optimized music genre classification for FMA dataset...")
    
    # Configure process-based parallelism
    n_jobs = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
    print(f"Using {n_jobs} CPU cores for parallel processing")
    
    # Load and process data with efficient batching
    features, labels = load_data_batched()
    print(f"Loaded dataset with {len(features)} samples across {len(np.unique(labels))} genres")
    
    # Apply simplified data augmentation
    augmented_features, augmented_labels = simplified_augmentation(features, labels)
    print(f"Original dataset size: {len(features)}")
    print(f"Augmented dataset size: {len(augmented_features)}")
    
    # Prepare datasets
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = prepare_datasets(
        augmented_features, augmented_labels
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Build model (simplified)
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(label_encoder.classes_)
    model = build_simplified_model(input_shape, num_classes)
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
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,  # Reduced from 100
        batch_size=64,
        callbacks=[checkpoint, early_stop]
    )
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save model
    model.save('models/fma_classifier_simplified.keras')
    print("Model saved as 'models/fma_classifier_simplified.keras'")

if __name__ == "__main__":
    main()
