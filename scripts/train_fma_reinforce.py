import pandas as pd
import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Path configuration (update accordingly)
METADATA_PATH = '../data/fma_metadata/tracks.csv'  # From fma_metadata folder
AUDIO_BASE_PATH = '../data/fma_medium/'  # Main fma_medium folder
MODEL_PATH = '../model/genre_classification_reinforcement.keras'
LOG_DIR = '../logs/training/'  # Directory for TensorBoard logs

# Create directories if they don't exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Load metadata and filter for medium subset
print("Loading metadata...")
metadata = pd.read_csv(METADATA_PATH, index_col=0, header=[0, 1])
medium_tracks = metadata[metadata['set', 'subset'] == 'medium']

# Get genre labels (using top-level genres)
genre_labels = medium_tracks['track', 'genre_top'].values
unique_genres = pd.unique(genre_labels)
genre_to_id = {genre: idx for idx, genre in enumerate(unique_genres)}
id_to_genre = {idx: genre for genre, idx in genre_to_id.items()}
print(f"Found {len(unique_genres)} genres: {', '.join(unique_genres)}")

# Build file paths by checking all subdirectories
print("Building file paths...")
file_paths = []
track_ids = []  # Added to track corresponding track IDs
for root, dirs, files in os.walk(AUDIO_BASE_PATH):
    for file in files:
        if file.endswith(".mp3"):
            track_id = int(os.path.splitext(file)[0])
            if track_id in medium_tracks.index:
                file_paths.append(os.path.join(root, file))
                track_ids.append(track_id)

print(f"Found {len(file_paths)} valid audio files")

# Filter corresponding genre labels
y_filtered = []
filtered_file_paths = []
filtered_track_ids = []
for i, track_id in enumerate(track_ids):
    if track_id in medium_tracks.index:
        y_filtered.append(genre_to_id[medium_tracks.loc[track_id]['track', 'genre_top']])
        filtered_file_paths.append(file_paths[i])
        filtered_track_ids.append(track_id)

y = np.array(y_filtered)
file_paths = filtered_file_paths

# Print some dataset statistics
print(f"Final dataset size: {len(file_paths)} audio files")
genre_counts = np.bincount(y)
for genre_id, count in enumerate(genre_counts):
    print(f"  {id_to_genre[genre_id]}: {count} tracks")

# Audio augmentation functions
def pitch_shift(signal, sr):
    """Apply random pitch shifting"""
    steps = np.random.uniform(-2, 2)
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=steps)

def time_stretch(signal):
    """Apply random time stretching"""
    rate = np.random.uniform(0.9, 1.1)
    return librosa.effects.time_stretch(signal, rate=rate)

def add_noise(signal):
    """Add random noise"""
    noise_factor = np.random.uniform(0.001, 0.005)
    noise = np.random.randn(len(signal))
    return signal + noise_factor * noise

# Enhanced feature extraction function
def extract_features(file_path, augment=False, sample_rate=22050):
    try:
        signal, sr = librosa.load(file_path, sr=sample_rate, duration=30)  # Limit to 30 seconds
        
        # Apply augmentation during training with 50% chance
        if augment and random.random() > 0.5:
            aug_type = random.choice(['pitch', 'stretch', 'noise'])
            if aug_type == 'pitch':
                signal = pitch_shift(signal, sr)
            elif aug_type == 'stretch':
                signal = time_stretch(signal)
            else:
                signal = add_noise(signal)
        
        # Extract multiple features
        # MFCCs with first and second derivatives
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Spectral features
        chroma = librosa.feature.chroma_stft(y=signal, sr=sr, n_chroma=12)
        spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
        
        # Energy features
        rms = librosa.feature.rms(y=signal)
        
        # Normalize each feature independently
        features_list = [mfccs, mfcc_delta, mfcc_delta2, chroma, 
                        spectral_centroid, spectral_bandwidth, spectral_rolloff, rms]
        
        normalized_features = []
        for feat in features_list:
            feat_mean = np.mean(feat)
            feat_std = np.std(feat)
            if feat_std > 0:  # Avoid division by zero
                normalized_features.append((feat - feat_mean) / feat_std)
            else:
                normalized_features.append(feat - feat_mean)
        
        # Concatenate all features
        combined_features = np.concatenate(normalized_features, axis=0)
        
        return combined_features.T  # Shape: (timesteps, features)
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Process audio files with tqdm
print("Extracting features...")
features = []
labels = []
successful_paths = []  # Keep track of successfully processed files

for i, fp in enumerate(tqdm(file_paths, desc="Extracting Features")):
    feature = extract_features(fp)
    if feature is not None:
        features.append(feature)
        labels.append(y[i])
        successful_paths.append(fp)

print(f"Successfully processed {len(features)} out of {len(file_paths)} files")

# Split dataset (stratified by genre)
X_train_paths, X_val_paths, y_train, y_val = train_test_split(
    successful_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Training set: {len(X_train_paths)} tracks")
print(f"Validation set: {len(X_val_paths)} tracks")

# Data generator for on-the-fly feature extraction and augmentation
class AudioDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, batch_size=32, augment=False, max_timesteps=500):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment
        self.max_timesteps = max_timesteps
        self.indexes = np.arange(len(self.file_paths))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.file_paths) / self.batch_size))
    
    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Generate data
        X, y = self._generate_batch(indexes)
        return X, y
    
    def _generate_batch(self, indexes):
        # Initialize batch arrays
        batch_features = []
        batch_labels = []
        
        # Generate data for each index
        for i in indexes:
            # Extract features with potential augmentation
            features = extract_features(self.file_paths[i], augment=self.augment)
            
            # Handle None or empty features
            if features is None or len(features) == 0:
                continue
                
            # Handle timesteps
            if len(features) > self.max_timesteps:
                # Randomly select a segment if longer than max_timesteps
                start = np.random.randint(0, len(features) - self.max_timesteps)
                features = features[start:start + self.max_timesteps]
            else:
                # Pad with zeros if shorter
                padding = np.zeros((self.max_timesteps - len(features), features.shape[1]))
                features = np.vstack((features, padding))
            
            batch_features.append(features)
            batch_labels.append(self.labels[i])
        
        # Convert to numpy arrays
        if not batch_features:  # Handle empty batch
            # Return empty arrays with correct shapes
            feature_dim = extract_features(self.file_paths[0]).shape[1]
            return np.empty((0, self.max_timesteps, feature_dim)), np.empty((0,))
            
        return np.array(batch_features), np.array(batch_labels)
    
    def on_epoch_end(self):
        # Shuffle indexes after each epoch
        np.random.shuffle(self.indexes)

# Get feature dimensionality by extracting sample features
sample_features = extract_features(X_train_paths[0])
MAX_TIMESTEPS = 500  # Fixed sequence length
FEATURE_DIM = sample_features.shape[1]
print(f"Feature dimensionality: {FEATURE_DIM}")

# Create data generators
train_generator = AudioDataGenerator(
    X_train_paths, y_train, batch_size=32, augment=True, max_timesteps=MAX_TIMESTEPS
)
val_generator = AudioDataGenerator(
    X_val_paths, y_val, batch_size=32, augment=False, max_timesteps=MAX_TIMESTEPS
)

# Define callbacks
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir=LOG_DIR,
        histogram_freq=1
    )
]

# Define the model - Hybrid CNN-LSTM
def create_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # 1D Convolutional layers to extract features
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # LSTM layers for temporal features
    x = LSTM(128, return_sequences=True, dropout=0.3)(x)
    x = LSTM(64, dropout=0.3)(x)
    
    # Dense layers for classification
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Create and compile the model
model = create_model((MAX_TIMESTEPS, FEATURE_DIM), len(unique_genres))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Train the model
print("Training model...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=150,
    callbacks=callbacks,
    verbose=1
)

# Save the final model
model.save(MODEL_PATH.replace('.keras', '_final.keras'))

# Plot training history
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('../model/training_history.png')
plt.close()

# Additional feature: Create a simple inference function
def predict_genre(audio_file, model_path=MODEL_PATH):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Extract features from multiple segments
    features = extract_features(audio_file)
    
    if features is None or len(features) == 0:
        return "Error processing audio file"
    
    # Create segments with overlap for more robust prediction
    segments = []
    segment_length = 500
    hop_length = 250
    
    if len(features) >= segment_length:
        for i in range(0, len(features) - segment_length + 1, hop_length):
            segment = features[i:i + segment_length]
            segments.append(segment)
    else:
        # Pad if too short
        padding = np.zeros((segment_length - len(features), features.shape[1]))
        padded_features = np.vstack((features, padding))
        segments.append(padded_features)
    
    # Make predictions for each segment
    predictions = []
    for segment in segments:
        segment = np.expand_dims(segment, axis=0)
        pred = model.predict(segment, verbose=0)
        predictions.append(pred[0])
    
    # Average predictions across segments
    avg_prediction = np.mean(predictions, axis=0)
    predicted_class = np.argmax(avg_prediction)
    
    return {
        'genre': id_to_genre[predicted_class],
        'confidence': float(avg_prediction[predicted_class]),
        'all_probabilities': {id_to_genre[i]: float(avg_prediction[i]) for i in range(len(avg_prediction))}
    }
print("Training complete!")
print(f"Model saved to {MODEL_PATH}")
print(f"Training history saved to ../model/training_history.png")
print(f"Inference example script saved to ../model/inference_example.py")
