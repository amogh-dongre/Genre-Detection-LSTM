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
import warnings
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

# Path to GTZAN dataset - adjust as needed
DATASET_PATH = "data/Data/genres_original"
# CSV_PATH = "data/features.csv"

def extract_features(file_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    """Extract MFCCs from music file and split into segments"""
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Ensure consistent length
        if len(y) < SAMPLES_PER_TRACK:
            y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)))
        
        features = []
        labels = []
        
        # Extract genre from file path
        genre = file_path.split('/')[-2]
        
        # Process all segments of the audio file
        for s in range(num_segments):
            start_sample = SAMPLES_PER_SEGMENT * s
            end_sample = start_sample + SAMPLES_PER_SEGMENT
            
            # Extract MFCCs
            mfcc = librosa.feature.mfcc(y=y[start_sample:end_sample], 
                                        sr=sr, 
                                        n_mfcc=n_mfcc, 
                                        n_fft=n_fft, 
                                        hop_length=hop_length)
            # Transpose to get time series format (time_steps, features)
            mfcc = mfcc.T
            
            features.append(mfcc)
            labels.append(genre)
            
        return features, labels
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

def load_data():
    """Load and process the GTZAN dataset"""
    features = []
    labels = []
    
    # Check if preprocessed features exist
    # if os.path.exists(CSV_PATH):
    #     print("Loading preprocessed features...")
    #     data = pd.read_csv(CSV_PATH)
    #     return data
    
    print("Extracting features from audio files...")
    # Iterate through all genres
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in tqdm(files, desc=f"Processing {os.path.basename(root)}"):
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                segment_features, segment_labels = extract_features(file_path)
                
                if segment_features:
                    features.extend(segment_features)
                    labels.extend(segment_labels)
    
    # Convert features and labels to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    # Check if we have any features
    if not len(features):
        raise Exception("No features extracted. Check dataset path.")
        
    return features, labels

def data_augmentation(features, labels):
    """Apply data augmentation techniques to increase dataset size"""
    print("Applying data augmentation...")
    augmented_features = []
    augmented_labels = []
    
    # Add original data
    augmented_features.extend(features)
    augmented_labels.extend(labels)
    
    for i, feature in enumerate(tqdm(features, desc="Augmenting data")):
        # Time stretching/pitch shifting (simulated by resampling)
        stretched = resample(feature, int(feature.shape[0] * 0.9))
        if stretched.shape[0] > 10:  # Ensure we have enough time steps
            stretched = stretched[:features[0].shape[0]]  # Match original shape
            if stretched.shape[0] < features[0].shape[0]:
                # Pad if needed
                stretched = np.pad(stretched, ((0, features[0].shape[0] - stretched.shape[0]), (0, 0)))
            augmented_features.append(stretched)
            augmented_labels.append(labels[i])
        
        # Add noise
        noise_factor = 0.05
        noisy = feature + noise_factor * np.random.normal(0, 1, feature.shape)
        augmented_features.append(noisy)
        augmented_labels.append(labels[i])
        
        # Time shifting (roll the array)
        shift = np.roll(feature, int(feature.shape[0] * 0.2), axis=0)
        augmented_features.append(shift)
        augmented_labels.append(labels[i])
    
    return np.array(augmented_features), np.array(augmented_labels)

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
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder

def build_lstm_model(input_shape, num_classes):
    """Build an LSTM model for genre classification"""
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_history(history):
    """Plot training and validation accuracy/loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def main():
    """Main function to run the entire pipeline"""
    print("Starting music genre classification with LSTM...")
    
    # Load and process data
    features, labels = load_data()
    
    # Apply data augmentation to handle small dataset
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
    checkpoint = ModelCheckpoint(
        'best_model.keras',
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
        epochs=50,
        batch_size=32,
        callbacks=[checkpoint, early_stop]
    )
    
    # Plot results
    plot_history(history)
    
    # Evaluate on test set
    test_accuracy = model.evaluate(X_test, y_test)[1]
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save class mapping
    mapping = {i: genre for i, genre in enumerate(label_encoder.classes_)}
    print("Genre mapping:", mapping)
    
    # Save model
    model.save('genre_classifier_lstm.keras')
    print("Model saved as 'genre_classifier_lstm.keras'")

if __name__ == "__main__":
    main()   
