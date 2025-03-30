import pandas as pd
import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

# Path configuration (update accordingly)
METADATA_PATH = '../data/fma_metadata/tracks.csv'  # From fma_metadata folder
AUDIO_BASE_PATH = '../data/fma_medium/'  # Main fma_medium folder

# Load metadata and filter for medium subset
metadata = pd.read_csv(METADATA_PATH, index_col=0, header=[0, 1])
medium_tracks = metadata[metadata['set', 'subset'] == 'medium']

# Get genre labels (using top-level genres)
genre_labels = medium_tracks['track', 'genre_top'].values
unique_genres = pd.unique(genre_labels)
genre_to_id = {genre: idx for idx, genre in enumerate(unique_genres)}
y = np.array([genre_to_id[g] for g in genre_labels])

# Build file paths by checking all subdirectories
file_paths = []
track_ids = []  # Added to track corresponding track IDs
for root, dirs, files in os.walk(AUDIO_BASE_PATH):
    for file in files:
        if file.endswith(".mp3"):
            track_id = os.path.splitext(file)[0]
            if int(track_id) in medium_tracks.index: # changed to int
                file_paths.append(os.path.join(root, file))
                track_ids.append(int(track_id)) # added track_ids to correspond to file_paths

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

# Audio preprocessing function
def extract_features(file_path, sample_rate=22050, n_mfcc=13):
    try:
        signal, _ = librosa.load(file_path, sr=sample_rate)
        mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc)
        return mfccs.T  # Shape: (timesteps, features)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None  # Return None for problematic files

# Process audio files with tqdm
features = []
for fp in tqdm(file_paths, desc="Extracting Features"):
    features.append(extract_features(fp))

# Remove None features and corresponding labels
valid_features = []
valid_labels = []
for i, feature in enumerate(features):
    if feature is not None:
        valid_features.append(feature)
        valid_labels.append(y[i])

X = valid_features
y = np.array(valid_labels)

# Padding sequences to fixed length
MAX_TIMESTEPS = 500  # Adjust based on your analysis
X = pad_sequences(X, maxlen=MAX_TIMESTEPS, padding='post', dtype='float32')

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

# Create TensorFlow datasets with batching
BATCH_SIZE = 32
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)

# Add Early Stopping
# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor='val_accuracy',
#     patience=5,
#     restore_best_weights=True
# )

# LSTM Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(MAX_TIMESTEPS, 13)),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(unique_genres), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train with early stopping and tqdm progress bar
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    # callbacks=[early_stopping],
    verbose=1 # show progress bar from keras
)
model.save('../model/genre_classification_fma.keras')
