#!/usr/bin/env python
"""
Script to download and prepare a test dataset for evaluating the GTZAN music genre classifier.
This script is used in the CI/CD pipeline to ensure consistent model evaluation.
"""
import os
import sys
import argparse
import requests
import zipfile
import tarfile
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd

# Default dataset location for GTZAN or similar dataset
DEFAULT_DATASET_URL = "https://storage.googleapis.com/dataset-mirrors/gtzan/gtzan_test_subset.tar.gz"
DEFAULT_OUTPUT_DIR = "test_data"

def download_file(url, output_path):
    """
    Download a file with progress bar
    """
    response = requests.get(url, stream=True)
    
    if response.status_code != 200:
        print(f"Error downloading file: HTTP status code {response.status_code}")
        return False
        
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(output_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
            
    progress_bar.close()
    
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR: Something went wrong during download")
        return False
        
    return True

def extract_archive(archive_path, output_dir):
    """
    Extract tar.gz or zip archive
    """
    print(f"Extracting {archive_path} to {output_dir}")
    
    if archive_path.endswith('.tar.gz'):
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=output_dir)
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    else:
        print(f"Unsupported archive format: {archive_path}")
        return False
        
    return True

def create_validation_split(data_dir, validation_size=0.2, random_seed=42):
    """
    Create a validation split from the dataset if it doesn't already have one
    """
    # Check if there are directories for each genre
    genres = [d for d in os.listdir(data_dir) 
              if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
    
    if not genres:
        print(f"No genre directories found in {data_dir}")
        return False
        
    print(f"Found genres: {genres}")
    
    # Create validation directory
    validation_dir = os.path.join(data_dir, 'validation')
    os.makedirs(validation_dir, exist_ok=True)
    
    np.random.seed(random_seed)
    
    for genre in genres:
        # Create genre directory in validation
        genre_val_dir = os.path.join(validation_dir, genre)
        os.makedirs(genre_val_dir, exist_ok=True)
        
        # Get all audio files
        genre_dir = os.path.join(data_dir, genre)
        audio_files = [f for f in os.listdir(genre_dir) 
                      if f.endswith(('.wav', '.au', '.mp3')) and os.path.isfile(os.path.join(genre_dir, f))]
        
        # Randomly select files for validation
        n_val = max(1, int(len(audio_files) * validation_size))
        val_files = np.random.choice(audio_files, size=n_val, replace=False)
        
        print(f"Moving {len(val_files)} files to validation set for genre '{genre}'")
        
        # Move files to validation directory
        for file in val_files:
            src = os.path.join(genre_dir, file)
            dst = os.path.join(genre_val_dir, file)
            shutil.copy2(src, dst)  # copy2 preserves metadata
            
    return True

def check_dataset_integrity(data_dir):
    """
    Check if dataset has expected structure and files
    """
    print("Checking dataset integrity...")
    
    # Check for genre directories
    genres = [d for d in os.listdir(data_dir) 
              if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
    
    if not genres:
        print(f"ERROR: No genre directories found in {data_dir}")
        return False
        
    # Check file counts in each genre
    file_counts = {}
    for genre in genres:
        genre_path = os.path.join(data_dir, genre)
        audio_files = [f for f in os.listdir(genre_path) 
                      if f.endswith(('.wav', '.au', '.mp3')) and os.path.isfile(os.path.join(genre_path, f))]
        file_counts[genre] = len(audio_files)
        
    print("File counts per genre:")
    for genre, count in file_counts.items():
        print(f"  {genre}: {count} files")
        
    # Check if any genre has zero files
    if any(count == 0 for count in file_counts.values()):
        print("WARNING: Some genres have no audio files")
        
    return True

def generate_metadata_file(data_dir, output_file="test_dataset_metadata.csv"):
    """
    Generate a metadata file with information about each audio file
    """
    print(f"Generating metadata file: {output_file}")
    
    metadata = []
    
    # Walk through all directories
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.wav', '.au', '.mp3')):
                # Determine genre from directory structure
                rel_path = os.path.relpath(root, data_dir)
                genre = rel_path.split(os.path.sep)[0] if rel_path != '.' else 'unknown'
                
                # Add to metadata
                file_path = os.path.join(root, file)
                metadata.append({
                    'file_path': file_path,
                    'genre': genre,
                    'split': 'validation' if 'validation' in rel_path else 'train',
                    'filename': file
                })
    
    # Create DataFrame and save to CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(output_file, index=False)
    
    print(f"Saved metadata for {len(metadata_df)} files to {output_file}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Download and prepare GTZAN test dataset')
    parser.add_argument('--url', default=DEFAULT_DATASET_URL,
                        help='URL to download the dataset (default: GTZAN test subset)')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for the dataset')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download if files already exist')
    parser.add_argument('--create-validation', action='store_true',
                        help='Create a validation split if not present')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Determine archive filename from URL
    archive_filename = os.path.basename(args.url)
    archive_path = os.path.join(args.output, archive_filename)
    
    # Download the dataset if needed
    if os.path.exists(archive_path) and args.skip_download:
        print(f"Archive {archive_path} already exists, skipping download")
    else:
        print(f"Downloading {args.url} to {archive_path}")
        success = download_file(args.url, archive_path)
        if not success:
            print("Failed to download dataset")
            sys.exit(1)
    
    # Extract the archive
    extract_success = extract_archive(archive_path, args.output)
    if not extract_success:
        print("Failed to extract dataset")
        sys.exit(1)
    
    # Create validation split if requested
    if args.create_validation:
        create_validation_split(args.output)
    
    # Check dataset integrity
    integrity_ok = check_dataset_integrity(args.output)
    if not integrity_ok:
        print("Dataset integrity check failed")
        sys.exit(1)
    
    # Generate metadata file
    generate_metadata_file(args.output)
    
    print("Dataset preparation complete!")

if __name__ == "__main__":
    main()
