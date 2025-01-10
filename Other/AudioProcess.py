#!/usr/bin/env python3

# Import necessary libraries for handling audio files, performing FFT, calculating moments, and file operations
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
from scipy.stats import moment
import os
import random

# Function to read a wav file and convert it to a 1D vector
# - Converts stereo audio to mono by averaging channels if necessary
# - Returns a flattened array of audio samples
def wav_to_vector(file_path):
    sample_rate, data = wavfile.read(file_path)  # Read wav file
    if len(data.shape) == 2:  # Check if the audio is stereo
        data = np.mean(data, axis=1)  # Convert to mono by averaging channels
    return data.flatten()  # Flatten the array to 1D

# Function to normalize a vector by its maximum absolute value
# - Prevents division by zero by returning the vector unchanged if max value is zero
def normalize_vector(vector):
    max_val = np.max(np.abs(vector))  # Find max absolute value in the vector
    return vector if max_val == 0 else vector / max_val  # Normalize or return unchanged if max_val is 0

# Function to perform the Fast Fourier Transform (FFT) on a vector
# - FFT converts the time-domain signal into the frequency domain
def perform_fft_on_vector(vector):
    return fft(vector)

# Function to calculate statistical moments of a vector
# - Uses scipy.stats.moment to calculate up to the given number of moments
# - Returns a list of moments (e.g., mean, variance, skewness, kurtosis, etc.)
def calculate_moments(vector, num_moments=150):
    return [moment(vector, moment=i) for i in range(1, num_moments + 1)]

# Main processing function that handles multiple steps:
# - Randomly samples files from the directory
# - Loads audio data, checks length, normalizes, and extracts subsets of vectors
# - Saves the audio data (X and Y matrices), FFT results, and statistical moments
# - Logs the files used in processing
def process_files_and_save(directory_path, output_dir, sample_size=1000, min_length=9 * 16000):
    # List all wav files with specific naming patterns for features and targets
    feature_files = sorted([os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('_Feature.wav')])
    target_files = sorted([os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('_Target.wav')])

    # Randomly sample a specified number of files (default: 1000) or the available number
    sampled_indices = random.sample(range(len(feature_files)), min(sample_size, len(feature_files)))
    sampled_feature_files = [feature_files[i] for i in sampled_indices]
    sampled_target_files = [target_files[i] for i in sampled_indices]

    # Initialize lists to store processed feature and target data
    X, Y = [], []
    used_files = []  # List of processed files for logging

    # Inform the user about the number of files being processed
    print(f"Processing {len(sampled_feature_files)} files...", flush=True)

    # Loop over the sampled feature and target files
    for idx, (feature_file, target_file) in enumerate(zip(sampled_feature_files, sampled_target_files), start=1):
        print(f"Processing file {idx}/{len(sampled_feature_files)}: {feature_file}, {target_file}", flush=True)

        # Convert wav files to vectors
        feature_vector = wav_to_vector(feature_file)  #first 20 seconds of audio
        target_vector = wav_to_vector(target_file) #rest of audio past 20 seconds 

        # Double Check if the vectors are long enough for processing (each vector has more than 9 seconds of audio)
        if len(feature_vector) >= min_length and len(target_vector) >= min_length:
            # Normalize the vectors
            feature_vector = normalize_vector(feature_vector)
            target_vector = normalize_vector(target_vector)

            # Append truncated versions of the vectors to lists (X for features, Y for targets)
            X.append(feature_vector[:20*16000])  # Use first 20 seconds (16,0000*20 samples) for features
            Y.append(target_vector[:4*16000])  # Use first 4 seconds (16,0000*4 samples) samples for targets

            # Record the filenames used for future reference
            used_files.append(f"{feature_file}, {target_file}")
        else:
            # Skip the file if it does not meet the length requirement
            print(f"Skipping file {feature_file} or {target_file} due to insufficient length.", flush=True)

    # Convert the lists to numpy arrays and save to disk
    print(f"Saving amplitude matrices (X and Y)...", flush=True)
    X = np.array(X)
    Y = np.array(Y)
    
    np.save(os.path.join(output_dir, 'ampX.npy'), X)  # Save feature matrix X
    np.save(os.path.join(output_dir, 'ampY.npy'), Y)  # Save target matrix Y

    # Perform FFT on both X and Y matrices and save the results
    print(f"Performing FFT and saving results...", flush=True)
    fftX = np.array([perform_fft_on_vector(row) for row in X])
    fftY = np.array([perform_fft_on_vector(row) for row in Y])
    
    np.save(os.path.join(output_dir, 'fftX.npy'), fftX)  # Save FFT of features
    np.save(os.path.join(output_dir, 'fftY.npy'), fftY)  # Save FFT of targets

    # Calculate moments for each row of X and save the results
    print(f"Calculating moments for X and saving results...", flush=True)
    momentsX = np.array([calculate_moments(row) for row in X])
    np.save(os.path.join(output_dir, 'momX.npy'), momentsX)  # Save moments of features

    # Save the list of processed files for tracking
    print(f"Saving list of used files...", flush=True)
    with open(os.path.join(output_dir, 'used_files.txt'), 'w') as f:
        for line in used_files:
            f.write(f"{line}\n")

    print("Processing completed successfully!", flush=True)

# Main function to set up paths and call the processing function
# - Checks if the output directory exists, creates it if not
# - Calls the process_files_and_save function to begin processing
def main():
    # Define the input and output directories
    directory_path = '/storage/group/zps5164/default/shared/ADZE-Pipeline/Music/fma_med_wav'
    output_dir = '/storage/group/zps5164/default/shared/ADZE-Pipeline/Music/three'

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process the files and save the results
    process_files_and_save(directory_path, output_dir)

# Entry point for the script, calls the main function if the script is executed
if __name__ == "__main__":
    main()