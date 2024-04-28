#!/usr/bin/env python3

# File: datastrat.py
# Principal Investigator: Dr. Zachary Szpiech
# Date: 10 April 2024
# Author: Andres del Castillo
# Purpose: Stratifies 5 case-specific CSV datasets into a 80/10/10 split for training/test/validation by 'Class', then saves each to a new CSV file

import pandas as pd
from sklearn.model_selection import train_test_split

# Description:
#     This function aggregates multiple datasets from specified file paths into a single Pandas DataFrame.
# Accepts:
#     list paths: A list of strings, where each string is a file path to a CSV file containing part of the dataset to be combined.
# Returns:
#     DataFrame: A Pandas DataFrame resulting from the concatenation of all CSV files specified in the paths.
def combine_datasets(paths):
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)

# Description:
#     Splits a given DataFrame into training, testing, and validation sets stratified by a specified class column. This function ensures that data 
#     from each class is represented proportionally in each of the three datasets. The function first segregates the DataFrame by unique values in the class column,
#     then splits these segregated parts into training (80% of each class), testing (10% of each class),
#     and validation datasets (10% of each class) using the `train_test_split` method from scikit-learn, with shuffling enabled 
#     to randomize the data selection. These individual splits are then aggregated into their respective datasets for training, testing, and 
#     validation purposes.
# Accepts:
#     DataFrame df: The DataFrame to be stratified and split.
#     str class_column='Class': The name of the column in df that contains the class labels based on which the stratification is to be done. This 
#                               column identifies the different classes or categories in the data that need to be equally represented in the splits.
# Returns:
#     DataFrame, DataFrame, DataFrame: Three DataFrames corresponding to the training, testing, and validation datasets, respectively.
def stratify_data(group, train_size=0.8, test_size=0.1, val_size=0.1):
    # Ensure sizes sum to 1
    assert train_size + test_size + val_size == 1, "Sizes must sum to 1"

    # Shuffle replicates
    replicates = group['Replicate'].unique()
    train_replicates, temp_replicates = train_test_split(replicates, train_size=train_size, shuffle=True)
    test_replicates, val_replicates = train_test_split(temp_replicates, train_size=test_size/(test_size + val_size), shuffle=True)

    # Assign sets
    train_set = group[group['Replicate'].isin(train_replicates)]
    test_set = group[group['Replicate'].isin(test_replicates)]
    val_set = group[group['Replicate'].isin(val_replicates)]

    return train_set, test_set, val_set

# Description:
#     Randomly shuffles the rows of a given DataFrame and saves the shuffled DataFrame to a CSV file specified by the filename. This function uses 
#     the `sample` method with `frac=1` to shuffle all rows of the DataFrame, ensuring a random order. The `random_state` parameter is set to a 
#     fixed value to allow for reproducibility of the shuffle. After shuffling, the DataFrame's index is reset to ensure a continuous index from 0 
#     without any gaps, and the original index is dropped. The shuffled DataFrame is then saved to a CSV file, with `index=False` to prevent the 
#     index from being saved as a separate column in the file.
# Accepts:
#     DataFrame df: The DataFrame to be shuffled.
#     str filename: The path and name of the CSV file where the shuffled DataFrame will be saved.
# Returns:
#     None.
def shuffle_and_save(df, filename):
    df.sample(frac=1, random_state=42).reset_index(drop=True).to_csv(filename, index=False)

# Description:
#     This function completes the preprocessing of multiple datasets. It first combines datasets from a list of CSV file paths into a single DataFrame, then stratifies this 
#     combined dataset into training, testing, and validation sets, ensuring proportional representation of classes across these sets. After stratification, each set is shuffled
#     to randomize the order of its rows and then saved to separate CSV files: 'training.csv' for the training set, 'testing.csv' for the testing set, and 'validation.csv' for the validation set. 
# Accepts:
#     list csv_paths: A list of strings, where each string is a path to a CSV file containing a portion of the data to be processed. 
# Returns:
#     None: 
def process_datasets(csv_paths):
    df = combine_datasets(csv_paths)
    
    train_frames = []
    test_frames = []
    val_frames = []
    
    for class_name, group in df.groupby('Class'):
        train, test, val = stratify_data(group)
        train_frames.append(train)
        test_frames.append(test)
        val_frames.append(val)
    
    train_df, test_df, val_df = pd.concat(train_frames), pd.concat(test_frames), pd.concat(val_frames)

    shuffle_and_save(train_df, './ADZE_pipeline/3pop/StratData/training.csv')
    shuffle_and_save(test_df, './ADZE_pipeline/3pop/StratData/testing.csv')
    shuffle_and_save(val_df, './ADZE_pipeline/3pop/StratData/validation.csv')

process_datasets(csv_paths)
