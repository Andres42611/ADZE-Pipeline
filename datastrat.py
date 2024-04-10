#!/usr/bin/env python3

# File: datastrat.py
# Principal Investigator: Dr. Zachary Szpiech
# Date: 10 April 2024
# Author: Andres del Castillo
# Purpose: Stratifies 5 case-specific CSV datasets into a 80/10/10 split for training/test/validation by 'Class', then saves each to a new CSV file

import pandas as pd
from sklearn.model_selection import train_test_split
from parser import init_datasplit_Parser

#initialize parser
parsed = init_datasplit_Parser()
args = parsed.parse_args()

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
def stratify_data(df, class_column='Class'):
    training_data = []
    testing_data = []
    validation_data = []

    for class_value in df[class_column].unique():
        class_df = df[df[class_column] == class_value]
        train, temp = train_test_split(class_df, test_size=0.2, random_state=42, shuffle=True)
        test, validate = train_test_split(temp, test_size=0.5, random_state=42, shuffle=True)
        
        training_data.append(train)
        testing_data.append(test)
        validation_data.append(validate)

    return pd.concat(training_data), pd.concat(testing_data), pd.concat(validation_data)

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
    combined_df = combine_datasets(csv_paths)
    training_df, testing_df, validation_df = stratify_data(combined_df)
    
    shuffle_and_save(training_df, 'args.direc/StratData/training.csv')
    shuffle_and_save(testing_df, 'args.direc/StratData/testing.csv')
    shuffle_and_save(validation_df, 'args.direc/StratData/validation.csv')

# Paths to the case-specific CSV files
csv_paths = [
  args.direc + '/CaseA/caseA_data.csv',
  args.direc + '/CaseB/caseB_data.csv',
  args.direc + '/CaseC/caseC_data.csv',
  args.direc + '/CaseD/caseD_data.csv',
  args.direc + '/CaseE/caseE_data.csv'
]
    
process_datasets(csv_paths)
