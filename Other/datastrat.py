#!/usr/bin/env python3

# File: datastrat.py
# Principal Investigator: Dr. Zachary Szpiech
# Date: 10 April 2024
# Author: Andres del Castillo
# Purpose: Stratifies 5 case-specific CSV datasets into an 80/10/10 split for training/test/validation by 'Class' and 'Replicate', then saves each to a new CSV file

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from parser import init_datasplit_Parser

# Initialize parser
parsed = init_datasplit_Parser()
args = parsed.parse_args()

def combine_datasets(paths):
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)

def stratify_data(df, train_size=0.8, test_size=0.1, val_size=0.1):
    # Ensure sizes sum to 1
    assert train_size + test_size + val_size == 1, "Sizes must sum to 1"

    # Create a grouping column
    df['Group'] = df['Class'].astype(str) + '_' + df['Replicate'].astype(str)

    # Initialize GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, test_size=test_size + val_size, random_state=42)
    train_idx, temp_idx = next(gss.split(df, groups=df['Group']))

    # Split data into training and temporary sets
    train_set = df.iloc[train_idx]
    temp_set = df.iloc[temp_idx]

    # Further split temporary set into validation and test sets
    gss = GroupShuffleSplit(n_splits=1, train_size=test_size / (test_size + val_size), test_size=val_size / (test_size + val_size), random_state=42)
    test_idx, val_idx = next(gss.split(temp_set, groups=temp_set['Group']))

    test_set = temp_set.iloc[test_idx]
    val_set = temp_set.iloc[val_idx]

    # Drop the 'Group' column before returning the final sets
    train_set = train_set.drop(columns=['Group'])
    test_set = test_set.drop(columns=['Group'])
    val_set = val_set.drop(columns=['Group'])

    return train_set, test_set, val_set

def shuffle_and_save(df, filename):
    df.sample(frac=1, random_state=42).reset_index(drop=True).to_csv(filename, index=False)

def process_datasets(csv_paths):
    df = combine_datasets(csv_paths)
    
    train_df, test_df, val_df = stratify_data(df)

    shuffle_and_save(train_df, args.direc + '/StratData/training.csv')
    shuffle_and_save(test_df, args.direc + '/StratData/testing.csv')
    shuffle_and_save(val_df, args.direc + '/StratData/validation.csv')

# Paths to the case-specific CSV files
csv_paths = [
  args.direc + '/CaseA/caseA_data.csv',
  args.direc + '/CaseB/caseB_data.csv',
  args.direc + '/CaseC/caseC_data.csv',
  args.direc + '/CaseD/caseD_data.csv',
  args.direc + '/CaseE/caseE_data.csv'
]
    
process_datasets(csv_paths)
