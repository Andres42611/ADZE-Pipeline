#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from parser import init_datasplit_Parser

#initialize parser
parsed = init_datasplit_Parser()
args = parsed.parse_args()

def combine_datasets(paths):
    """Read and combine CSV files into one DataFrame."""
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)

def stratify_data(df, class_column='Class'):
    """Split data by class and then stratify into training, testing, and validation sets."""
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

def shuffle_and_save(df, filename):
    """Shuffle a DataFrame and save it to a CSV file."""
    df.sample(frac=1, random_state=42).reset_index(drop=True).to_csv(filename, index=False)

def process_datasets(csv_paths):
    """Process datasets: Read, combine, stratify by class, and save to new CSV files."""
    combined_df = combine_datasets(csv_paths)
    training_df, testing_df, validation_df = stratify_data(combined_df)
    
    shuffle_and_save(training_df, 'training.csv')
    shuffle_and_save(testing_df, 'testing.csv')
    shuffle_and_save(validation_df, 'validation.csv')

# Paths to the CSV files
csv_paths = [
  args.direc + '/caseA_data.csv',
  args.direc + '/caseB_data.csv',
  args.direc + '/caseC_data.csv',
  args.direc + '/caseD_data.csv',
  args.direc + '/caseE_data.csv'
]
    
process_datasets(csv_paths)
