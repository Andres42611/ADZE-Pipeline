#!/usr/bin/env python3

# File: ADZEtoCSV.py
# Principal Investigator: Dr. Zachary Szpiech
# Date: 11 June 2024
# Author: Andres del Castillo
# Purpose: Flattens dataset into 1D replicate vectors and computes the euclidean, standardized euclidean, correlation, or with an RBF-Kernel
#    with parameter sigma: sigma is the median of the euclidean distances

import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd

# Description:
#   Convert a string representation of a vector into a list of floats.
#   Assumes the input string is formatted as '(x1,x2,x3,...)'.
# Accepts:
#   str string: String representation of a vector.
# Returns:
#   list: List of floats parsed from the input string.
def convert_to_float_vector(string):
    # Remove the parentheses and split by comma
    return [float(x) for x in string.strip('()').split(',')]

# Description:
#   Process the input DataFrame to convert specific columns into lists of floats
#   and then aggregate these lists by class and replicate.
# Accepts:
#   pd.DataFrame df: Input DataFrame containing the data to process.
#   bool only_mean: Flag indicating whether to keep only the mean value of each list.
# Returns:
#   dict: A dictionary with class_replicate as keys and aggregated vectors as values.
def process_data(df, only_mean=True):
    # Initialize the final dictionary
    fin_dict = {}

    # Preprocess columns to convert them into lists of floats
    for column in ['alpha_1', 'alpha_2', 'alpha_3', 'pi_1', 'pi_2', 'pi_3', 'pihat_12', 'pihat_13', 'pihat_23']:
        df[column] = df[column].apply(convert_to_float_vector)

    # Group data by 'Class' and 'Replicate'
    grouped = df.groupby(['Class', 'Replicate'])

    # Iterate through each unique class and replicate
    for (cls, rep), group in grouped:
        possible_k = sorted(group['k'].unique())

        # Assert all k in [2, 199] are found
        assert possible_k == list(range(2, 200)), f"Missing k values for class {cls} and replicate {rep}"

        fin_v = []

        # Process each specified column
        for column in ['alpha_1', 'alpha_2', 'alpha_3', 'pi_1', 'pi_2', 'pi_3', 'pihat_12', 'pihat_13', 'pihat_23']:
            # Extract the values as a numpy array
            values = np.array(group[column].tolist())

            if only_mean:
                # Keep only the mean value (first element) of each list
                values = values[:, 0]

            # Flatten the values and append to the final vector
            fin_v.extend(values.flatten())

        fin_dict[f"{cls}_{rep}"] = fin_v

    return fin_dict

# Description:
#   Compute and save distances between vectors in the input dictionary based on the specified distance type.
# Accepts:
#   dict fin_dict: A dictionary with class_replicate as keys and vectors as values.
#   str output_file: The file path to save the distances.
#   str distance_type: The type of distance to compute. Options are 'euclidean', 'seuclidean', 'correlation', 'rbf'.
# Returns:
#   None
def compute_distances(fin_dict, output_file, distance_type='euclidean'):
    # Extract keys and corresponding vectors
    keys = list(fin_dict.keys())
    vectors = np.array(list(fin_dict.values()))

    # Select the appropriate distance metric
    if distance_type == 'seuclidean':
        variances = np.var(vectors, axis=0, ddof=1)
        distances = pdist(vectors, metric='seuclidean', V=variances)
    elif distance_type == 'correlation':
        distances = pdist(vectors, metric='correlation')
    elif distance_type == 'rbf':
        euclidean_distances = pdist(vectors, metric='euclidean')
        sigma = np.median(euclidean_distances)
        distances = np.exp(-euclidean_distances**2 / (2 * sigma**2))
    else:  # default to 'euclidean'
        distances = pdist(vectors, metric='euclidean')

    # Get the indices for the upper triangle
    indices = np.triu_indices(len(keys), k=1)

    # Determine the appropriate file header
    header = 'Replicate1,Replicate2,Distance\n' if distance_type != 'rbf' else 'Replicate1,Replicate2,RBF_Distance\n'

    # Save the distances and corresponding replicate pairs to a file
    with open(output_file, 'w') as f:
        f.write(header)
        for (i, j), dist in zip(zip(*indices), distances):
            f.write(f'{keys[i]},{keys[j]},{dist}\n')
