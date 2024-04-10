#!/usr/bin/env python3

# File: ADZEtoCSV.py
# Principal Investigator: Dr. Zachary Szpiech
# Date: 10 April 2024
# Author: Andres del Castillo
# Purpose: Converts all three ADZE output files into a processed CSV dataset
import csv
import pandas as pd
import numpy as np
from parser import init_ADZEtoCSV_Parser

#Parse command-line arguments
parsed = init_ADZEtoCSV_Parser()
args = parsed.parse_args()


# Description:
#     Converts the contents of a file specified by the path to a structured NumPy array. This function is designed to process files containing 
#     statistical data from ADZE output, which can be in two formats. The function parses each line of the ADZE output file, extracts relevant 
#     the vector (mean, variance, and standard deviation error), and organizes them into a structured array with identifiers for each data point 
#     (either a population ID for alpha/pi files or a combination ID for pihat files).
# Accepts:
#     str path: The file path to the ADZE output file. Conceptually, this is the source of the statistical data to be parsed and converted.
#     bool alphafile: A flag indicating whether the file is an alpha file (True) or not (False).
#     bool pihatfile: A flag indicating whether the file is a pihat file (True) or not (False).
# Returns:
#     ndarray data_array: A NumPy array containing the parsed data from the statistics file.
def ADZE_to_array(path, alphafile=False, pihatfile=False):
  data_list = []
  with open(path, 'r') as file:
    rep_counter = 1
    prev_k = 2
    for line in file:
        columns = line.split()
        if columns:
            stat_vector = f"({columns[-3]}, {columns[-2]}, {columns[-1]})"
            if pihatfile: #for pihat files
              comb_size = int(path[-1])
              combinationID = "pihat_" + ''.join(columns[:comb_size])

              data_row = [combinationID] + [columns[comb_size]] + [stat_vector] # format is 'combination', 'k', '(mean, var, SDE)' 
            elif columns: #for alpha files
              popID = f"alpha_{columns[0]}" if alphafile else f"pi_{columns[0]}"
              data_row = [popID] + [columns[1]] + [stat_vector]

            data_list.append(data_row)

  # Convert list to NumPy array
  data_array = np.array(data_list, dtype=None)
  return data_array


# Description:
#     Generates a list of file paths for pihat files based on a specified output prefix and a range of combination sizes. This function constructs 
#     the paths by appending a combination identifier ('comb_') followed by an index that represents the combination size (from 2 to J-1, where J is
#     the total number of populations) to the given output prefix. 
# Accepts:
#     str output_prefix: The base path or prefix to which the combination identifier and index will be appended. Conceptually, this represents the 
#                        directory and/or initial part of the filename before the combination size specification.
#     int J: The number of populations in the simulations.
# Returns:
#     list paths: A list of strings, each representing a file path for a pihat file corresponding to a specific combination size.
def pihat_pathgen(output_prefix, J):
  paths = []
  for i in range(2, J):
    paths.append(output_prefix + 'comb_' + str(i))
  
  return paths


# Description:
#     This function generates a CSV file containing statistical data processed from multiple ADZE output files, including alpha diversity files, pi 
#     diversity files, and pihat files. It first constructs the base file path from the first stat file, extracts data from alpha and pi files using 
#     ADZE_to_array, and generates paths for pihat files based on the number of unique alpha identifiers. It then processes each pihat file and 
#     combines all statistics into a single Pandas DataFrame. The DataFrame is organized with columns for each statistic (alpha, pi, and pihat values), 
#     a 'Class' column indicating the biological or experimental group, and a 'Replicate' column specifying the replicate number. The function 
#     finally exports this DataFrame to a CSV file named after the output prefix with '.csv' extension. This CSV file facilitates the analysis of 
#     biodiversity data by consolidating various statistics into a single, easily accessible format.
# Accepts:
#     list stat_files: A list containing the paths to the stat files (alpha and pi files) from which the data is to be processed. Conceptually, these 
#                      files contain the raw statistical output from ADZE analysis which needs to be aggregated.
#     str label: The label for the 'Class' column in the output DataFrame. This label indicates the biological or experimental group associated with 
#                the data, providing a way to categorize or differentiate data in the analysis.
# Returns:
#     int: Always returns 0, indicating that the function has completed its execution successfully. The conceptual purpose of this return value is to 
#          signify the end of the operation, although it doesn't convey any outcome of the process itself.
def CSV_generator(stat_files, label):
  #get file path to ADZE statistics files
  output_prefix = (stat_files[0]).split('richness')[0]
  
  #get array of alpha and pi values across all standardized sample sizes (k)
  alpha_arr = ADZE_to_array(stat_files[0], True)
  pi_arr = ADZE_to_array(stat_files[1])

  #get statistic IDs for header
  alphas, pis = np.unique(alpha_arr[:,0]), np.unique(pi_arr[:,0])
  
  #moving to pihat files, get the paths
  pihat_paths = pihat_pathgen(output_prefix, len(alphas))
  
  #now get all pihat values for all possible [2, J-1]-tuples across all standardized sample sizes
  all_pihat_arrs = []
  for path in pihat_paths:
    all_pihat_arrs.append(ADZE_to_array(path, False, True))

  #get pihat statistic IDs for header
  pihats = [np.unique(pihat_arr[:,0]) for pihat_arr in all_pihat_arrs]

  #begin building replicate's dataframe
  headers = alphas.tolist() + pis.tolist() + np.concatenate(pihats).tolist() + ['Class', 'Replicate']
  k_val = [str(i) for i in range(2,200)] #assuming sample size 200
  df_final = pd.DataFrame(index=k_val, columns=headers)
  df_final['Class'] = label

  #find replicate number using string manipulation
  rep_num = output_prefix.split(label+'_rep_')[-1]
  df_final['Replicate'] = int(rep_num)

  #map arrays to dataframe
  all_stat_arrs = [alpha_arr, pi_arr] + all_pihat_arrs 
  for arr in all_stat_arrs:
    for row in arr:
      col_index, row_index, value = row
      df_final.at[row_index, col_index] = value

  df_final.reset_index(inplace=True)
  df_final.rename(columns={'index': 'k'}, inplace=True)

  #save dataframe as CSV
  df_final.to_csv(output_prefix + '.csv', index=False)
  
  return 0


stat_files = [args.rout, args.pout]
CSV_generator(stat_files, args.Case)
