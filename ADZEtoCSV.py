#!/usr/bin/env python3

# File: ADZEtoCSV.py
# Principal Investigator: Dr. Zachary Szpiech
# Date: 21 February 2024
# Author: Andres del Castillo
# Purpose: Converts all three ADZE output files into a processed CSV dataset
import csv
from parser import init_ADZEtoCSV_Parser

#Parse command-line arguments
parsed = init_ADZEtoCSV_Parser()
args = parsed.parse_args()


# Description:
#     This function reads ADZE statistical data from a specified file and organizes it into a dictionary for analysis. 
#     The function can operate in two modes: default and "pihat" mode. 
#     
#     In default mode, it organizes data by an integer key (the second element of each line) and collects statistics as tuples in a list. 
#     In "pihat" mode, it uses a combination of the first two elements of each line as keys for
#     sub-dictionaries, with an integer key (the third element) as the 
#     main dictionary key, organizing the statistics as tuples in these sub-dictionaries.
# Accepts:
#     str file_path, the path to the file from which to read the statistical data. The file is expected to contain 
#     whitespace-separated values where the specific columns used depend on the mode of operation (is_pihat);
#     bool is_pihat (optional), a flag indicating whether to operate in "pihat" mode. In "pihat" mode, the function 
#     uses a different schema for interpreting and organizing data. If False or omitted, the function operates in default mode;
# Returns:
#     dict data_dict, a dictionary containing the organized statistical data. In default mode, this dictionary maps integer 
#     keys to lists of statistic tuples. In "pihat" mode, it maps integer keys to dictionaries, which then map population 
#     combination strings to statistic tuples;
def read_stats_from_file(file_path, is_pihat=False):
    data_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            
            if is_pihat:
                # For pihat, use the combination of the first two numbers as the population combination key
                pop_comb = parts[0] + parts[1]
                k = int(parts[2])
                stats = tuple(map(float, parts[4:]))
                if k not in data_dict:
                    data_dict[k] = {}
                data_dict[k][pop_comb] = stats
            else:
                k = int(parts[1])
                stats = tuple(map(float, parts[3:]))
                if k not in data_dict:
                    data_dict[k] = []
                data_dict[k].append(stats)
    return data_dict


# Description:
#     This function combines ADZE statistical data from three different sources (alpha, pi, and pihat statistics) into a single 
#     consolidated dictionary. It iterates over the union of keys from all three input dictionaries and aggregates the data 
#     for each key into a nested dictionary structure. This structure includes default values for missing data, ensuring that 
#     each key in the combined dictionary has a consistent format.
# Accepts:
#     dict alpha_stats, a dictionary where each key maps to a list containing three alpha statistical measures;
#     dict pi_stats, a dictionary where each key maps to a list containing three pi statistical measures;
#     dict pihat_stats, a dictionary where each key maps to a sub-dictionary. The sub-dictionaries map string identifiers 
#     (representing population combinations) to tuples containing three pihat statistical measures;
# Returns:
#     dict combined_data, a dictionary that combines the statistics from alpha_stats, pi_stats, and pihat_stats. For each 
#     key found in any of the three input dictionaries, combined_data will contain a nested dictionary with three entries:
#     'alpha', 'pi', and 'pihat', each storing the corresponding statistics from the input dictionaries. If any statistics 
#     are missing for a key, default values of [None, None, None] or {'12': (None, None, None), '13': (None, None, None), 
#     '23': (None, None, None)} are used for 'alpha'/'pi' and 'pihat', respectively.
def combine_statistics(alpha_stats, pi_stats, pihat_stats):
    combined_data = {}
    for k in set(list(alpha_stats.keys()) + list(pi_stats.keys()) + list(pihat_stats.keys())):
        combined_data[k] = {
            'alpha': alpha_stats.get(k, [None, None, None]),
            'pi': pi_stats.get(k, [None, None, None]),
            'pihat': pihat_stats.get(k, {'12': (None, None, None), '13': (None, None, None), '23': (None, None, None)})
        }
    return combined_data


# Description:
#     This function writes the combined statistical data to a CSV file. The combined data includes statistics from alpha, 
#     pi, and pihat analyses, formatted and organized by g-valued key 'k'. The function creates a CSV file with a header row 
#     followed by rows for each key in the combined data, including statistical values and a class label for 
#     each row.
# Accepts:
#     dict combined_data, a dictionary containing the combined statistics for different keys. Each key maps to a nested 
#     dictionary with 'alpha', 'pi', and 'pihat' statistics, each of which may contain a tuple of (mean, variance, std. error);
#     str output_csv_path, the file path where the CSV file will be written. This specifies the location and name of the output file;
#     str label, classification label;
# Returns:
#     None. The function writes the combined statistical data to a CSV file at the specified path;
def write_combined_csv(combined_data, output_csv_path, label):
    with open(output_csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        header = ['k', 
                  'alpha_1', 'alpha_2', 'alpha_3',
                  'pi_1', 'pi_2', 'pi_3',
                  'pihat_12', 'pihat_13', 'pihat_23',
                  'Class']
        csv_writer.writerow(header)

        for k, data in sorted(combined_data.items()):
            row = [k]
            # Process alpha and pi statistics first
            for stat_type in ['alpha', 'pi']:
                stats = data[stat_type]
                for stat in stats:
                    if stat is not None:
                        row.append(','.join([f"{value:.5f}" for value in stat]))  # Apply formatting to each element
                    else:
                        row.append('()')

            # Process pihat statistics
            pihat_stats = data['pihat']
            for pop_comb in ['12', '13', '23']:
                stats = pihat_stats.get(pop_comb, (None, None, None))
                if all(stats):  # Check if all elements in stats are not None
                    row.append(','.join([f"{value:.7f}" for value in stats]))  # Apply formatting to each element
                else:
                    row.append('()')

            row.append(label)
            csv_writer.writerow(row)

# Description:
#     This function processes the three types of statistical files (alpha, pi, and pihat) output by ADZE by reading 
#     the statistics from each file, combining them into a single data structure, and then writing the combined data 
#     to a CSV file. The output CSV file is saved in the same directory as the alpha file but with a '.csv' extension.
# Accepts:
#     str alpha_file, the file path to the alpha statistics file;
#     str pi_file, the file path to the pi statistics file;
#     str pihat_file, the file path to the pihat statistics file;
#     str label, classification label;
# Returns:
#     None. The function directly writes the combined statistical data to a CSV file;
def process_files(alpha_file, pi_file, pihat_file, label):
    output_csv_path = alpha_file.split('richness')[0] + '.csv'  #same file directory as ADZE output but csv 

    alpha_stats = read_stats_from_file(alpha_file)
    pi_stats = read_stats_from_file(pi_file)
    pihat_stats = read_stats_from_file(pihat_file, is_pihat=True)
    combined_data = combine_statistics(alpha_stats, pi_stats, pihat_stats)
    write_combined_csv(combined_data, output_csv_path, label)


process_files(args.rout, args.pout, args.cout, args.Case)
