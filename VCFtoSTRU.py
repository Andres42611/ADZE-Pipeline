#!/usr/bin/env python3

# File: VCFtoSTRU.py
# Principal Investigator: Dr. Zachary Szpiech
# Date: 21 February 2024
# Author: Andres del Castillo
# Purpose: Converts all simulation VCF files in a given directory to .stru files

from parser import create_parser

#Parse command-line arguments
parsed = init_VCFtoSTRU_Parser()
args = parsed.parse_args()

# Description:
#     This function processes a population file to extract and sort individual sample names and their associated data.
#     It reads from a specified file where each row contains sample information, sorts this information based on a specified 
#     criterion (the second element in each row (the assigned population), must be an integer), and formats the data for further processing.
# Accepts:
#     str file, the path to the file containing population data. The file is expected to have rows where the first element
#     is the sample name and the second element is a numeric value used for sorting purposes.
# Returns:
#     list formatted_rows, a list where each element is a string formatted as '<sample_name> <numeric_value>', reflecting
#     the sorted order based on the numeric value. This list is ready for further processing or output;
#     list input_sample_names, a list of sample names extracted and sorted based on the associated numeric value. This list
#     can be used for identifying samples in subsequent analyses.
def process_population_file(file):
    with open(file) as infile:
       rows = [line.split() for line in infile]

    # Sort the rows based on the second element, converting it to integer for comparison
    sorted_rows = sorted(rows, key=lambda x: int(x[1]))
    input_sample_names = [ind[0] for ind in sorted_rows]

    # Format each row as specified and store in a new list
    formatted_rows = ['{} {}'.format(row[0], row[1]) for row in sorted_rows]

    # The formatted_rows list now contains the elements in the desired format
    return formatted_rows, input_sample_names


# Description:
#     This function converts a Variant Call Format (VCF) files into a .stru format files used for ADZE genetic analysis. 
#     It reads a VCF file, extracts genotype information for specified samples, and writes the genotype data 
#     into a new file in STRU format. The STRU format is designed for population genetics analyses and includes 
#     loci information along with the genotype data for each sample across all loci.
# Accepts:
#     str vcf_file_path, the path to the VCF file containing variant call information;
#     list input_sample_names, a list of user-inputted sample names to extract genotype information for from the VCF file;
#     list formatted_rows, a list of formatted row identifiers or metadata for each sample to include in the STRU file.
# Returns:
#     int 0, void indicates successful execution and completion of the function.
def convert_all_vcf_to_stru(vcf_file_path, input_sample_names, formatted_rows):
    # Initialize variables
    loci_alleles = {}
    sample_genotype_data = {name: ([], []) for name in input_sample_names}  # Store genotypes for each sample

    # Open VCF file once to read all necessary data
    with open(vcf_file_path, 'r') as vcf:
        for line in vcf:
            line = line.strip()
            if line.startswith('##'):
                continue
            if line.startswith('#'):
                # Process header line to find indices of samples
                headers = line.split('\t')
                sample_indices = [headers.index(name) for name in input_sample_names]
                continue

            data = line.split('\t')
            chrom, pos, ref, alt = data[0], data[1], data[3], data[4]
            loci_key = f'{chrom}-{pos}'
            all_alleles = [ref] + alt.split(',')
            loci_alleles[loci_key] = {allele: str(i) for i, allele in enumerate(all_alleles)}

            # Extract genotype data for each sample
            genotype_info_index = data[8].split(':').index('GT')
            for sample_name, sample_index in zip(input_sample_names, sample_indices):
                genotype_data = data[sample_index].split(':')[genotype_info_index]
                alleles = genotype_data.replace('|', '/').split('/')
                alleles_values = [loci_alleles[loci_key][all_alleles[int(index)]] if index.isdigit() else '-9' for index in alleles]

                sample_genotype_data[sample_name][0].append(alleles_values[0])
                sample_genotype_data[sample_name][1].append(alleles_values[1] if len(alleles_values) > 1 else alleles_values[0])

    # Write to the output file (same name as VCF file, just with .stru format)
    stru_file_path = vcf_file_path.split('.vcf')[0] + '.stru'
    with open(stru_file_path, 'w') as output:
        # Write header for loci
        output.write(' '.join(loci_alleles.keys()) + '\n')
        # Write genotype data for each sample
        for counter, sample_name in enumerate(input_sample_names):
            genotype_data_first_line, genotype_data_second_line = sample_genotype_data[sample_name]
            output.write(f'{formatted_rows[counter]} ' + ' '.join(genotype_data_first_line) + '\n')
            output.write(f'{formatted_rows[counter]} ' + ' '.join(genotype_data_second_line) + '\n')

    return 0

formatted_rows, input_sample_names = process_population_file(args.POP)

convert_all_vcf_to_stru(args.VCF, input_sample_names, formatted_rows)
