#!/usr/bin/env python3
# File: parser.py
# Principal Investigator: Dr. Zachary Szpiech
# Date: 21 February 2024
# Author: Andres del Castillo
# Purpose: Initializes parsers for parsing command line arguments using argparse

import argparse
    
def init_Simulation_Parser():
    parser = argparse.ArgumentParser(description='msprime Simulation parameters')
    
    # Files and mandatory inputs for MSPRIME simulation
    parser.add_argument('-D', '--direc', type=str, required=True, metavar='SAVING_DIRECTORY', help='Directory to save simulation .tree files')
    parser.add_argument('-s', '--source', type=str, required=False, default='0', metavar='SOURCE_MIGRATION_POP', help="Source population for migration; default: '0' for no migrationn")
    parser.add_argument('-d', '--dest', type=str, required=False, default='0', metavar='DESTINATION_MIGRATION_POP', help="Destination population for migration; default: '0' for no migration")
    parser.add_argument('-C', '--Case', type=str, required=True, metavar='MIGRATION_CASE', help='Migration Case for Study')
    
    return parser
    
def init_VCFtoSTRU_Parser():
    parser = argparse.ArgumentParser(description='msprime Simulation parameters')

    # Files and mandatory inputs for converting VCF file to .stru file
    parser.add_argument('-V', '--VCF', type=str, required=True, metavar='VCF_FILE', help='VCF file for conversion')
    parser.add_argument('-P', '--POP', type=str, required=True, metavar='POPULATION_FILE', help='Population tab-delimited txt file with IDs and Pop IDs')

    return parser

def init_ADZEtoCSV_Parser():
    parser = argparse.ArgumentParser(description='ADZE output to .csv parameters')

    # Files and mandatory inputs for converting VCF file to .stru file
    parser.add_argument('-r', '--rout', type=str, required=True, metavar='RICHNESS_FILE', help='R_OUT File Path')
    parser.add_argument('-p', '--pout', type=str, required=True, metavar='PRIVATE_RICHNESS_FILE', help='P_OUT File Path')
    parser.add_argument('-c', '--cout', type=str, required=True, metavar='COMB_RICHNESS_FILE', help='C_OUT File Path')
    parser.add_argument('-C', '--Case', type=str, required=True, metavar='CLASS', help='Classification label')

    return parser
