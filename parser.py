#!/usr/bin/env python3


import argparse
    
def init_Simulation_Parser():
    parser = argparse.ArgumentParser(description='msprime Simulation parameters')
    
    # Files and mandatory inputs for MSPRIME simulation
    parser.add_argument('-D', '--direc', type=str, required=True, metavar='SAVING_DIRECTORY', help='Directory to save simulation .tree files')
    parser.add_argument('-s', '--source', type=str, required=True, metavar='SOURCE_MIGRATION_POP', help='Source population for migration')
    parser.add_argument('-d', '--dest', type=str, required=True, metavar='DESTINATION_MIGRATION_POP', help='Destination population for migration')
    parser.add_argument('-C', '--Case', type=str, required=True, metavar='MIGRATION_CASE', help='Migration Case for Study')
    
    return parser
    
def init_VCFtoSTRU_Parser():
    parser = argparse.ArgumentParser(description='msprime Simulation parameters')

    # Files and mandatory inputs for converting VCF file to .stru file
    parser.add_argument('-V', '--VCF', type=str, required=True, metavar='VCF_FILE', help='VCF file for conversion')
    parser.add_argument('-P', '--POP', type=str, required=True, metavar='POPULATION_FILE', help='Population tab-delimited txt file with IDs and Pop IDs')

    return parser