import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='Calculate the R(A,B) ratio')

    # Files and mandatory inputs for MSPRIME simulation
    parser.add_argument('-D', '--direc', type=str, required=False, metavar='SAVING_DIRECTORY', help='Directory to save simulation .tree files')
    parser.add_argument('-s', '--source', type=str, required=False, metavar='SOURCE_MIGRATION_POP', help='Source population for migration')
    parser.add_argument('-d', '--dest', type=str, required=False, metavar='DESTINATION_MIGRATION_POP', help='Destination population for migration')
    parser.add_argument('-C', '--Case', type=str, required=False, metavar='MIGRATION_CASE', help='Migration Case for Study')
    
    #Files and mandatory inputs for VCF to STRU file conversion
    parser.add_argument('-P', '--POP', type=str, required=False, metavar='POPULATION_FILE_PATH', help='Path to tab-separated population file')

    return parser
