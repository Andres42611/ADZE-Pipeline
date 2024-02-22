#!/bin/bash
# File: Pipeline.sh
# Principal Investigator: Dr. Zachary Szpiech
# Date: 21 February 2024
# Author: Andres del Castillo
# Purpose: Using positional arguments (in same order as parser): conducts simulations, converts .trees files to .vcf and finally to .stru, 
# and cleans + saves given directory


# Script runs provided positional arguments:
#    $1: directory for storing data files (preferably empty)
#    $2: source population for migration BACKWARDS in time (read more at: https://tskit.dev/msprime/docs/stable/demography.html#migration)
#    $3: destination population for migration BACKWARDS in time (read more at: https://tskit.dev/msprime/docs/stable/demography.html#migration)
#    $4: Classification label for simulation case (e.g. A,B,C, etc.)
#    $5: File path to tab-delimited txt population file (column 1: sample ID, column 2: population ID (must be integer))

#Begin by running 100 replications of msprime sims
./simulation.py -D "$1" -s "$2" -d "$3" -C "$4"

# Convert .trees files to .vcf and delete the .trees files
for i in {0..99}; do
    #First convert from .trees to .vcf
    python3 -m tskit vcf "$1/$4_rep_${i}.trees" > "$1/$4_rep_${i}.vcf"

    # Now process the VCF files into STRU format
    ./VCFtoSTRU.py -V "$1/$4_rep_${i}.vcf" -P "$5"

    num_loci=$(awk 'NR==1{print gsub(/ /," ")+1; exit}' "$1/$4_rep_${i}.stru")
    ./ADZE-1.0/adze-1.0 sim_paramfile.txt -f "$1/$4_rep_${i}.stru" -l ${num_loci} -r "$1/$4_rep_${i}richness.txt" -p "$1/$4_rep_${i}private.txt" -o "$1/$4_rep_${i}comb.txt" 

    ./
    #Finally clean the directory to only leave .stru file
    rm "$1/$4_rep_${i}.trees"
    rm "$1/$4_rep_${i}.vcf"
done
