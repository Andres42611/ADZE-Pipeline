#!/bin/bash
# File: datagen.sh
# Principal Investigator: Dr. Zachary Szpiech
# Date: 10 April 2024
# Author: Andres del Castillo
# Purpose: Using positional arguments (in same order as parser): conducts simulations for user-specified number of replicates, converts .trees files to .vcf and finally to .stru, 
# and cleans + saves given directory


# Script runs provided positional arguments:
#    $1: directory for storing data files (preferably empty)
#    $2: source population for migration BACKWARDS in time (read more at: https://tskit.dev/msprime/docs/stable/demography.html#migration)
#    $3: destination population for migration BACKWARDS in time (read more at: https://tskit.dev/msprime/docs/stable/demography.html#migration)
#    $4: Classification label for simulation case (e.g. A,B,C, etc.)
#    $5: File path to tab-delimited txt population file (column 1: sample ID, column 2: population ID (must be integer))
#    $6: Number of replicates to run

#Begin by running 100 replications of msprime sims
./simulation.py -D "$1" -s "$2" -d "$3" -C "$4" -r $6

#Initialize final dataset for case with header
> "$1/case$4_data.csv"

# Convert .trees files to .vcf and delete the .trees files for all replicates
for ((i = 1; i <= $6; i++)); do
    # First convert from .trees to .vcf
    python3 -m tskit vcf "$1/$4_rep_${i}.trees" > "$1/$4_rep_${i}.vcf"

    # Now process the VCF files into STRU format
    ./VCFtoSTRU.py -V "$1/$4_rep_${i}.vcf" -P "$5"

    # Move to ADZE analysis (ZA Szpiech, M Jakobsson, NA Rosenberg. (2008) ADZE: a rarefaction approach for counting alleles private to combinations of populations. Bioinformatics 24: 2498-2504.)
    # using both the paramfile and command line arguments (read more at: https://github.com/szpiech/ADZE)
    num_loci=$(awk 'NR==1{print gsub(/ /," ")+1; exit}' "$1/$4_rep_${i}.stru") #get number of loci from VCF
    ./ADZE-1.0/adze-1.0 sim_paramfile.txt -f "$1/$4_rep_${i}.stru" -l ${num_loci} -r "$1/$4_rep_${i}richness" -p "$1/$4_rep_${i}private" -o "$1/$4_rep_${i}comb" -pp 0 

    # After obtaining our ADZE analysis files, convert the replicate's data to a CSV dataset (making sure to add a _2 to the comb file name)
    ./ADZEtoCSV.py -r "$1/$4_rep_${i}richness" -p "$1/$4_rep_${i}private" -C "$4"

    if [ "$i" -eq 0 ]; then
        # For the first file, include the header
        cat "$1/$4_rep_${i}.csv" >> "$1/case$4_data.csv"
    else
        # For subsequent files, skip the header
        tail -n +2 "$1/$4_rep_${i}.csv" >> "$1/case$4_data.csv"
    fi

    # Finally clean the directory to only leave the final dataset
    rm "$1/$4_rep_${i}.trees" "$1/$4_rep_${i}.vcf" "$1/$4_rep_${i}.stru" "$1/$4_rep_${i}richness" "$1/$4_rep_${i}private" "$1/$4_rep_${i}comb_2" "$1/$4_rep_${i}richness_summary" "$1/$4_rep_${i}.csv"
done
