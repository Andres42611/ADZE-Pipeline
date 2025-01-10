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
echo "Class,Replicate,Pops,D,Z,p,f4,BBAA,ABBA,BABA" > "$1/case$4_DSTAT.csv"

# Read m_rates.txt and create a flattened array of floats
M_RATES=($(< "$1/mrates.txt"))
export M_RATES

# Get the length of the M_RATES array
M_RATES_LENGTH=${#M_RATES[@]}

# Loop based on the number of replicates and the length of M_RATES
for ((i = 1; i <= $6 * M_RATES_LENGTH; i++)); do
    # First convert from .trees to .vcf
    python3 -m tskit vcf "$1/$4_rep_${i}.trees" > "$1/$4_rep_${i}.vcf"

    # Now process the VCF files into STRU format
    ./VCFtoSTRU.py -V "$1/$4_rep_${i}.vcf" -P "$5"

    # Move to ADZE analysis (ZA Szpiech, M Jakobsson, NA Rosenberg. (2008) ADZE: a rarefaction approach for counting alleles private to combinations of populations. Bioinformatics 24: 2498-2504.)
    # using both the paramfile and command line arguments (read more at: https://github.com/szpiech/ADZE)
    num_loci=$(awk 'NR==1{print gsub(/ /," ")+1; exit}' "$1/$4_rep_${i}.stru") #get number of loci from VCF
    ./ADZE-1.0/adze-1.0 sim_paramfile.txt -f "$1/$4_rep_${i}.stru" -l ${num_loci} -r "$1/$4_rep_${i}richness" -p "$1/$4_rep_${i}private" -o "$1/$4_rep_${i}comb" -pp 0 

    # Calculate the index for M_RATES
    mx=$((($i - 1) / $6))
    rate=${M_RATES[$mx]}
    # After obtaining our ADZE analysis files, convert the replicate's data to a CSV dataset
    ./ADZEtoCSV.py -r "$1/$4_rep_${i}richness" -p "$1/$4_rep_${i}private" -C "$4" -R $6 -m "$rate"

    if [ "$i" -eq 1 ]; then
        # For the first file, include the header
        cat "$1/$4_rep_${i}.csv" >> "$1/case$4_data.csv"
    else
        # For subsequent files, skip the header
        tail -n +2 "$1/$4_rep_${i}.csv" >> "$1/case$4_data.csv"
    fi

    /Users/ard/Desktop/ADZE_Pipeline/Dsuite/Build/Dsuite Dtrios "$1/$4_rep_${i}.vcf" /Users/ard/Desktop/ADZE_Pipeline/3pop/DSTATPOP.txt -o "$1/$4rep_${i}"
      # Check if the output file exists
    bbaa_file="$1/$4rep_${i}_BBAA.txt"
    if [[ -f "$bbaa_file" ]]; then
        # Extract the relevant line from the BBAA file (skip the header)
        read -r P1 P2 P3 D Z p f4 BBAA ABBA BABA < <(tail -n 1 "$bbaa_file")

        # Extract the last characters of P1, P2, P3 for the Pops field
        pops="${P1: -1}${P2: -1}${P3: -1}"

        # Append the results to the CSV file
        echo "$4,$i,$pops,$D,$Z,$p,$f4,$BBAA,$ABBA,$BABA" >> "$1/case$4_DSTAT.csv"
    else
        echo "Warning: $bbaa_file not found for $1/$4_rep_${i}.vcf" >&2
        fi

    # Finally clean the directory to only leave the final dataset
    rm "$4rep_${i}_BBAA.txt" "$1/$4_rep_${i}.vcf" "$1/$4_rep_${i}.trees" "$1/$4_rep_${i}.stru" "$1/$4_rep_${i}richness" "$1/$4_rep_${i}private" "$1/$4_rep_${i}comb_2" "$1/$4_rep_${i}richness_summary" "$1/$4_rep_${i}.csv"
done
