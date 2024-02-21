#!/bin/bash
#Using positional arguments (in same order as parser): conducts simulations, converts .trees files to .vcf and finally to .stru, 
#and cleans + saves given directory


# Run simulation with provided arguments
./simulation.py -D "$1" -s "$2" -d "$3" -C "$4"

# Convert .trees files to .vcf and delete the .trees files
for i in {0..99}; do
    python3 -m tskit vcf "$1/$4_rep_${i}.trees" > "$1/$4_rep_${i}.vcf"
    rm "$1/$4_rep_${i}.trees"
done

# Process the VCF files into STRU format
./VCFtoSTRU.py -P "$5"
