#!/bin/bash
# File: data_processing.sh
# Principal Investigator: Dr. Zachary Szpiech
# Date: 10 April 2024
# Author: Andres del Castillo
# Purpose: Using positional arguments (in same order as parser): conducts 


#make data directories if they don't exist, otherwise: continue
mkdir -p "$1/CaseA" "$1/CaseB" "$1/CaseC" "$1/CaseD" "$1/CaseE" "$1/StratData" 

./datagen.sh "$1/CaseA" B A A ./POP.txt     #Case A data generation
./datagen.sh "$1/CaseB" A B B ./POP.txt     #Case B data generation
./datagen.sh "$1/CaseC" B C C ./POP.txt     #Case C data generation
./datagen.sh "$1/CaseD" C B D ./POP.txt     #Case D data generation
./datagen.sh "$1/CaseE" "0" "0" E ./POP.txt #Case E data generation

python3 datastrat.py -D "$1"
