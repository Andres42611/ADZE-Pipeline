# Principal Investigator: Dr. Zachary Szpiech
# Date: 10 April 2024
# Author: Andres del Castillo
# Purpose: Makefile for data generation + data processing of ML Pipeline

# Base directory for data (passed as argument)
BASEDIR ?= /path/to/data

# Target directories
DIRS := $(BASEDIR)/CaseA $(BASEDIR)/CaseB $(BASEDIR)/CaseC $(BASEDIR)/CaseD $(BASEDIR)/CaseE $(BASEDIR)/StratData

# POP.txt path
POP_TXT := $(BASEDIR)/POP.txt

# Default target
all: data_dirs caseA caseB caseC caseD caseE datastrat

# Create data directories
data_dirs:
	@mkdir -p $(DIRS)

caseA: data_dirs
	./datagen.sh "$(BASEDIR)/CaseA" B A A $(POP_TXT)

caseB: data_dirs
	./datagen.sh "$(BASEDIR)/CaseB" A B B $(POP_TXT)

caseC: data_dirs
	./datagen.sh "$(BASEDIR)/CaseC" B C C $(POP_TXT)

caseD: data_dirs
	./datagen.sh "$(BASEDIR)/CaseD" C B D $(POP_TXT)

caseE: data_dirs
	./datagen.sh "$(BASEDIR)/CaseE" "0" "0" E $(POP_TXT)

datastrat: caseA caseB caseC caseD caseE
	python3 datastrat.py -D "$(BASEDIR)"

.PHONY: all data_dirs caseA caseB caseC caseD caseE datastrat
