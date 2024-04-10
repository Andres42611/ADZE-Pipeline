#!/usr/bin/env python3
# File: simulation.py
# Principal Investigator: Dr. Zachary Szpiech
# Date: 10 April 2024
# Author: Andres del Castillo
# Purpose: MSPRIME simulation script for given demography over 100 replicates

import msprime
import argparse
import numpy as np
from parser import init_Simulation_Parser

#initialize parser
parsed = init_Simulation_Parser()
args = parsed.parse_args()

#initialize demography population splits
demography = msprime.Demography()

demography.add_population(name="A", initial_size=10_000)
demography.add_population(name="B", initial_size=10_000)
demography.add_population(name="C", initial_size=10_000)
demography.add_population(name="AB", initial_size=10_000)
demography.add_population(name="ABC", initial_size=10_000)

demography.add_population_split(time=100, derived=["A", "B"], ancestral="AB")
demography.add_population_split(time=200, derived=["AB", "C"], ancestral="ABC")

#Check if migraton occurs:
migration = (args.source != '0' and args.dest != '0')

#Go through args.repnum number of simulation replicates
for rep in range(1, args.repnum+1):
  #if migration will occur
  if migration:
    demography.set_migration_rate(source=args.source, dest=args.dest, rate=0.25) #if migration, use fixed migration rate
  
  #simulate coalescence
  ts = msprime.sim_ancestry(samples={"A": 200, "B": 200, "C": 200},
                          sequence_length=1e6,  # 1 Mbp
                          recombination_rate=1.78e-8,
                          ploidy=2,
                          demography=demography)
  #Sprinkle tolerant mutations
  finalrep = msprime.sim_mutations(ts, rate=2e-8, keep=True)

  #Save as .trees file
  path_to_save = args.direc + "/" + str(args.Case) + "_rep_" + str(rep) + ".trees"
  finalrep.dump(path_to_save)
