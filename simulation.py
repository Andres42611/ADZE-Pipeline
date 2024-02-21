#!/usr/bin/env python3
import msprime
import argparse
import numpy as np
from parser import create_parser

#initialize parser
parsed = create_parser()
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

#set incremental migration rates for each replicate
migration_rates = np.linspace(0.05, 0.5, 100)

#Go through 100 simulation replicates, each with incremented migration
for rep in range(100):
  demography.set_migration_rate(source=args.source, dest=args.dest, rate=migration_rates[rep])
  
  #simulate coalescence
  ts = msprime.sim_ancestry(samples={"A": 200, "B": 200, "C": 200},
                          sequence_length=1e6,  # 1 Mbp
                          recombination_rate=2e-8,
                          ploidy=2,
                          demography=demography)
  
  #Sprinkle tolerant mutations
  finalrep = msprime.sim_mutations(ts, rate=1e-7, keep=True)

  #Save as .trees file
  path_to_save = args.direc + "/" + str(args.Case) + "_rep_" + str(rep) + ".trees"
  finalrep.dump(path_to_save)
