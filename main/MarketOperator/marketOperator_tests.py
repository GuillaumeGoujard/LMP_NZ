from main.Network.Topology.Topology import Topology
from main.Network.PriceBids.Load.Load import Load
from main.Network.PriceBids.Generator.Generator import Generator
from main.MarketOperator.marketOperator import marketOperator

import main.Network.PriceBids.Load.Load as ld

import pandas as pd
from cvxpy import *
import numpy as np


"""
One node test : 1 GEN, 1 LD
"""
#
# # Initializing topology, generator and load
# ONDE = Topology(network = "ONDE")
# GEN = Generator("GEN", "NDE", 0, "dummy", Pmax=20, Pmin=0, marginal_cost=10)
# LD = Load("LD", "NDE", 0, "dummy", 10)
# ONDE_market = marketOperator
#
# # Adding generator and load to topology
# ONDE.add_generator(GEN)
# ONDE.add_load(LD)
#
# # Updating Topology characteristics
# ONDE.create_Mn()
# ONDE.create_Pmin_Pmax()
# ONDE.create_Ag_qg()
# ONDE.create_Qt_at()
# ONDE.create_Au_xt_ct()
#
# # Adding characteristics to market
# ONDE_market.Topology = ONDE
# ONDE_market.Loads = LD
# ONDE_market.Generators = GEN
#
# # Clearing market
# marketClear = ONDE_market.clear_market
#
# Qt, at = ONDE_market.Topology.Qt, ONDE_market.Topology.at
# H, h = ONDE_market.Topology.H, ONDE_market.Topology.h
# Ag, qg = ONDE_market.Topology.Ag, ONDE_market.Topology.qg
# Au, xt, ct = ONDE_market.Topology.Au, ONDE_market.Topology.xt, ONDE_market.Topology.ct
# Mn = ONDE_market.Topology.Mn
# dt = np.array([[ONDE_market.Loads.d]])  # Just for the purpose of this example
#
# # Defining sizes
# n = ONDE_market.Topology.number_nodes
# g = ONDE_market.Topology.number_generators
#
# # Defining optimization variables
# pt = np.array([[0]]) #Variable((n,1))
# gt = Variable((g,1))
# ut = np.array([[0]]) #Variable((n,1))
#
# # Defining objective function
# objective = Minimize(1/2 * gt.T @ Qt @ gt + at.T @ gt + ct.T @ ut)
#
# # Defining constraints
# constraints = [] # Initializing
#
# constraints += [np.ones((n,1)).T@pt == 0]
#
# # constraints += [H@pt <= h]
# constraints += [Ag@gt <= qg]
# constraints += [Au@ut <= xt]
# constraints += [pt == Mn@gt + ut - dt]
#
# # Clearing the market
# marketClear = Problem(objective, constraints)
# marketClear.solve()


# ONDE.

# """
# One node test : 2 GEN, 3 LD
# """
# # Creating skeleton network
# ONDE = Topology(network = "ONDE")
#
# # Adding generator and load
# GEN = Generator("GEN", "NDE", 0, "dummy", Pmax=20, Pmin=0, marginal_cost=10)
# ONDE.add_generator(GEN)
# LD = Load("LD", "NDE", 0, "dummy", 10)
# ONDE.add_load(LD)
#
# # Updating Topology characteristics
# ONDE.create_H_h()
# ONDE.create_Mn()
# ONDE.
#
"""
Two node test : 1 GEN on NTH, 1 LD on STH
"""
# Initializing topology, generator and load
NSNDE = Topology(network = "NSNDE")
GEN = Generator("GEN", "NTH", 0, "dummy", Pmax=20, Pmin=0, marginal_cost=10)
LD = Load("LD", "STH", 0, "dummy", 10)
NSNDE_market = marketOperator

# Adding generator and load to topology
NSNDE.add_generator(GEN)
NSNDE.add_load(LD)

# Updating Topology characteristics
NSNDE.create_Mn()
NSNDE.create_Pmin_Pmax()
NSNDE.create_Ag_qg()
NSNDE.create_Qt_at()
NSNDE.create_Au_xt_ct()

# Adding characteristics to market
NSNDE_market.Topology = NSNDE
NSNDE_market.Loads = LD
NSNDE_market.Generators = GEN

# Clearing market
NSNDE_market.clear_market



# """
# Two node test : 1 GEN/1 LD on NTH, 1 GEN/2 LD on STH
# """
# # Creating skeleton network
# ONDE = Topology(network = "ONDE")
#
# # Adding generator and load
# GEN = Generator("GEN", "NDE", 0, "dummy", Pmax=20, Pmin=0, marginal_cost=10)
# ONDE.add_generator(GEN)
# LD = Load("LD", "NDE", 0, "dummy", 10)
# ONDE.add_load(LD)
#
# # Updating Topology characteristics
# ONDE.create_H_h()
# ONDE.create_Mn()
# ONDE.
#
# """
# ABM TEST
# """



