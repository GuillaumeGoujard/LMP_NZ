import numpy as np
from cvxpy import *

from lmpnz.Network.Topology.Topology import Topology


class marketOperator:
    def __init__(self, network = ""):
        if network is "ONDE":
            self.name = None
            self.type = None
            self.Bids = None
            self.Offers = None
            self.Loads = None
            self.Generators = None
            self.Topology = Topology(network = 'One node')
            self.LMP = None



        if network is "NSNDE":
            self.name = None
            self.type = None
            self.Bids = None
            self.Offers = None
            self.Topology = Topology(network='North-South node')
            self.LMP = None

        if network is "ABM":
            self.name = 'SPD'
            self.type = None
            self.Bids = None
            self.Offers = None
            self.Topology = None
            self.LMP = None

        else:
            self.name = None
            self.type = None
            self.Bids = None
            self.Offers = None
            self.Topology = None
            self.LMP = None

    ## Methods

    # Adding elements
    def add_bid(self):
        return

    def add_offer(self):
        return

    # Creating elements


    # Performing actions
    def Q(self, ):
        pass

    def one_node_market_clearing(self):

        # Importing characteristics
        Qt, at = self.Topology.Qt, self.Topology.at
        H, h = self.Topology.H, self.Topology.h
        Ag, qg = self.Topology.Ag, self.Topology.qg
        Au, xt, ct = self.Topology.Au, self.Topology.xt, self.Topology.ct
        Mn = self.Topology.Mn
        dt = 0

        # Defining sizes
        n = 0
        g = 0

        # Defining optimization variables
        pt = Variable((n, 1))
        gt = Variable((g, 1))
        ut = Variable((n, 1))

        # Defining objective function
        objective = Minimize(1 / 2 * gt.T @ Qt @ gt + at.T @ gt + ct.T @ ut)

        # Defining constraints
        constraints = []  # Initializing

        constraints += [np.ones((n, 1)).T @ pt == 0]

        constraints += [H @ pt <= h]
        constraints += [Ag @ gt <= qg]
        constraints += [Au @ ut <= xt]
        constraints += [pt == Mn @ gt + ut - dt]

        # Clearing the market
        marketClear = Problem(objective, constraints)
        marketClear.solve()

        return pt, gt, ut

    def clear_market(self):

        # Importing characteristics
        Qt, at = self.Topology.Qt, self.Topology.at
        H, h = self.Topology.H, self.Topology.h
        Ag, qg = self.Topology.Ag, self.Topology.qg
        Au, xt, ct = self.Topology.Au, self.Topology.xt, self.Topology.ct
        Mn = self.Topology.Mn
        dt = self.Loads.d

        # Defining sizes
        n = self.Topology.number_nodes
        g = self.Topology.number_generators

        # Defining optimization variables
        pt = Variable((n,1))
        gt = Variable((g,1))
        ut = Variable((n,1))

        # Defining objective function
        objective = Minimize(1/2 * gt.T @ Qt @ gt + at.T @ gt + ct.T @ ut)

        # Defining constraints
        constraints = [] # Initializing

        constraints += [np.ones((n,1)).T@pt == 0]

        constraints += [H@pt <= h]
        constraints += [Ag@gt <= qg]
        constraints += [Au@ut <= xt]
        constraints += [pt == Mn@gt + ut - dt]

        # Clearing the market
        marketClear = Problem(objective, constraints)
        marketClear.solve()

        # Outputting resolution
        self.pt = pt
        self.gt = gt
        self.ut = ut

        return self.pt, self.gt, self.ut