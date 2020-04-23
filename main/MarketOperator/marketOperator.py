from datetime import datetime as datetime
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from main.Network.PriceBids.Generator.Generator import Generator
from main.Network.PriceBids.Load.Load import Load
import os
from cvxpy import *

from main.Network.Topology.Topology import Topology
from main.Network.PriceBids.Load.Load import Load
from main.Network.PriceBids.Generator.Generator import Generator

class marketOperator:
    def __init__(self, network):
        if network is "One node":
            self.name = None
            self.type = None
            self.Bids = None
            self.Offers = None
            self.Topology = None
            self.LMP = None

        if network is "North-South node":
            self.name = None
            self.type = None
            self.Bids = None
            self.Offers = None
            self.Topology = None
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


    def Q(self, ):
        pass



    def market_clearing(self):

        # Importing characteristics
        Q_t =
        a_t =
        c_t =
        H =
        h =
        A_g =
        q_g =
        A_u =
        x_t =
        M_n =
        d_t =

        # Defining optimization variables
        p_t = Variable(???)
        g_t = Variable(???)
        # u_t =

        # Defining objective function
        objective = Minimize(1/2 * g_t.T @ Q_t @ g_t + a_t.T @ g_t + c_t.T @ u_t)

        # Defining constraints
        constraints = [] # Initializing

        constraints += [np.ones((???,1)).T@p_t == 0]

        constraints += [H@p_t <= h]
        constraints += [A_g@g_t <= q_g]
        constraints += [A_u@u_t <= x_t]
        constraints += [p_t == M_n@g_t + u_t - d_t]