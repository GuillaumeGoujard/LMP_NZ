import numpy as np
import pandas as pd

def H_matrix(self, NLeave, NEnter, y):
    '''
    Shift Factor matrix

    Args:
        NLeave: Leaving node vector
            Size: (m,)
            Type: np.array
            Unit: integer value
            Description: List of nodes through which line of index l leaves

        NEnter: Entering node vector
            Size: (m,)
            Type: np.array
            Unit: integer value
            Description: List of nodes through which line of index l enters

        y: Impedance vector
            Size: (m,)
            Type: np.array
            Unit: Ohms
            Description: Array of impedances of each line, in Ohms. Each row corresponds to the indice of line l.

    Returns:
        H: Shift factor matrix
            Size: (2*m,n)
            Type: np.array
            Unit: ???
            Description:

    '''
    # Initializing sizes
    m = NLeave.size
    n = np.unique(np.concatenate((NLeave, NEnter))).size

    N = np.unique(np.concatenate((NLeave, NEnter)))

    # Initializing M
    M = np.zeros((n, m))

    # Creating M
    for i in range(n):
        for l in range(m):
            if NLeave[l] == N[i]:
                M[i, l] = 1
            elif NEnter[l] == N[i]:
                M[i, l] = -1

    Delta_y = np.diag(y)

    Y = M @ Delta_y @ M.T

    Y_bar = np.delete(np.delete(Y, 0, 0), 0, 1)

    Y_dag = np.concatenate((np.zeros((1, Y_bar.shape[0] + 1)),
                            np.concatenate((np.zeros((Y_bar.shape[0], 1)),
                                            np.linalg.inv(Y_bar)),
                                           axis=1)),
                           axis=0)

    H_hat = np.diag(y) @ M.T @ Y_dag

    H = np.concatenate((np.eye(m), -np.eye(m)), axis=0) @ H_hat

    return H


Network = pd.read_csv("../../data/ABM/ABM_Network_details.csv") # PATH TO BE CORRECTED TO CORRECTLY REFERENCE CSVs!!!

Nodes = np.unique(np.concatenate((np.unique(Network.LEAVE), np.unique(Network.ENTER))))

Network['NLeave'] = np.array([np.where(Nodes == Network['LEAVE'][l])[0][0] for l in range(m)])
Network['NEnter'] = np.array([np.where(Nodes == Network['ENTER'][l])[0][0] for l in range(m)])

z = np.sqrt(Network['Resistance (Ohms)'] ** 2 + Network['Reactance (Ohms)'] ** 2)

y = 1 / z

H = H_matrix(Network['NLeave'], Network['NEnter'], y)

np.set_printoptions(precision=2)
print(H.shape)
print(H)
