"""
Copyright (c) 2024 University of Strasbourg
Author: Massyl Moudoud <mmoudoud@unistra.fr>
Contributor(s) : Céline Meillier <meillier@unistra.fr>, Vincent Mazet <vincent.mazet@unistra.fr>

This work has been supported by the ANR project DynaSTI: ANR-22-CE45-0008

This software is governed by the CeCILL  license under French law and
abiding by the rules of distribution of free software.  You can  use,
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability.

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

import scipy as sp

import sys

functionPath = "../src/"
sys.path.append(functionPath)

from utile.make_noisy_matrix import make_noise


def state_generator(R, li):
    """
    R : nb of ROIs
    li : list of list of size 3 : [ROI1, ROI2, coef]
    """
    C = np.zeros((R, R))
    for l in li:
        ROI1 = l[0]
        ROI2 = l[1]
        coeff = l[2]
        C[ROI1, ROI2] = coeff
        C[ROI2, ROI1] = coeff
    return C


def FCU_generator(R, li, plot=False):
    ROI_name = [f"{i}" for i in range(R)]
    etat = state_generator(R, li)
    df = pd.DataFrame(data=etat, columns=ROI_name, index=ROI_name)
    G = nx.from_pandas_adjacency(df)
    weights = [G[u][v]["weight"] * 4 for u, v in G.edges()]
    M = G.number_of_edges()
    edge_colors = [G[u][v]["weight"] for u, v in G.edges()]

    if plot:
        plt.figure(figsize=(6, 6))
        nx.draw_networkx(
            G,
            with_labels=True,
            # pos=position,
            width=weights,
            edge_color=edge_colors,
            # edge_cmap=cmap,
            edge_vmin=-1,
            edge_vmax=1,
        )
    return etat, df, G


def get_dictionary():
    """
    returns the dictionary from hard coded FCUs (networks and subnetworks)
    """
    R = 10  # number of ROIs
    # Réseau 1a

    li1a = [
        [0, 1, 0.74],
        [0, 2, 0.8],
        [0, 3, 0.82],
        [1, 2, 0.72],
        [1, 3, 0.69],
        [2, 3, 0.83],
    ]
    fcu1a, df1a, G1a = FCU_generator(R, li1a)

    li1b = [
        [5, 6, 0.65],
        [5, 7, 0.7],
        [5, 8, 0.67],
        [6, 7, 0.71],
        [6, 8, 0.69],
        [7, 8, 0.71],
    ]
    fcu1b, df1b, G1b = FCU_generator(R, li1b)

    li1 = (
        li1a
        + li1b
        + [[9, 8, 0.13], [9, 5, 0.11], [4, 6, 0.14], [4, 5, 0.19], [4, 0, 0.13]]
    )
    fcu1, df1, G1 = FCU_generator(R, li1)

    li2a = [
        [2, 3, 0.87],
        [2, 4, 0.62],
        [2, 5, 0.62],
        [3, 4, 0.53],
        [3, 5, 0.45],
        [4, 5, 0.89],
    ]
    FCU2a, df2a, G2a = FCU_generator(R, li2a)

    li2b = [
        [0, 9, 0.58],
        [0, 7, 0.68],
        [0, 8, 0.40],
        [9, 7, 0.71],
        [9, 8, 0.91],
        [7, 8, 0.65],
    ]
    FCU2b, df2b, G2b = FCU_generator(R, li2b)

    li2 = li2a + li2b
    FCU2, df2, G2 = FCU_generator(R, li2)

    # Creating D
    # Here we consider each network and its subnetworks as FCUs.
    # The matrix D contrans the weights of the edges of each pair of ROIs in a network or subnetwork

    FCU_list = ["1", "1a", "1b", "2", "2a", "2b"]

    # dictionary of networks as keys and subnewtorks as values (expressed as indexes of the FCU_list)
    subnetwork_idx = {0: [1, 2], 3: [4, 5]}
    P = len(FCU_list)

    D = np.zeros((int(R * (R - 1) / 2), P))

    idx = np.tril(np.ones((R, R)), -1) > 0

    for i, FCU in enumerate(FCU_list):
        pairs_list = eval("li" + FCU)
        C_FCU = np.zeros((R, R))
        for edge in pairs_list:
            C_FCU[edge[0], edge[1]] = edge[2]
            C_FCU[edge[1], edge[0]] = edge[2]

        D[:, i] = C_FCU[idx]

    D_tilde = np.int32(D > 0)

    return D, D_tilde, subnetwork_idx


def generate_data(T, snr):
    """
    generate sunthetic data according to the model C = DA and add noise to a given snr level
    Inputs:
        E: number of pairs of ROI
        P: Number of FCUs
        T: Number of time points
    """

    # Get dictionary D

    D, D_tilde, subnetwork_idx = get_dictionary()

    E, N_networks = D.shape
    P = N_networks
    # initialization
    A = np.zeros((N_networks, T))

    # generate the activation probability at the start of the recording
    p = 0.5  # probability of a region to be active at the start of the recording
    initial_states = np.random.default_rng().binomial(1, p, size=N_networks)
    A[:, 0] = initial_states

    for i in range(N_networks):
        # generate the change time indicies
        durations = np.ceil(np.random.default_rng().uniform(40, 150, T // 100))
        indices = np.cumsum(durations)
        indices = indices[indices < T]

        for j in range(1, T):
            if np.isin(j, indices):  # at transition
                A[i, j] = 1 - A[i, j - 1]  # if 0 become 1 and if 1 becomes 0
            else:
                A[i, j] = A[i, j - 1]

    # check that when a network is active, none of its subnetworks is active simultanously

    for i in range(T):
        for n in subnetwork_idx.keys():
            # if all the subnetworks are active, activate the network
            if np.all(A[subnetwork_idx[n], i]):
                A[n, i] = 1

            # if network active deactivate all its subnetworks
            if A[n, i] == 1:
                A[subnetwork_idx[n], i] = 0

    # save the support of A
    A_tilde = A.copy()

    # remove too short activations
    active_min = 20

    transitions = np.abs(np.diff(A_tilde, axis=1, prepend=A_tilde[:, 0, np.newaxis]))
    for p in range(N_networks):  # for each FCU
        change_idx = np.where(transitions[p, :])[0]
        Nb_change = len(change_idx)
        for i in range(Nb_change - 1):  # discard last percept (stopped simulation)
            idx = change_idx[i]
            active_time = change_idx[i + 1] - change_idx[i]

            if active_time < active_min:
                A_tilde[p, idx : idx + active_time] = A_tilde[p, idx - 1]

    # Check that all FCU are active at least once
    if not np.all(np.count_nonzero(A_tilde, axis=1) > 0):
        # just call the function again
        # This is not propbable to happen so there litle risc on performance
        return generate_data(T, snr)

    # Add variation to A
    noise = np.random.default_rng().normal(0.7, 0.5, size=(P, T + T // 10))

    # apply a lowpass butterworth filter
    sos = sp.signal.butter(N=15, Wn=0.08, btype="low", output="sos")

    filtered = sp.signal.sosfilt(sos, noise, axis=1)
    # remove first samples
    filtered = filtered[:, T // 10 :]
    # apply mask
    A_noise = filtered * A_tilde

    # enforce C<1

    def loss(A, A_noise):
        return np.linalg.norm(A - A_noise, ord=2) ** 2 / 2

    def jac(A, A_noise):
        return A - A_noise

    def hess(A, A_noise):
        return np.eye(A.shape[0])

    cons1 = {"type": "ineq", "fun": lambda A, D: 0.9 - D @ A, "args": (D,)}

    cons2 = {"type": "ineq", "fun": lambda A: A, "jac": lambda A: np.eye(A.shape[0])}

    opt = {"maxiter": 3, "disp": False}

    A = np.zeros((N_networks, T))

    for i in range(T):
        res = sp.optimize.minimize(
            loss,
            A_noise[:, i],
            args=A_noise[:, i],
            jac=jac,
            hess=None,
            constraints=[cons1, cons2],
            method="SLSQP",
            options=opt,
        )

        A[:, i] = res.x

    A = A * A_tilde

    # compute C
    C = D @ A

    # compute C_noise
    C_noise = make_noise(C, snr)

    return C_noise, C, A, D, D_tilde
