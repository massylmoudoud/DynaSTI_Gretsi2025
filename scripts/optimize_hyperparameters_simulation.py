# -*- coding: utf-8 -*-
"""
Created on Thr Jan 02 2025 2025 13∶30∶45

Copyright (c) 2025 University of Strasbourg
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

try:
    import mkl

    mkl.set_num_threads(1)
except:
    pass

import numpy as np
import multiprocessing as mp
import itertools
import sys
import os
import json
import gzip  # to compress the result
import argparse  # To set size of grid

# Set paths
import pathlib

try:
    script_path = str(pathlib.Path(__file__).parent.resolve()) + "/"
except NameError:
    script_path = ""


functionPath = script_path + "../src/"
sys.path.append(functionPath)

from utile.make_noisy_matrix import make_noise
from joint_estimation_A_D.joint_optimization_A_D import joint_estimation_A_D
from joint_estimation_A_D.joint_optimization_A_D_AMORS import joint_estimation_A_D_AMORS
from joint_estimation_A_D.joint_optimization_A_D_vector_AMORS import (
    joint_estimation_A_D_vector_AMORS,
)


# disable warnings
import warnings

warnings.filterwarnings("ignore")

from time import time


"""
Script to optimize the hyperparameters of FISTA for the joint estimaton of A and D on simulation data
"""


########################### Data ###############################################
# Load the simulation data
path = script_path + "../data/simulation/simulation_A_D_2024_12_18.npz"
data = np.load(path)
A_VT = data["A"]
D_VT = data["D"]

R, P = D_VT.shape
T = A_VT.shape[1]

D_tilde = np.int32(D_VT != 0)
C_VT = D_VT @ A_VT


##################################################################################################
##################################################################################################

max_iter = 500


def eval_joint_optim_A_D(
    lambda_L1_A,
    lambda_L2_D,
    lambda_TV,
    Cvec,
    D_init,
    C_VT=C_VT,
    A_VT=A_VT,
    D_VT=D_VT,
    D_tilde=D_tilde,
    max_iter=max_iter,
):
    start_time = time()

    # set max iterations
    maxIter_A = 500
    maxIter_D = 500

    param_A = {"lambda_l1": lambda_L1_A, "maxIter": maxIter_A, "lambda_TV": lambda_TV}

    param_D = {"lambda_l2": lambda_L2_D, "maxIter": maxIter_D}

    # handle error cases:
    nan = np.array(np.nan)

    # classical alternating minimization AM
    #print("classical AM")
    try:
        D_est, A_est, err = joint_estimation_A_D(
            Cvec,
            D_tilde,
            max_iter,
            param_D,
            param_A,
            D_init=D_init,
            ret_err=True,
            C_VT=C_VT,
            D_VT=D_VT,
            A_VT=A_VT,
        )
    except:
        # print("error in AM")
        D_est = nan
        A_est = nan
        err = nan
    # C_est = D_est@A_est

    # AMORS
    #print("AMORS")
    try:
        D_est_AMORS, A_est_AMORS, err_AMORS, gamma_AMORS = joint_estimation_A_D_AMORS(
            Cvec,
            D_tilde,
            max_iter,
            param_D,
            param_A,
            D_init=D_init,
            ret_err=True,
            C_VT=C_VT,
            D_VT=D_VT,
            A_VT=A_VT,
        )
    except:
        # print("error in AMORS")
        D_est_AMORS = nan
        A_est_AMORS = nan
        err_AMORS = nan
        gamma_AMORS = nan
    # C_est_AMORS = D_est_AMORS @ A_est_AMORS

    # vector AMORS (our proposed method)
    #print("VectAMORS")
    try:
        D_est_VectAMORS, A_est_VectAMORS, err_VectAMORS, gamma_VectAMORS = (
            joint_estimation_A_D_vector_AMORS(
                Cvec,
                D_tilde,
                max_iter,
                param_D,
                param_A,
                D_init=D_init,
                ret_err=True,
                C_VT=C_VT,
                D_VT=D_VT,
                A_VT=A_VT,
            )
        )
    except:
        # print("error in VectAMORS")
        D_est_VectAMORS = nan
        A_est_VectAMORS = nan
        err_VectAMORS = nan
        gamma_VectAMORS = nan
    # C_est_VectAMORS = D_est_VectAMORS @ A_est_VectAMORS

    # return result in json compatible format (convert arrays to lists of list to be serializable)
    return {
        "lambda_l1_A": lambda_L1_A,
        "lambda_l2_D": lambda_L2_D,
        "lambda_TV": lambda_TV,
        "D_est": D_est.tolist(),
        "A_est": A_est.tolist(),
        "err": err.tolist(),
        "D_est_AMORS": D_est_AMORS.tolist(),
        "A_est_AMORS": A_est_AMORS.tolist(),
        "err_AMORS": err_AMORS.tolist(),
        "gamma_AMORS ": gamma_AMORS.tolist(),
        "D_est_VectAMORS": D_est_VectAMORS.tolist(),
        "A_est_VectAMORS": A_est_VectAMORS.tolist(),
        "err_VectAMORS": err_VectAMORS.tolist(),
        "gamma_VectAMORS ": gamma_VectAMORS.tolist(),
    }


##################################################################################################
###########################    MAIN     ##########################################################


if __name__ == "__main__":
    mp.set_start_method("spawn")
    save_path = script_path + "./optimization_results/"
    N_processes = max(len(os.sched_getaffinity(0))//2, 1)

    ##################################################################################################
    # Check if full or reduced grid

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--full",
        help="Run optimization on full grid of parameters (slow excecution)",
        action="store_true",
    )

    parser.add_argument(
        "-r",
        "--reduced",
        help="Run optimization on reduced grid of parameters (fast excecution)",
        action="store_true",
    )

    args = parser.parse_args()

    if args.full and not args.reduced:
        print("Optimization on full grid")
        N_lambda_L1 = 12
        N_lambda_L2 = 12
        N_lambda_TV = 7
    else:  # args.reduced:
        print("Optimization on reduced grid")
        N_lambda_L1 = 6
        N_lambda_L2 = 6
        N_lambda_TV = 3
    ##################################################################################################
    # add noise to C
    snr = 0  # dB
    C_noise = make_noise(C_VT, snr)

    # Initialize D
    C_mean = np.mean(C_noise, axis=1)

    D_init = D_tilde * C_mean[:, np.newaxis]

    # Eval joint optimization A and D
    lambda_L1_min = 0.01
    lambda_L1_max = 0.8
    lambda_L1_A_vals = np.linspace(lambda_L1_min, lambda_L1_max, N_lambda_L1)
    lambda_L2_D_vals = np.linspace(lambda_L1_min, lambda_L1_max, N_lambda_L2)

    lambda_TV_min = 0.1
    lambda_TV_max = 1
    lambda_TV_vals = np.linspace(lambda_TV_min, lambda_TV_max, N_lambda_TV)

    param_A_D = itertools.product(
        lambda_L1_A_vals, lambda_L2_D_vals, lambda_TV_vals, [C_noise], [D_init]
    )

    chunksize = max((N_lambda_L1 * N_lambda_L2 * N_lambda_TV) // N_processes, 1)

    with mp.Pool(N_processes) as p:
        result_A_D = [
            i for i in p.starmap(eval_joint_optim_A_D, param_A_D, chunksize=chunksize)
        ]

    # use compressed json
    path = save_path + f"FISTA_result_A_D_SNR_{snr}.json.gz"
    with gzip.open(path, "wt", encoding="UTF-8") as zipfile:
        json.dump(result_A_D, zipfile)
