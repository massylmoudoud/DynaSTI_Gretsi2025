# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 2025 14∶54∶45

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
import argparse  # To set number of realization

# disable warnings
import warnings

warnings.filterwarnings("ignore")

# Set paths
import pathlib

try:
    script_path = str(pathlib.Path(__file__).parent.resolve()) + "/"
except NameError:
    script_path = ""

functionPath = script_path + "../src/"
sys.path.append(functionPath)


from utile.generation_data import generate_data
from joint_estimation_A_D.joint_optimization_A_D import joint_estimation_A_D
from joint_estimation_A_D.joint_optimization_A_D_AMORS import joint_estimation_A_D_AMORS
from joint_estimation_A_D.joint_optimization_A_D_vector_AMORS import (
    joint_estimation_A_D_vector_AMORS,
)


from time import time


"""
Script to optimize the hyperparameters of FISTA for the joint estimaton of A and D on simulation data
"""


########################### Data ###############################################
# Load the simulation data
path = (
    script_path + "../data/simulation/simulation_A_D_2024_12_18.npz"
)
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
    params: dict, input_matrices: dict, D_VT=D_VT, D_tilde=D_tilde, max_iter=max_iter
):
    # Get params
    lambda_L1_A = params["lambda_L1_A"]
    lambda_L2_D = params["lambda_L2_D"]
    lambda_TV = params["lambda_TV"]
    param_label = params["label"]

    # Get data
    Cvec = input_matrices["Cvec"]
    A_VT = input_matrices["A_VT"]
    C_VT = D_VT @ A_VT

    # Initialize D
    C_mean = np.mean(Cvec, axis=1)

    D_init = D_tilde * C_mean[:, np.newaxis]

    # set max iterations
    maxIter_A = 500
    maxIter_D = 500

    param_A = {"lambda_l1": lambda_L1_A, "maxIter": maxIter_A, "lambda_TV": lambda_TV}

    param_D = {"lambda_l2": lambda_L2_D, "maxIter": maxIter_D}

    # classical alternating minimization AM

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

    # AMORS

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

    # vector AMORS (our proposed method)

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

    # return result in json compatible format (convert arrays to lists of list to be serializable)
    return {
        "param_label": param_label,
        "A_VT": A_VT.tolist(),
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
    # Get number of realizations
    parser = argparse.ArgumentParser()
    
    parser.add_argument("realizations", help="Number of realization",
                    type=int,  nargs='?', default = 100)
                    
    args = parser.parse_args()    
    
    N_realizations = args.realizations
    
    #if N_realizations < N_processes:
    #    N_processes = N_realizations
         ##################################################################################################
    # Generate data
    
    # Signal length
    T = 1000
    snr = 0  # dB

    input_data = list()

    for i in range(N_realizations):
        C_noise, _, A, _, _ = generate_data(T, snr)

        input_data.append({"Cvec": C_noise, "A_VT": A})

    ##################################################################################################
    # Set params
    params = list()

    # Best AM
    params.append(
        {"label": "AM-AMORS", "lambda_L2_D": 0.8, "lambda_L1_A": 0.08, "lambda_TV": 1}
    )

    # Best AMORS
    # Same as AM
    # params.append({"label": "AMORS",
    #               "lambda_L2_D": 0.8,
    #               "lambda_L1_A": 0.08,
    #               "lambda_TV": 1
    #               })

    # Best Vect AMORS
    params.append(
        {
            "label": "VectAMORS",
            "lambda_L2_D": 0.66,
            "lambda_L1_A": 0.01,
            "lambda_TV": 0.4,
        }
    )

    # Average
    params.append(
        {"label": "mean", "lambda_L2_D": 0.75, "lambda_L1_A": 0.058, "lambda_TV": 0.8}
    )
    ##################################################################################################

    param_A_D = itertools.product(params, input_data)

    chunksize = max((len(params) * N_realizations) // N_processes, 1)

    with mp.Pool(N_processes) as p:
        result_A_D = [
            i for i in p.starmap(eval_joint_optim_A_D, param_A_D, chunksize=chunksize)
        ]

    # use compressed json
    path = save_path + f"FISTA_multi_realisation_result_A_D_SNR_{snr}.json.gz"
    with gzip.open(path, "wt", encoding="UTF-8") as zipfile:
        json.dump(result_A_D, zipfile)
