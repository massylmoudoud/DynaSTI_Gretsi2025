# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 2024  17:26:00

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


def eval_error(C_est, D_est, A_est, C_VT, D_VT, A_VT, verbose=True):
    """
    Compute the normalised mean suqred error on the estimated matrices C , D and A
    if verbose is True, only print the error
    if verbose is False compute and return the errors in the tuple (err_C, err_D, err_A)
    """

    err_C = np.linalg.norm(C_est - C_VT, ord="fro") / np.linalg.norm(C_VT, ord="fro")
    err_D = np.linalg.norm(D_est - D_VT, ord="fro") / np.linalg.norm(D_VT, ord="fro")
    err_A = np.linalg.norm(A_est - A_VT, ord="fro") / np.linalg.norm(A_VT, ord="fro")

    if verbose is False:
        return err_C, err_D, err_A

    # else print errors
    print("############")
    print("error on C is", end=" ")
    print(err_C)

    print("############")
    print("error on D is", end=" ")
    print(err_D)

    print("############")
    print("error on A is", end=" ")
    print(err_A)


def get_rescaled_errors(D_est, A_est, D_VT, A_VT):
    D_est_rescaled, A_est_rescaled = rescale_D_A_oracle(D_est, A_est, D_VT, A_VT)

    error_D = np.linalg.norm(
        D_est_rescaled - D_VT, ord="fro"
    )  # /np.linalg.norm( D_VT , ord='fro')
    error_A = np.linalg.norm(
        A_est_rescaled - A_VT, ord="fro"
    )  # /np.linalg.norm( A_VT , ord='fro')

    return error_D, error_A


def rescale_D_A(D, A):
    beta = np.max(np.abs(D), axis=0, keepdims=True)
    beta[np.isclose(beta, 0)] = 1

    return D / beta, A * beta.T


def rescale_D_A_oracle(D_est, A_est, D_VT, A_VT):
    P, T = A_VT.shape
    E = D_VT.shape[0]

    beta = np.linalg.norm(D_est, ord=2, axis=0, keepdims=True) / np.linalg.norm(
        D_VT, ord=2, axis=0, keepdims=True
    )
    beta[np.isclose(beta, 0)] = 1

    return D_est / beta, A_est * beta.T
