# -*- coding: utf-8 -*-
"""
Created on Wed Oct 02 11∶42∶44 2024


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

# to check correct data type
import collections.abc
import numbers


"""
A method to estimate the activations through time of the different FCU,
 by reconstructing the correlations matrices on the dictionary of FCU.
"""


from Estimation_A.prox_TV import prox_TV


def FISTA_A(
    D,
    C,
    A_init=None,
    lambda_l1=None,
    lambda_TV=None,
    maxIter=100,
    mu=0,
    ret_err=False,
    C_VT=None,
    A_VT=None,
):
    """
    Find the activation matrix A by solving the REDUCED optimization problem
        A = argmin( ||C - DA||F^2  + lambda_l1 ||A||1 such that A > 0
     Using FISTA algotrithm

     Inputs:
        C :
        D :
        A_init :
        lambda_l1: vector of hyperparameters each FCU has one lambda
        maxIter :
        ret_err :
     Outputs:
        A : estimated matrix
        err: (if ret_err = True) return the convergence curve of the reconstruction error of C
    """
    # get sizes
    T = C.shape[1]
    P = D.shape[1]

    eps = np.finfo("float").eps

    def soft(x, lambda_l1):
        lambda_l1 = lambda_l1 + eps
        y = np.maximum(np.abs(x) - lambda_l1, 0)
        y = y / (y + lambda_l1) * x
        return y

    # check if lambda_l1 is scallar of vector (per FCU)
    if lambda_l1 is None:
        pass  # no L1 penalty
    elif isinstance(lambda_l1, collections.abc.Sized):  # vector
        if len(lambda_l1) != P:
            raise ValueError(
                f"lambda_l1 should be of length {P=} but is of length {len(lambda_l1)}"
            )

    elif isinstance(lambda_l1, numbers.Real):  # number
        lambda_l1 = np.ones(P) * lambda_l1
    else:
        raise ValueError(
            f"lambda_l1 should scalar or vector but {type(lambda_l1)} was given"
        )

    # check if lambda_TV is scallar or vector (per FCU)
    if lambda_TV is None:
        pass  # no TV penalty reg_TV = 0

    elif isinstance(lambda_TV, collections.abc.Sized):  # vector
        if len(lambda_TV) != P:
            raise ValueError(
                f"lambda_TV should be of length {P=} but is of length {len(lambda_TV)}"
            )

    elif isinstance(lambda_TV, numbers.Real):  # number
        lambda_TV = np.ones(P) * lambda_TV

    else:
        raise ValueError(
            f"lambda_TV should scalar or vector but {type(lambda_TV)} was given"
        )

    if ret_err:
        err = np.zeros((maxIter, 2))

    # Initialization
    D_T_D = D.T @ D
    D_T_C = D.T @ C
    # Lipsitz constant
    L = np.max(np.abs(np.linalg.eigvals(D_T_D)))

    # FISTA weight
    t = np.zeros(maxIter + 1)
    t[0] = 1

    # A_init
    if A_init is None:
        # TODO initialize A
        A_init = np.ones((P, T))

        # Latent variable
    Y = A_init.copy()

    A = Y.copy()
    A_prev = Y.copy()

    for i in range(maxIter):
        # Gradient descent
        Z = (np.eye(P) - (1 / L) * D_T_D) @ Y + (1 / L) * D_T_C

        # Proximal operators
        # L1

        if lambda_l1 is not None:
            for fcu, l1 in enumerate(lambda_l1):
                A[fcu, :] = soft(Z[fcu, :], l1 / L)

            # TV
        if lambda_TV is not None:
            for fcu, TV in enumerate(lambda_TV):
                A[fcu, :] = prox_TV(A[fcu, :], TV / L)

            # indicator function
            # positivity
        A[A < 0] = 0

        # Weigth update
        t[i + 1] = (1 + np.sqrt(1 + 4 * t[i] ** 2)) / 2

        # Latent variable update
        Y = A + ((t[i] - 1) / t[i + 1]) * (A - A_prev)

        # Check convergence
        # TODO

        # Keep track of error
        if ret_err:
            if C_VT is None:
                err[i, 0] = np.linalg.norm(C - D @ A, ord="fro")
            else:  # C_VT is known
                err[i, 0] = np.linalg.norm(C_VT - D @ A, ord="fro")

            if A_VT is None:
                err[i, 1] = np.linalg.norm(A - A_prev, ord="fro")
            else:  # A_VT is known
                err[i, 1] = np.linalg.norm(A_VT - A, ord="fro")

        # keep past value of A (A[i-1])
        A_prev = A.copy()

    if ret_err:
        return A, err

    return A
