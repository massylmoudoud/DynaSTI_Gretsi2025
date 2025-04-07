# -*- coding: utf-8 -*-
"""
Created on Thr  Apr 18 2024 17:36:00

@author: MassylMoudoud
"""

import numpy as np

# to check correct data type
import collections.abc
import numbers


def FISTA_D(
    C,
    D_tilde,
    A,
    D_init=None,
    lambda_l2=0,
    maxIter=100,
    ret_err=False,
    C_VT=None,
    D_VT=None,
):
    """
    Find the activation matrix A by solving the optimization problem
        A = argmin( ||C - DA||F^2 + lambda_TV TV(A) + lambda_l1 ||A||1 such that A in [0 , 1]
     Using FIST algotrithm

     Inputs:
        C :
        D_tilde :
        D_init :
        lambda_l1:
        maxIter :
        ret_err :
        D_VT: ground truth
     Outputs:
        D : estimated matrix
        err: (if ret_err = True) return the convergence curve of the reconstruction error of C
    """

    eps = np.finfo("float").eps

    def soft(x, T):
        T = T + eps
        y = np.maximum(np.abs(x) - T, 0)
        y = y / (y + T) * x
        return y

    # Get sizes
    T = C.shape[1]
    P = A.shape[0]

    # check if lambda_l2 is scallar of vector (per FCU)
    if lambda_l2 is None:
        pass  # no L1 penalty
    elif isinstance(lambda_l2, collections.abc.Sized):  # vector
        if len(lambda_l2) != P:
            raise ValueError(
                f"lambda_l2 should be of length {P=} but is of length {len(lambda_l2)}"
            )

    elif isinstance(lambda_l2, numbers.Real):  # number
        lambda_l2 = np.ones(P) * lambda_l2
    else:
        raise ValueError(
            f"lambda_l2 should scalar or vector but {type(lambda_l2)} was given"
        )

    if ret_err:
        err = np.zeros((maxIter, 2))

    # Initialization
    A_A_T = A @ A.T
    C_A_T = C @ A.T

    # Lipsitz constant
    L = np.max(np.abs(np.linalg.eigvals(A_A_T + lambda_l2 * np.eye(P))))

    # Multipicative update in Gradient descent
    M = (1 - lambda_l2 / L) * np.eye(P) - (1 / L) * A_A_T

    # FISTA weight
    t = np.zeros(maxIter + 1)
    t[0] = 1

    if D_init is None:
        D_init = D_tilde.astype(np.float64).copy()

        # Latent variable
    Y = D_init.copy()

    D = Y.copy()
    D_prev = Y.copy()

    for i in range(maxIter):
        # Gradient descent
        D = Y @ M + (1 / L) * C_A_T

        # Proximal operators
        # Indicator function
        # structure on D_tilde
        D = D * D_tilde

        # Positivity
        D[D < 0] = 0

        # Weigth update
        t[i + 1] = (1 + np.sqrt(1 + 4 * t[i] ** 2)) / 2

        # Latent variable update
        Y = D + ((t[i] - 1) / t[i + 1]) * (D - D_prev)

        # Check convergence
        # TODO

        # Keep track of error
        if ret_err:
            if C_VT is None:
                err[i, 0] = np.linalg.norm(C - D @ A, ord="fro")
            else:  # C_VT is known
                err[i, 0] = np.linalg.norm(C_VT - D @ A, ord="fro")
            if D_VT is None:
                err[i, 1] = np.mean(np.abs(D - D_prev))
            else:  # D_VT is known
                err[i, 1] = np.mean(np.abs(D_VT - D))

        # Keep past value of D (D[i-1])
        D_prev = D.copy()

    if ret_err:
        return D, err

    return D
