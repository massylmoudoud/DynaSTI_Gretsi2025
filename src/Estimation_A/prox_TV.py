# -*- coding: utf-8 -*-
"""
Created on Fri  Apr 12 2024 10:51:00

Interface to call the Ccode to find prox operator of TV regularization using Condat Code

The shared object is compiled with the command:
gcc -fpic -O3 -march=native -Wall -shared -o lib_prox_TV_1D_Condat.so Condat_TV_1D_v2.c

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

# imports
import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer

import os

path = os.path.dirname(os.path.realpath(__file__))


# Load C library
# path to the shared object of the compiled C function
lib_path = path + "/lib_prox_TV_1D_Condat.so"
lib = ctypes.cdll.LoadLibrary(lib_path)

c_prox_TV_1D = lib.TV1D_denoise_v2
c_prox_TV_1D.restype = None
c_prox_TV_1D.argtypes = [
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ctypes.c_size_t,
    ctypes.c_double,
]


def prox_TV(A_i, lambda_TV):
    """
    Compute proximal operator of TV regularization on each row of a matrix A (anisotropic TV)
    For each Row call the 1D prox TV of Condat
    Inputs:
        A_i: 1 x T vector (one row of A) T is the number of time points
        Lambda_TV: TV regularization constant

    Output:
        out: out = prox_TV(A , lambda_TV) = argmin_U (lambda_TV* TV(U) + ||A - U||_F^2)
    """

    T = len(A_i)
    # initialize output
    out = np.empty_like(A_i)

    # Compute prox in C
    c_prox_TV_1D(A_i, out, T, lambda_TV)

    return out
