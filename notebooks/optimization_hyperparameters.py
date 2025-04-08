#!/usr/bin/env python
# coding: utf-8

# # Evaluate AMORS methods
# This notebook explores the results of evaluating the AMORS and VectAMORS algorithms with different hyperparameters and data conditions

# Copyright (c) 2024 University of Strasbourg
# Author: Massyl Moudoud <mmoudoud@unistra.fr> 
# Contributor(s) : Céline Meillier <meillier@unistra.fr>, Vincent Mazet <vincent.mazet@unistra.fr>
# 
# This work has been supported by the ANR project DynaSTI: ANR-22-CE45-0008
# 
# This software is governed by the CeCILL  license under French law and
# abiding by the rules of distribution of free software.  You can  use, 
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info". 
# 
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability. 
# 
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or 
# data to be ensured and,  more generally, to use and operate it in the 
# same conditions as regards security. 
# 
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

# In[1]:


# If you have a valid latex install set to True
# to render figure legends and labels in Latex style
use_Latex = False


# In[2]:


# Set paths
import pathlib
try:
    script_path = str(pathlib.Path(__file__).parent.resolve()) +"/"
except NameError:
    script_path = ""


# In[3]:


#Imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys
import json
import gzip

functionPath = script_path + "../src"
sys.path.append(functionPath)

from utile.eval_error import get_rescaled_errors


# In[4]:


# Check is latex is used# Check is latex is used
if use_Latex:
    matplotlib.rcParams.update({'font.size': 35, 'text.usetex' : True, 'font.family':"ptm"})
    matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
else:
    plt.rcParams.update({'font.size': 35})


# In[5]:


#disable warnings
import warnings
warnings.filterwarnings("ignore")


# In[6]:


#load ground truth data
data_path = script_path + "../data/simulation/simulation_A_D_2024_12_18.npz"

data  = np.load(data_path)
A_VT = data["A"]
D_VT = data["D"]

E, P = D_VT.shape
T = A_VT.shape[1]

D_tilde = np.int32(D_VT != 0)
C_VT = D_VT @ A_VT


# # Effect of $\lambda_{\ell_1}$ and $\lambda_{TV}$ hyperparameters

# In[7]:


result_path = script_path + "../scripts/optimization_results/"


# In[8]:


snr_val = 0


res_path = result_path + f"FISTA_result_A_D_SNR_{snr_val}.json.gz"
with gzip.open(res_path, 'rt', encoding='UTF-8') as zipfile:
        res_A_D = json.load(zipfile)
        
        
#Convert back array to numpy
for i , res in enumerate(res_A_D):
    for key, val in res.items():
        res_A_D[i][key] = np.array(val)
        
#keep track of the variables names in the array
idx_to_var = {idx : var for idx ,var in enumerate(res_A_D[0].keys())}
var_to_idx = {var : idx for idx ,var in enumerate(res_A_D[0].keys())}

Nb_experiments = len(res_A_D)


# In[9]:


error = np.zeros((Nb_experiments, 4))
error_AMORS = np.zeros((Nb_experiments, 4))
error_VectAMORS = np.zeros((Nb_experiments, 4))

lambda_L2_D = np.zeros(Nb_experiments)
lambda_L1_A = np.zeros(Nb_experiments)
lambda_TV_A = np.zeros(Nb_experiments)


for i, res in enumerate(res_A_D):
    error[i,:] = np.NaN if np.any(np.isnan(res["err"])) else res["err"][-1,:]
    error_AMORS[i,:] = np.NaN if np.any(np.isnan(res["err_AMORS"])) else res["err_AMORS"][-1,:]
    error_VectAMORS[i,:] = np.NaN if np.any(np.isnan(res["err_VectAMORS"])) else res["err_VectAMORS"][-1,:]
    lambda_L2_D[i] = res["lambda_l2_D"]
    lambda_L1_A[i] = res["lambda_l1_A"]
    lambda_TV_A[i] = res["lambda_TV"]


# In[10]:


#rescale estimated matrices with oracle rescaling
correction_count = 0

for i, res in enumerate(res_A_D):
    error[i,2:] = np.NaN if np.any(np.isnan(error[i,:])) else  get_rescaled_errors(res["D_est"], res["A_est"], D_VT, A_VT)
    error_AMORS[i,2:] = np.NaN if np.any(np.isnan(error_AMORS[i,:])) else get_rescaled_errors(res["D_est_AMORS"], res["A_est_AMORS"], D_VT, A_VT)
    error_VectAMORS[i,2:] = np.NaN if np.any(np.isnan(error_VectAMORS[i,:])) else get_rescaled_errors(res["D_est_VectAMORS"], res["A_est_VectAMORS"], D_VT, A_VT)


# # show best estimate per method

# In[11]:


i = 1 #select best for C
best_AM_idx = np.argmin(error[:, i])
best_AMORS_idx = np.argmin(error_AMORS[:, i])
best_VectAMORS_idx = np.argmin(error_VectAMORS[:, i])

print("####################################")
print("Optimal hyperparameters per method")
print(f"AM:\n{lambda_L2_D[best_AM_idx]=}\n{lambda_L1_A[best_AM_idx]=}\n{lambda_TV_A[best_AM_idx]=}")
print("####################################")
print(f"AMORS:\n{lambda_L2_D[best_AMORS_idx]=}\n{lambda_L1_A[best_AMORS_idx]=}\n{lambda_TV_A[best_AMORS_idx]=}")
print("####################################")
print(f"VectAMORS:\n{lambda_L2_D[best_VectAMORS_idx]=}\n{lambda_L1_A[best_VectAMORS_idx]=}\n{lambda_TV_A[best_VectAMORS_idx]=}")


# # joint effect of parameters

# In[12]:


#error in percent
mat = 1 #which matrix 0:criterion 1:C, 2:D, 3:A

norm_C = np.linalg.norm(C_VT, ord = "fro")/100

tmp_dict = {"lambda_L2_D": lambda_L2_D, "lambda_L1_A": lambda_L1_A, "lambda_TV_A": lambda_TV_A, 
            "AM": error[:, mat]/norm_C, "AMORS": error_AMORS[:, mat]/norm_C, "VectAMORS": error_VectAMORS[:, mat]/norm_C}
result_DF = pd.DataFrame(tmp_dict)


# # effect of two params with the best result of the the third

# In[13]:


def plots_best( DF,
                cmap, 
                disp_param = 0, # l1_A
                param_idx1 = 1,# l2_D
                param_idx2 = 2, #TV  
                path  = None
              ):
    """
    Plots the pairwise plots of the errors vs hyperparameters
    """        
        
    lambda_l1_A_vals = np.unique(DF["lambda_L1_A"])
    lambda_l2_D_vals = np.unique(DF["lambda_L2_D"])
    lambda_TV_vals = np.unique(DF["lambda_TV_A"])

                    
    
    params =[lambda_l1_A_vals , lambda_l2_D_vals , lambda_TV_vals]
    
    param_names =["lambda_L1_A" , "lambda_L2_D" , "lambda_TV_A"]
    param_names_disp = [r"$\boldsymbol{\mu}$", r"$\boldsymbol{\lambda}$" , r"$\boldsymbol{\nu}$"]
       
    
#     disp_param = 0#TV
#     param_idx1 = 1#l1_A
#     param_idx2 = 2#l2_D
    
    
    
        
    AM = np.zeros((len(params[param_idx2]), len(params[param_idx1])))
    AMORS = np.zeros((len(params[param_idx2]), len(params[param_idx1])))
    VectAMORS = np.zeros((len(params[param_idx2]), len(params[param_idx1])))

    

    for j, p1 in enumerate(params[param_idx1]):
        for k, p2 in enumerate(params[param_idx2]):
            tmp = DF.query(f"{param_names[param_idx1]} == {p1} and {param_names[param_idx2]} == {p2}")
            AM[k , j] = np.min(tmp["AM"])
            AMORS[k , j] = np.min(tmp["AMORS"])
            VectAMORS[k , j] = np.min(tmp["VectAMORS"])
            
        
    min_val = 11#np.min((np.min(AM), np.min(AMORS), np.min(VectAMORS)))
    max_val = 33#np.max((np.max(AM), np.max(AMORS), np.max(VectAMORS)))
    
    fig = plt.figure(layout = "constrained" , figsize=(18, 4.5))

    #### AM
    ax1 = plt.subplot(131)
    plt.pcolormesh(params[param_idx1] , params[param_idx2], AM ,shading = 'gouraud', cmap=cmap, vmin = min_val, vmax =max_val)
    #plt.colorbar()  
    plt.xlabel(param_names_disp[param_idx1])
    plt.ylabel(param_names_disp[param_idx2])
    plt.title("Méthode [4]")



    ############## AMORS
    plt.subplot(132 , sharex = ax1)
    plt.pcolormesh(params[param_idx1] , params[param_idx2], AMORS , shading = 'gouraud', cmap= cmap,  vmin = min_val, vmax =max_val)
    #plt.colorbar()
    plt.xlabel(param_names_disp[param_idx1])
    plt.yticks([])
    plt.title("AMORS [6]")



    ########## VectAMORS
    plt.subplot(133 , sharex = ax1)
    plt.pcolormesh(params[param_idx1] , params[param_idx2], VectAMORS ,cmap=cmap, shading = 'gouraud',  vmin = min_val, vmax =max_val)#
    plt.colorbar(ticks=np.linspace(min_val,  max_val, 4), label = "Erreur \%")
    plt.xlabel(param_names_disp[param_idx1])
    plt.yticks([])
    plt.title("Méthode proposée")



    ######
    #plt.suptitle(param_names[disp_param] + "=" + str(val))
    #plt.figlegend(loc=(0.81, 0.4))

    
    if path is not None:
        plt.savefig(path,  bbox_inches='tight')
    
    
    return AM, AMORS, VectAMORS, fig


# In[14]:


spath = None#script_path + "./figs/Figure2a.jpeg"
AM, AMORS, VectAMORS, fig = plots_best(result_DF,
                                  cmap = 'viridis', 
                                  disp_param = 1, # l1_A
                                  param_idx1 = 0,# l2_D
                                  param_idx2 = 2, #TV  
                                  path  = spath)

# If in script close the figures
if script_path != "":
    plt.close(fig)


# In[15]:


spath = script_path + "./figs/Figure2b.jpeg"
AM, AMORS, VectAMORS, fig = plots_best(result_DF,
                                  cmap = 'viridis', 
                                  disp_param = 0, # l1_A
                                  param_idx1 = 1,# l2_D
                                  param_idx2 = 2, #TV  
                                  path  = spath)

# If in script close the figures
if script_path != "":
    plt.close(fig)


# In[ ]:




