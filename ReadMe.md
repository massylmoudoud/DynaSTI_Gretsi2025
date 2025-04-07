# Disclaimer
This code is a proof of concept and part of an ongoing work. It is provided for reproducing the results of a published paper.

> [!CAUTION]
>__This code is NOT intented for production use__

# ANR DynaSTI: Paper Gretsi 2025
This repository contains the code to reproduce the results of the paper;
"Correction matricielle de l'indétermination d'échelle pour l'optimisation alternée"
Submitted to the conference Gretsi 2025


# Aknowlegement
This work has been supported by the ANR (French research agency "Agence Nationale de la Recherche") project DynaSTI: ANR-22-CE45-0008

# Code structure
The code is structured as follows:

```Shell   
├──data
├──external
|   └──prox_TV_Condat
├──notebooks
|   ├──figs
|   ├──figs_Gretsi2025_soumission
|   ├──Convergence_tests.ipynb
|   ├──Convergence_tests.py
|   ├──Plot_simulation_data.ipynb
|   ├──Plot_simulation_data.py
|   ├──evaluate_mutlirealization.ipynb
|   ├──evaluate_mutlirealization.py
|   ├──optimization_hyperparameters.ipynb
|   └──optimization_hyperparameters.py
├──scripts
|   ├──optimization_results
|   ├──optimization_results_Gretsi_soumission
|   ├──evaluate_model_mutliple_data.py
|   └──optimize_hyperparameters_simulation.py
├──src
├──__init__.py
├──LICENCE_FRENCH
├──ReadMe.md
├──gretsi_soumission_ID1416.pdf
├──licence
├──requirements.txt
└──run_simulations.sh
```

The `data` folder contains the example dataset used in the paper.

The `external` folder contains the C code of the function to compyte the proximal operator of the total variation proposed and impemented by L. Condat.

The `notebooks` folder contains the jupyter notebooks to analyse the results and plot the figures

The folder **`notebooks/figs_GRETSI_soumission`** contains the exact figures that are used in the manuscript

The figures generated when running the code are stored in folder  **`notebooks/figs`** 

The codes to run the methods for mutliple datasets or on a grid of hyperparameters are in the `scripts` folder.

The source code of the methods are in the `src` folder

The PDF of the manuscript submitted to Gretsi 2025 is 
**`gretsi_soumission_ID1416.pdf`**

> [!IMPORTANT]
> All the notebooks (`.ipnb`) in **`notebooks`** where converted into scripts (`.py`) of the same names to 
> allow automated excecution.
> The scripts should __NOT__ be modified.


# How to reproduce the results:
After clonning the repository, open a terminal in the root folder of the project.

    
- First install the requirements in the file **`requiremnts.txt`** by running the folwing commends in athe terminal:
  This requires a valid [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) (or Anaconda) installation 
    
    ```Shell
    # Create a new environment
    conda create --name Grest2025_ID1416
    # Activate environment
    conda activate Grest2025_ID1416
    # Install dependencies
    conda install --yes --file requirements.txt
    # Compile the c library of L. Condat used to compute the proximal operator of total variation
    gcc -fpic -O3 -march=native -Wall -shared -o src/Estimation_A/lib_prox_TV_1D_Condat.so external/prox_TV_Condat/Condat_TV_1D_v2.c
    
    ```
        
- You can run all the scripts at once with the bash script **`run_simulations.sh`**
    
    ```bash
    # Activate environment if it is not activated
    conda activate Grest2025_ID1416
    # Run the simulations
    bash run_simulations.sh
    ```


Running the script will:
- Plot the example of simulation data used in the paper (Figure1)
- Run the optimization for the three methods on a reduced grid of hyperparameters to produce Figure 2.
(To run on the full grid of parameters as in the paper use the commend
**`bash run_simulations.sh full`** )

- Evaluate the three methods for the best set of hyperparameters for 100 realizations of the simulation data.
- Plot the box plot of the reconstruction error on these realizations (Figure 3)
- Evaluate the effect of initialization scaling on the convergence of the methods and plot the corresponding Figure 4.
- Save all the figures presented in the paper in the folder `notebooks/figs`

>[!IMPORTANT] 
>It should be noted that for the study of the effect of hyperparameters, we evaluated 1008 combinations. 
>Running these evaluations took more than 3 hours while using 52 CPU cores.
>With fewers cores, it will take much longer.
>The default behavior of the provided code is to run on a reduced grip of 108 parameters.
>You can run full the evaluation by adding the parameter `full` to the commend:
>**`bash run_simulations.sh full`**




