
date > time.txt
# Plot example of simulation data (Figure 1)
echo '################################################'
echo 'Plot example of simulation data (Figure 1)'
python notebooks/Plot_simulation_data.py 


# Optimize hyperparemeters
echo '################################################'
echo 'Optimizing hyperparemeters'

if [[ "$1" == "full" ]]; then
   echo 'Running on the full grid (will take time)'
   python scripts/optimize_hyperparameters_simulation.py --full
else
   echo 'Running on reduced grid'
   #python scripts/optimize_hyperparameters_simulation.py 
fi


# Plot reconstruction errro vs hyperparameters (Figure 2)
echo '################################################'
echo 'Plot reconstruction error vs hyperparameters (Figure 2)'
python notebooks/optimization_hyperparameters.py 


# Evaluate methods on multiple datasets with best parameters
echo '################################################'
echo 'Evaluate methods on multiple datasets with best parameters'


if [[ "$1" == "full" ]]; then
   echo 'Runnin 100 realizations (will take time)'
   python scripts/evaluate_model_mutliple_data.py
else
   echo 'Running 25 realizations'
   python scripts/evaluate_model_mutliple_data.py  25
fi


# Plot reconstruction error box plot (Figure 3)
echo '################################################'
echo 'Plot reconstruction error box plot (Figure 3)'
python notebooks/evaluate_mutlirealization.py 

# Plot convergence curves (Fgure 4)
echo '################################################'
echo 'Plot convergence curves (Fgure 4)'
python notebooks/Convergence_tests.py 

date >> time.txt

#shutdown '+5' -P --no-wall
