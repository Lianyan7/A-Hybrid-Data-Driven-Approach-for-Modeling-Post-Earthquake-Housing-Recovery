# A Hybrid Data-Driven Approach for Modeling Post-Earthquake Housing Recovery
Supplemental Materials for the Paper "A Hybrid Data-Driven Approach for Modeling Post-Earthquake Housing Recovery"

This project models post-earthquake housing recovery at both individual building and community levels. At the individual building level, a probabilistic Bayesian estimation framework, incorporating both MCMC and EB methods, is employed to estimate and update parameters of key temporal variables (recovery time, financing time, delay time, inspection time, and repair time) involved in building recovery modeling. At the community level, a range of machine learning (ML) models (Random Forest (RF), Gradient Boosting Machine (GBM), Decision Tree (DT), and Extreme Gradient Boosting (XGBoost)) is applied to predict recovery trajectories at the community level. 

Since the data utilized in this study involves private insurance claim settlements and cannot be publicly disclosed, this repository only includes python codes and certain input information. The code was developed by Lianyan Li, and the underlying calculations are based on the following publication:

Li, L., Chang-Richards, A., et al. (2025). A Hybrid Data-Driven Approach for Modeling Post-Earthquake Housing Recovery. Earthquake Spectra.

Note: recovery trajectories in this study are defined as the temporal progression of housing capacity restoration, quantified by the proportion of buildings repaired over time, with full regional recovery achieved when all affected buildings have been restored. Additionally, the data utilized in this study involves private insurance claim settlements and cannot be publicly disclosed. Therefore, only the Python file (Machine Learning Algorithms Combined.py) is provided, with the grid search for hyperparameters detailed in the file Optimal Hyperparameter Configurations.xlsx.

For individual-level building recovery modeling, two python files (Metropolis-Hastings MCMC.py and Empirical Bayes.py) are provided to update parameters of temporal variables of interest, with the prior-information detailed in 

four machine learning models—Random Forest (RF), Gradient Boosting Machine (GBM), Decision Tree (DT), and Extreme Gradient Boosting (XGBoost)—to predict building recovery times for individual structures and generate recovery trajectories for affected buildings following earthquakes.
