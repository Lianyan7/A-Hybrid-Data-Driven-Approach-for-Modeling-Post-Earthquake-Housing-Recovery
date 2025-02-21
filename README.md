# A Hybrid Data-Driven Approach for Modeling Post-Earthquake Housing Recovery
Supplemental Materials for the Paper "A Hybrid Data-Driven Approach for Modeling Post-Earthquake Housing Recovery"

This script executes four machine learning models—Random Forest (RF), Gradient Boosting Machine (GBM), Decision Tree (DT), and Extreme Gradient Boosting (XGBoost)—to predict building recovery times for individual structures and generate recovery trajectories for affected buildings following earthquakes. The code was developed by Lianyan Li, and the underlying calculations are based on the following publication:

Li, L., Chang-Richards, A., et al. (2025). A Hybrid Data-Driven Approach for Modeling Post-Earthquake Housing Recovery. Earthquake Spectra.

Note: recovery trajectories in this study are defined as the temporal progression of housing capacity restoration, quantified by the proportion of buildings repaired over time, with full regional recovery achieved when all affected buildings have been restored. Additionally, the data utilized in this study involves private insurance claim settlements and cannot be publicly disclosed. Therefore, only the Python file (Machine Learning Algorithms Combined.py) is provided, with the grid search for hyperparameters detailed in the file Optimal Hyperparameter Configurations.xlsx.
