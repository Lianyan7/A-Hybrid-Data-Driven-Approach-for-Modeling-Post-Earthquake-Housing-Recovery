# A Hybrid Data-Driven Approach for Modeling Post-Earthquake Housing Recovery

Supplemental Materials for the Paper **"A Hybrid Data-Driven Approach for Modeling Post-Earthquake Housing Recovery"**

This project presents a hybrid data-driven approach for modeling post-earthquake housing recovery at both the individual building and community levels. At the individual building level, a probabilistic **Bayesian estimation framework** is employed, combining **Metropolis-Hastings MCMC and Empirical Bayes** methods to estimate and update parameters of key temporal variables (e.g., recovery time, financing time, delay time, inspection time, and repair time) that characterize building recovery processes. At the community level, a suite of machine learning models—including **Random Forest, Gradient Boosting Machine, Decision Tree, and Extreme Gradient Boosting**—is applied to predict recovery trajectories and assess the overall recovery performance of affected communities.

Due to the sensitive nature of the underlying data, which comprises private insurance claim settlements, only the Python code and essential input information are made available in this repository. The code was developed by Lianyan Li and is based on the methodologies described in the following publication:

Li, L., Chang-Richards, A., et al. (2025). ***A Hybrid Data-Driven Approach for Modeling Post-Earthquake Housing Recovery***. Earthquake Spectra.

For individual-level building recovery modeling, two Python scripts (_Metropolis-Hastings MCMC.py_ and _Empirical Bayes.py_) are provided. These scripts update the parameters of temporal variables using prior information detailed in _Prior beliefs of the parameters.xlsx_ and the descriptive statistics of the collected empirical dataset presented in _Summarized statistics of the collected empirical dataset.xlsx._

For community-level modeling, the repository includes the script _Machine Learning Algorithms Combined.py_, which evaluates four machine learning models to predict individual building recovery times and generate comprehensive recovery trajectories. Hyperparameter configurations for the grid search are documented in _Optimal Hyperparameter Configurations.xlsx._
