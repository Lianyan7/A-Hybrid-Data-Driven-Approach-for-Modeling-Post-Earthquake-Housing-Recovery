# A Hybrid Data-Driven Approach for Modeling Post-Earthquake Housing Recovery

## Overview
This repository contains supplemental materials for the paper titled **"A Hybrid Data-Driven Approach for Modeling Post-Earthquake Housing Recovery"**, published in *Earthquake Spectra* (2025). The study presents a hybrid data-driven framework for modeling post-earthquake housing recovery at both individual building and community levels.

## Methodology
The research employs a dual-level analytical approach:

**Individual Building Level:**
- Probabilistic Bayesian estimation framework combining Metropolis-Hastings MCMC and Empirical Bayes methods
- Estimation and updating of parameters for key temporal variables:
  - Recovery time
  - Financing time
  - Delay time
  - Inspection time
  - Repair time

**Community Level:**
- Machine learning ensemble for predicting individual building recovery times
- Generation of comprehensive recovery trajectories for affected buildings
- Implemented algorithms include:
  - Random Forest
  - Gradient Boosting Machine
  - Decision Tree
  - Extreme Gradient Boosting

## Reference
Li, L., Chang-Richards, A., et al. (2025). A Hybrid Data-Driven Approach for Modeling Post-Earthquake Housing Recovery. *Earthquake Spectra*, pp. 1-29. https://doi.org/10.1177/87552930251345841

## Ethics Approval
This research was conducted with ethics approval granted by the University of Auckland Human Participants Ethics Committee (Reference Number: 25474).

## Data Availability
The data utilized in this research comprises private insurance claim settlements and is not publicly available due to confidentiality restrictions. Only Python codes and essential input/output information are provided in this repository.

## Code Implementation

### Individual-Level Building Recovery Modeling
- `Kolmogorov–Smirnov (KS) test.py`: Theoretical distribution estimation for temporal variables
- `Metropolis-Hastings MCMC.py`: Parameter estimation using MCMC methods
- `Empirical Bayes.py`: Parameter updating based on prior information

### Community-Level Modeling
- `Machine Learning Algorithms Combined.py`: Implementation of four machine learning models for recovery time prediction and trajectory generation

### Supporting Files
- `Prior beliefs of the parameters.xlsx`: Prior information for Bayesian estimation
- `Summarized statistics of the collected empirical dataset.xlsx`: Descriptive statistics of empirical data
- `KS test results.xlsx`: Documentation of KS test outcomes
- `Optimal Hyperparameter Configurations.xlsx`: Hyperparameter settings for machine learning models

## Intended Use
This repository is intended for academic and professional use, contributing to the advancement of post-disaster recovery modeling methodologies. The code may serve as a reference for researchers and practitioners working in disaster recovery, urban resilience, and machine learning applications.

## Disclaimer
The code and materials are provided for research purposes only. Users are responsible for ensuring proper data handling and compliance with relevant ethical guidelines when adapting this work for other applications.
