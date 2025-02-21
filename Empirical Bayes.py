#!/usr/bin/env python3
"""
Hierarchical Empirical Bayesian Estimation for Lognormal Data

This script reads temporal data with damage states from an Excel file,
computes a global (hierarchical) prior in log-space, and then applies a
hierarchical Bayesian update (with shrinkage) to estimate posterior means
and variances for each damage state. The goodness-of-fit of the lognormal
likelihood is evaluated using the Kolmogorov–Smirnov (KS) test, and the
results are saved to an Excel file.
"""

import numpy as np
import pandas as pd
from scipy.stats import kstest
import logging

# Configure logging for progress reporting
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def compute_global_prior(log_data: np.ndarray) -> tuple:
    """
    Compute global prior parameters in log-space.

    Parameters:
        log_data (np.ndarray): Log-transformed temporal data.

    Returns:
        tuple: (prior_mu, prior_sigma2) where prior_mu is the mean and
               prior_sigma2 is the variance in log-space.
    """
    prior_mu = np.mean(log_data)
    prior_sigma2 = np.var(log_data, ddof=1)
    return prior_mu, prior_sigma2


def back_transform(mu: float, sigma2: float) -> tuple:
    """
    Back-transform parameters from log-space to the original scale.

    Parameters:
        mu (float): Mean in log-space.
        sigma2 (float): Variance in log-space.

    Returns:
        tuple: (mean_original, variance_original) computed using the
               moments of the lognormal distribution.
    """
    mean_orig = np.exp(mu + sigma2 / 2)
    var_orig = (np.exp(sigma2) - 1) * np.exp(2 * mu + sigma2)
    return mean_orig, var_orig


def process_damage_state(state_values: np.ndarray, prior_mu: float, prior_sigma2: float, lambda_val: float) -> dict:
    """
    Process data for a single damage state, performing a KS test and a
    hierarchical Bayesian update in log-space.

    Parameters:
        state_values (np.ndarray): Array of temporal data for the damage state.
        prior_mu (float): Global prior mean in log-space.
        prior_sigma2 (float): Global prior variance in log-space.
        lambda_val (float): Shrinkage parameter controlling prior influence.

    Returns:
        dict: Dictionary with KS test results and posterior estimates on the original scale.
    """
    # Compute sample estimates in log-space
    log_state = np.log(state_values)
    state_mean_log = np.mean(log_state)
    state_var_log = np.var(log_state, ddof=1)
    state_sigma = np.sqrt(state_var_log)

    # Perform KS test for lognormality
    ks_stat, ks_p = kstest(state_values, 'lognorm', args=(state_sigma, 0, np.exp(state_mean_log)))

    # Hierarchical Bayesian update in log-space (Bayesian shrinkage)
    n_i = len(state_values)
    post_mu_log = (n_i * state_mean_log + lambda_val * prior_mu) / (n_i + lambda_val)
    post_sigma2_log = (n_i * state_var_log + lambda_val * prior_sigma2) / (n_i + lambda_val)

    # Back-transform to original scale
    post_mean_orig, post_var_orig = back_transform(post_mu_log, post_sigma2_log)

    return {
        'KS Statistic': ks_stat,
        'KS p-value': ks_p,
        'Posterior Mean (Original Scale)': post_mean_orig,
        'Posterior Variance (Original Scale)': post_var_orig
    }


def main():
    # File paths configuration
    file_path = 'All data with three damage state.xlsx'  # Replace with your file path
    output_file_path = 'EB_results_with_Prior.xlsx'       # Output file path

    # Read Excel data (all sheets)
    excel_data = pd.read_excel(file_path, sheet_name=None)

    # List to store results
    results_list = []

    # Shrinkage parameter that modulates prior influence
    lambda_val = 1.0  # Adjust as needed

    # Process each sheet (temporal variable)
    for sheet_name, df in excel_data.items():
        df = df.dropna()
        if df.empty:
            continue

        logging.info(f"Processing Sheet: {sheet_name}")

        # Extract data: first column is damage state, second is temporal variable
        damage_states = df.iloc[:, 0].values
        temporal_variable = df.iloc[:, 1].values

        # Filter non-negative temporal values (required for log transform)
        temporal_variable = temporal_variable[temporal_variable >= 0]
        if len(temporal_variable) == 0:
            continue

        # Compute global prior in log-space using all data in the sheet
        log_temporal = np.log(temporal_variable)
        prior_mu, prior_sigma2 = compute_global_prior(log_temporal)
        prior_mean_orig, prior_var_orig = back_transform(prior_mu, prior_sigma2)

        # Process each unique damage state
        unique_states = np.unique(damage_states)
        for state in unique_states:
            state_mask = (damage_states == state)
            state_values = temporal_variable[state_mask]
            if len(state_values) == 0:
                continue

            state_results = process_damage_state(state_values, prior_mu, prior_sigma2, lambda_val)

            # Append results to list
            results_list.append({
                'Sheet': sheet_name,
                'Damage State': state,
                'Prior Mean (Original Scale)': prior_mean_orig,
                'Prior Variance (Original Scale)': prior_var_orig,
                'Posterior Mean (Original Scale)': state_results['Posterior Mean (Original Scale)'],
                'Posterior Variance (Original Scale)': state_results['Posterior Variance (Original Scale)'],
                'KS Statistic': state_results['KS Statistic'],
                'KS p-value': state_results['KS p-value']
            })

    # Save results to Excel
    if results_list:
        results_df = pd.DataFrame(results_list)
        results_df.to_excel(output_file_path, index=False)
        logging.info(f"Results saved to: {output_file_path}")
    else:
        logging.warning("No valid data processed; no results saved.")


if __name__ == "__main__":
    main()
