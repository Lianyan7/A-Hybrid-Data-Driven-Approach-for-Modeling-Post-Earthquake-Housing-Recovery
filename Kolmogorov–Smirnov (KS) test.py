#!/usr/bin/env python3
"""
Integrated KS Test Script

This script performs Kolmogorov–Smirnov (KS) tests on a dataset stored in an Excel file.
It provides two analyses:
    1. Sheet-Level Tests: Tests the variable (assumed to be the second column) for each sheet.
    2. Damage State Tests: Groups data by the "Damage States" (assumed to be the first column)
       in each sheet and performs tests for each damage state group.

Tested distributions include:
    - Normal ('norm')
    - Exponential ('expon')
    - Gamma ('gamma')
    - Beta ('beta')
    - Lognormal ('lognorm')
    - Poisson ('poisson')

Results are saved into a single Excel file with two sheets:
    - "Sheet-Level": Results for tests on the variable from each sheet.
    - "Damage State": Results for tests on each damage state group.

Usage:
    Adjust the file_path variable as necessary and run the script.
"""

import pandas as pd
import scipy.stats as stats
import warnings


def run_sheet_level_tests(file_path):
    """
    Run KS tests for each sheet in the Excel file, treating the variable as a whole.

    Assumptions:
        - The Excel file contains one or more sheets.
        - The variable to test is assumed to be in the second column of each sheet.

    Returns:
        A pandas DataFrame with the results.
    """
    excel_data = pd.ExcelFile(file_path)
    results = []
    distributions = ['norm', 'expon', 'gamma', 'beta', 'lognorm', 'poisson']

    for sheet_name in excel_data.sheet_names:
        data = pd.read_excel(file_path, sheet_name=sheet_name)
        variable = data.columns[1]  # Assume the second column contains the variable

        # Extract variable values (drop any NaNs)
        values = data[variable].dropna()

        for dist_name in distributions:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)

                    # Define the theoretical CDF function based on the distribution
                    if dist_name == 'gamma':
                        shape, loc, scale = stats.gamma.fit(values)
                        cdf_values = lambda x: stats.gamma.cdf(x, shape, loc=loc, scale=scale)

                    elif dist_name == 'beta':
                        min_val, max_val = values.min(), values.max()
                        if min_val == max_val:
                            raise ValueError("Beta distribution requires variation in the data.")
                        # Scale data to [0, 1]
                        scaled_values = (values - min_val) / (max_val - min_val)
                        a, b, loc, scale = stats.beta.fit(scaled_values)
                        cdf_values = lambda x: stats.beta.cdf((x - min_val) / (max_val - min_val), a, b, loc=loc,
                                                              scale=scale)

                    elif dist_name == 'lognorm':
                        shape, loc, scale = stats.lognorm.fit(values)
                        cdf_values = lambda x: stats.lognorm.cdf(x, shape, loc=loc, scale=scale)

                    elif dist_name == 'poisson':
                        # For Poisson, data should be non-negative integers.
                        if not set(values.unique()).issubset(set(range(0, int(values.max()) + 1))):
                            raise ValueError("Poisson distribution requires non-negative integer data.")
                        lam = values.mean()
                        cdf_values = lambda x: stats.poisson.cdf(x, lam)

                    else:  # For 'norm' and 'expon'
                        dist = getattr(stats, dist_name)
                        params = dist.fit(values)
                        cdf_values = lambda x: dist.cdf(x, *params)

                    # Perform the KS test
                    d_statistic, p_value = stats.kstest(values, cdf_values)

                    results.append({
                        'Sheet Name': sheet_name,
                        'Variable': variable,
                        'Distribution': dist_name,
                        'D-Statistic': d_statistic,
                        'P-Value': p_value,
                        'Significant': 'No' if p_value > 0.05 else 'Yes'
                    })
            except Exception as e:
                results.append({
                    'Sheet Name': sheet_name,
                    'Variable': variable,
                    'Distribution': dist_name,
                    'Error': str(e)
                })

    return pd.DataFrame(results)


def run_damage_state_tests(file_path):
    """
    Run KS tests for each damage state group within each sheet.

    Assumptions:
        - The Excel file contains one or more sheets.
        - The first column contains "Damage States" and the second column contains the variable.

    Returns:
        A pandas DataFrame with the results.
    """
    excel_data = pd.ExcelFile(file_path)
    results = []
    distributions = ['norm', 'expon', 'gamma', 'beta', 'lognorm', 'poisson']

    for sheet_name in excel_data.sheet_names:
        data = pd.read_excel(file_path, sheet_name=sheet_name)

        # Assume the first column is 'Damage States' and the second column is the variable
        damage_states = data.iloc[:, 0]
        variable = data.iloc[:, 1]

        # Group by the damage state
        grouped = data.groupby(damage_states.name)

        for damage_state, group in grouped:
            values = group.iloc[:, 1].dropna()

            for dist_name in distributions:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)

                        if dist_name == 'gamma':
                            shape, loc, scale = stats.gamma.fit(values)
                            cdf_values = lambda x: stats.gamma.cdf(x, shape, loc=loc, scale=scale)

                        elif dist_name == 'beta':
                            min_val, max_val = values.min(), values.max()
                            if min_val == max_val:
                                raise ValueError("Beta distribution requires variation in the data.")
                            scaled_values = (values - min_val) / (max_val - min_val)
                            a, b, loc, scale = stats.beta.fit(scaled_values)
                            cdf_values = lambda x: stats.beta.cdf((x - min_val) / (max_val - min_val), a, b, loc=loc,
                                                                  scale=scale)

                        elif dist_name == 'lognorm':
                            shape, loc, scale = stats.lognorm.fit(values)
                            cdf_values = lambda x: stats.lognorm.cdf(x, shape, loc=loc, scale=scale)

                        elif dist_name == 'poisson':
                            if not set(values.unique()).issubset(set(range(0, int(values.max()) + 1))):
                                raise ValueError("Poisson distribution requires non-negative integer data.")
                            lambda_hat = values.mean()
                            cdf_values = lambda x: stats.poisson.cdf(x, lambda_hat)

                        else:  # For 'norm' and 'expon'
                            dist = getattr(stats, dist_name)
                            params = dist.fit(values)
                            cdf_values = lambda x: dist.cdf(x, *params)

                        d_statistic, p_value = stats.kstest(values, cdf_values)

                        results.append({
                            'Sheet Name': sheet_name,
                            'Damage State': damage_state,
                            'Variable': variable.name,
                            'Distribution': dist_name,
                            'D-Statistic': d_statistic,
                            'P-Value': p_value,
                            'Significant': 'No' if p_value > 0.05 else 'Yes'
                        })
                except Exception as e:
                    results.append({
                        'Sheet Name': sheet_name,
                        'Damage State': damage_state,
                        'Variable': variable.name,
                        'Distribution': dist_name,
                        'Error': str(e)
                    })

    return pd.DataFrame(results)


def main():
    # Set the file path of the Excel file (adjust as necessary)
    file_path = 'All data with three damage state.xlsx'

    # Run both test functions and retrieve results DataFrames
    sheet_level_df = run_sheet_level_tests(file_path)
    damage_state_df = run_damage_state_tests(file_path)

    # Write both results DataFrames to a single Excel file with two sheets
    output_file = 'KS_Test_Results_Combined.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        sheet_level_df.to_excel(writer, sheet_name="Sheet-Level", index=False)
        damage_state_df.to_excel(writer, sheet_name="Damage State", index=False)

    print(f"Combined KS test results saved to {output_file}")


if __name__ == "__main__":
    main()
