import contextlib
from typing import List, Optional, Union

import joblib.parallel
import numpy as np
import pandas as pd
import seaborn as sns
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import BetaUnivariate, GaussianKDE, GaussianUnivariate
from joblib import Parallel, delayed
from stats_utils.regression.analysis import add_bootstrap_methods_to_ols
from tqdm import tqdm
import statsmodels.formula.api as smf


def sample_synthetic(
    dist: GaussianMultivariate, n_samples: int, binary_columns: List[str] = None
) -> pd.DataFrame:
    """
    Samples from a fitted GaussianMultivariate distribution.

    Args:
        dist (GaussianMultivariate): Fitted multivariate distribution.
        n_samples (int): Number of samples to generate.
        binary_columns (list[str], optional): List of columns that need to be converted to binary in the synthetic data.

    Returns:
        pd.DataFrame: The synthetic dataset.

    """

    # Generate smaples
    samples = dist.sample(n_samples)

    # Convert binary columns to binary
    if binary_columns:
        for col in binary_columns:
            samples[col] = samples[col].round(0)

    return samples


def fit_distribution(
    true_data: pd.DataFrame
) -> (GaussianMultivariate, pd.DataFrame):
    """
    Fits a GaussianMultivariate copula.

    Args:
        true_data (pd.DataFrame): The real dataset used to train the copula.

    Returns:
        GaussianMultivariate: Fitted multivariate distribution.

    Example:
        >>> true_data = qdata[['transition_var', 'AD', 'Compul', 'SW', 'gender', 'age', 'motivation']]
        >>> binary_columns = ['gender']
        >>> n_samples = 2000
        >>> dist = fit_distribution(true_data)
    """

    # Use GaussianKDE for every column
    distributions = dict(
        zip(true_data.columns, [GaussianKDE for _ in true_data.columns])
    )

    # Create the copula
    dist = GaussianMultivariate(distribution=distributions)

    # Fit the copula
    dist.fit(true_data)

    return dist


def simulation_func(
    dist: GaussianMultivariate,
    sample_size: int,
    predictors: List[str],
    dependent_var: str,
    binary_cols: List[str] = [],
) -> np.ndarray:
    """
    Simulate synthetic data, fit a regression model, and perform bootstrapping.

    Args:
        dist (GaussianMultivariate): Fitted multivariate distribution.
        sample_size (int): Sample size.
        predictors (List[str]): List of predictor variables.
        dependent_var (str): Dependent variable.
        binary_cols (List[str], optional): Columns to convert to binary. Defaults to an empty list.

    Returns:
        np.ndarray: p-values from bootstrap, or NaNs if an exception occurs.

    Example:
        >>> sample_data = pd.DataFrame(...)
        >>> results = sim_func(sample_data, ['AD', 'Compul', 'SW', 'gender', 'age', 'motivation'], 'transition_var', ['gender'])
    """

    # Generate synthetic data
    synthetic_data = sample_synthetic(dist, sample_size, binary_cols)

    # Construct the regression formula
    formula = f'{dependent_var} ~ {" + ".join(predictors)}'

    try:
        # Run regression
        model = smf.ols(formula, data=synthetic_data).fit()

        # Replace the class with the bootstrap results class
        fitted_model = add_bootstrap_methods_to_ols(model)

        # Run bootstrap
        fitted_model.bootstrap(1000)
        _ = fitted_model.conf_int_bootstrap()

        # Return p-values for predictors (ignoring the intercept)
        return fitted_model.pvalues_bootstrap[1:]

    except:
        # If an exception occurs, return NaNs of length equal to the number of predictors
        return np.zeros(len(predictors)) * np.nan


@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm) -> tqdm:
    """
    Context manager to patch joblib for reporting progress within a tqdm progress bar.

    This context manager is used to modify the behavior of joblib's parallel processing
    so that it updates a given tqdm progress bar after the completion of each batch.

    Args:
        tqdm_object (tqdm): Instance of tqdm progress bar to be updated.

    Yields:
        tqdm: The updated tqdm progress bar object.

    References:
        Adapted from: https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697
    """

    # Define a callback class to update tqdm after each batch in joblib is completed.
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs) -> None:
            tqdm_object.update(n=self.batch_size)
            super().__call__(*args, **kwargs)

    # Backup the original callback to restore later
    old_batch_callback = joblib.parallel.BatchCompletionCallBack

    # Replace joblib's callback with our custom callback
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield tqdm_object
    finally:
        # Restore original callback after exiting context
        joblib.parallel.BatchCompletionCallBack = old_batch_callback

        # Close the tqdm progress bar
        tqdm_object.close()


def power_analysis(
    data: pd.DataFrame,
    dependent_var: str,
    variables: Optional[List[str]],
    sample_sizes: np.ndarray = np.arange(200, 1000, 100, dtype=int),
    n_iter: int = 100,
    binary_cols: Optional[List[str]] = None,
    significance_threshold: float = 0.05,
    format: str = "long",
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Compute power analysis over various sample sizes and return results in a dataframe.

    Args:
        data (pd.DataFrame): The dataset for analysis.
        dependent_var (str): Dependent variable for the regression.
        variables (List[str]): List of predictor variables.
        sample_sizes (np.ndarray): Array of sample sizes to analyze. Defaults to np.arange(200, 1000, 100).
        n_iter (int): Number of iterations for each sample size. Defaults to 100.
        binary_cols (List[str]): List of columns to be treated as binary. If None, assumes no binary columns.
        significance_threshold (float): Significance level for power calculation. Defaults to 0.05.
        format (str): Output format, either "long" or "wide". Defaults to "long".
        n_jobs (int): Number of jobs to run in parallel. Runs without joblib if set to 1. Defaults to 4.

    Returns:
        pd.DataFrame: Power analysis results, in either long or wide format. Values represent
            the proportion of p-values that are less than the significance threshold for
            each sample size and variable.

    Example:
        >>> results = power_analysis(dist, data, np.arange(100, 2000, 200), 100, n_jobs=4)
    """

    # Convert sample sizes to int for consistency
    sample_sizes = sample_sizes.astype(int)

    # Ensure that sample sizes are positive for valid simulation
    assert np.all(sample_sizes > 0)

    # If variables are not explicitly specified, use all columns in the data
    if variables is None:
        variables = data.columns.tolist()

    # Set binary_cols to an empty list if not specified
    if binary_cols is None:
        binary_cols = []

    # Fit the distribution to the data
    dist = fit_distribution(data)

    # Initialize an array to store p-values for all iterations and sample sizes
    all_pvals = np.zeros((len(sample_sizes), n_iter, len(variables)))

    # Loop over each sample size
    for n, ss in enumerate(sample_sizes):
        # If using parallel processing (i.e., multiple jobs)
        if n_jobs != 1:
            with tqdm_joblib(
                tqdm(
                    desc=f"Calculating power for n={ss} in {n_jobs} jobs", total=n_iter
                )
            ) as progress_bar:
                iteration_pvals = Parallel(n_jobs=n_jobs)(
                    delayed(simulation_func)(
                        dist, ss, variables, dependent_var, binary_cols
                    )
                    for _ in np.arange(n_iter)
                )
                all_pvals[n, :, :] = np.stack(iteration_pvals)
        # If not using parallel processing (i.e., a single job)
        else:
            iteration_pvals = [
                simulation_func(dist, ss, variables, dependent_var, binary_cols)
                for _ in tqdm(np.arange(n_iter), desc=f"Calculating power for n={ss}")
            ]
            all_pvals[n, :, :] = np.stack(iteration_pvals)

    # Calculate the power by determining the proportion of p-values below the significance threshold
    power = np.mean(all_pvals < significance_threshold, axis=1)

    # Create a dataframe to store power results for each sample size and variable
    df = pd.DataFrame(columns=["sample_size"] + variables)
    df["sample_size"] = sample_sizes
    for i, var in enumerate(variables):
        df[var] = power[:, i]

    # Convert the dataframe to long format if specified
    if format == "long":
        df = df.melt(id_vars="sample_size", var_name="variable", value_name="power")

    return df
