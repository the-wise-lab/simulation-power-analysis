# Power analysis for regression models via simulation

This repository contains Python code for performing power analysis for regression models via simulation. This is particularly useful for estimating power for complex regression models with data that may not be normally distributed.

## Installation

To install the package, first clone or download the repository. Then, navigate to the repository directory and run the following command:

```bash
pip install -e .
```

## Usage

### Running power analyses

The package first generates simulated data using tools from the [Copulas](https://sdv.dev/Copulas/index.html) package. This data should closely approximate the distributions and covariance structure of the true data. It then samples from the fitted copula to generate synthetic data with given sample sizes.

Power analysis can be run using the `power_analysis` function. This takes as input a dataset along with the dependent and independent variables.

```python
from simulation_power_analysis.analysis import power_analysis

# data is a pandas dataframe
power_df = power_analysis(data, 'dependent', ['independent', 'variables'])

```

The function can also take a number of optional arguments, including the sample sizes to use, the number of simulations to run, and the number of cores to use. See the docstring for more details.

### Assessing the quality of the synthetic data

It is important to check how well the synthetic data approximates the true data. This can be done by generating synthetic data and inspecting its covariance structure and the distributions of included variables.

Synthetic data can be generated using the `fit_distribution` and `sample_synthetic` functions. The former fits a copula to the data, while the latter samples from the fitted copula to generate synthetic data.

```python
from simulation_power_analysis.analysis import fit_distribution, sample_synthetic

# Fit a copula to the data
dist = fit_distribution(data)

# Sample synthetic data
synthetic_data = sample_synthetic(dist, 1000)
```

This will return a pandas dataframe containing the synthetic data. The covariance structure of the synthetic data can be inspected using the `plot_covariance` function.

```python
from simulation_power_analysis.plotting import plot_covariance

plot_covariance(data, synthetic_data)
```

The distributions of the synthetic data can be inspected using the `plot_distributions` function.

```python
from simulation_power_analysis.plotting import plot_distributions

compare_distributions(data, synthetic_data)
```

## Notes

This package is limited in its functionality and only works for regression models. It estimates p-values using bootstrapping to allow for non-normality in the data. It also only works for continuous or binary variables, and does not currently support dummy coding. 