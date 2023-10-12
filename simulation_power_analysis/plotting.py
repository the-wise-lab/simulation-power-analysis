from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_covariance(
    true_data: np.ndarray,
    synthetic_data: np.ndarray,
    fig_kwargs: dict = {},
    imshow_kwargs: dict = {},
) -> None:
    """
    Plots the covariance matrices of the original data and sampled data side by side.

    Args:
        true_data (np.ndarray): The original data.
        synthetic_data (np.ndarray): The sampled synthetic data.
        fig_kwargs (dict, optional): Keyword arguments to be passed to plt.subplots().
        imshow_kwargs (dict, optional): Keyword arguments to be passed to ax.imshow().

    Returns:
        None: Displays a plot of the covariance matrices.

    Example:
        >>> true_data = pd.read_csv('data.csv')
        >>> dist = GaussianMultivariate()
        >>> dist.fit(true_data)
        >>> synthetic_data = dist.sample(1000)
        >>> plot_covariance(true_data, synthetic_data)

    """
    # check covariance of sampled data
    cov_synthetic = np.cov(synthetic_data.T)
    # and of the original data
    cov_true = np.cov(true_data.T)

    # plot the two side by side for comparison
    if not fig_kwargs:
        fig_kwargs = {"figsize": (10, 5)}
    if not "figsize" in fig_kwargs:
        fig_kwargs["figsize"] = (10, 5)
    fig, ax = plt.subplots(1, 2, **fig_kwargs)
    ax[0].imshow(cov_true, **imshow_kwargs)
    ax[0].set_title("Original Data")
    ax[1].imshow(cov_synthetic, **imshow_kwargs)
    ax[1].set_title("Sampled Data")

    # Colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.65])
    fig.colorbar(ax[0].imshow(cov_true), cax=cbar_ax)

    # Colorbar title
    cbar_ax.set_ylabel("Covariance", rotation=-90, va="bottom")

    plt.show()


def compare_distributions(
    true_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    fig_kwargs: dict = {},
    kdeplot_kwargs: dict = {},
) -> None:
    """
    Compare the distributions of original data and synthetic data for each column using KDE plots.

    Args:
        true_data (pd.DataFrame): The original dataset.
        synthetic_data (pd.DataFrame): The synthetic dataset.
        fig_kwargs (dict, optional): Keyword arguments to be passed to plt.subplots().
        kdeplot_kwargs (dict, optional): Keyword arguments to be passed to sns.kdeplot().

    Returns:
        None: Displays a comparison of KDE plots for each column.

    Example:
        >>> true_data = pd.DataFrame(...)
        >>> synthetic_data = pd.DataFrame(...)
        >>> compare_distributions(true_data, synthetic_data, fig_kwargs={'figsize': (14, 2.5)})

    """

    # Extract the column names from the true data
    columns = true_data.columns

    # Set default figure size based on the number of columns if not provided
    if not fig_kwargs:
        fig_kwargs = {"figsize": (len(columns) * 2, 2.2)}
    # Add default figsize to fig_kwargs if it doesn't exist
    if not "figsize" in fig_kwargs:
        fig_kwargs["figsize"] = (len(columns) * 2, 2.2)

    # Create subplots based on the number of columns
    fig, axes = plt.subplots(nrows=1, ncols=len(columns), **fig_kwargs)

    # Iterate over each column to plot KDE for both true and synthetic data
    for i, col in enumerate(columns):
        sns.kdeplot(true_data[col], ax=axes[i], **kdeplot_kwargs)
        sns.kdeplot(synthetic_data[col], ax=axes[i], **kdeplot_kwargs)
        # Set the title of the subplot to the column name
        axes[i].set_title(col)
        # Remove the y-axis label for all subplots except the first
        if i != 0:
            axes[i].set_ylabel("")

    # Adjust the layout for better visualization
    plt.tight_layout()
    # Display the plots
    plt.show()


def plot_power_curves(
    power_df: pd.DataFrame,
    required_power: float = 0.8,
    figure_kwargs: Optional[Dict] = None,
    plot_kwargs: Optional[Dict] = None,
) -> None:
    """
    Plot power curves for each variable using Seaborn.

    This function fits a curve to the data and uses this to report the sample size required to achieve the
    requested power level.

    Args:
        power_df (pd.DataFrame): Dataframe with columns 'sample_size', 'power', and 'variable'.
        required_power (float, optional): The power level to plot a vertical line for. Defaults to 0.8.
        figure_kwargs (Dict, optional): Keyword arguments for plt.figure(). Defaults to {'figsize': (5, 3)}.
        plot_kwargs (Dict, optional): Keyword arguments for sns.scatter(). If None, uses default settings.
    """

    if figure_kwargs is None:
        figure_kwargs = {"figsize": (5, 3)}

    if plot_kwargs is None:
        plot_kwargs = {}

    plt.figure(**figure_kwargs)

    if required_power > 1:
        raise ValueError("Required power must be between 0 and 1")
    

    # Get matplotlib default color cycle
    prop_cycle = plt.rcParams["axes.prop_cycle"]

    # Loop through each variable to plot
    for n, variable in enumerate(power_df["variable"].unique()):
        subset = power_df[power_df["variable"] == variable]

        # Scatter plot
        sns.scatterplot(data=subset, x="sample_size", y="power", label=variable)

        # Fit a polynomial (e.g., quadratic) curve to the data
        p = np.polyfit(subset["sample_size"], subset["power"], 3)
        curve_x = np.linspace(
            subset["sample_size"].min(), subset["sample_size"].max(), 1000
        )
        curve_y = np.polyval(p, curve_x)

        # Plot the fitted curve
        plt.plot(curve_x, curve_y)

        # Get sample size for 80% power
        if np.all(curve_y < required_power):
            # Raise warning and skip
            print(
                f"Warning: {variable} does not reach {required_power * 100}% power with the current sample size range."
            )
        else:
            idx_power = np.where(curve_y > required_power)[0][0]
            sample_size = curve_x[idx_power]

            # Print out the sample size for this variable
            print(
                f"Sample size for {required_power * 100}% power for {variable}: {sample_size:.0f}"
            )

            # Plot a vertical line at the sample size for 80% power, cutting off at the line
            # start y axis at the minimum power value currently plotted
            plt.plot(
                [sample_size, sample_size],
                [0, curve_y[idx_power]],
                color=prop_cycle.by_key()["color"][n],
                linestyle="--",
            )

    # Adjust y axis limits based on the minimum and maximum power values
    minimum_power = power_df["power"].min()
    maximum_power = power_df["power"].max()
    plt.ylim(minimum_power - 0.05, maximum_power + 0.05)

    sns.despine()

    plt.title("Power Curves by Variable")
    plt.xlabel("Sample Size")
    plt.ylabel("Power")
    plt.legend(title="Variable")
    plt.show()
