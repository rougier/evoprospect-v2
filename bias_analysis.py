# Social hierarchy influences monkeys' risky decisions
# Copyright 2024 (c) Naomi Chaix-Echel & Nicolas P Rougier
# Released under a BSD 2-clauses license

"""
This script contains functions for analyzing monkey bias in decision-making tasks.
It includes functions to compute bias, visualize bias vs. performance scores,
compare score differences, and generate related plots.

Functions:
- compute_bias(): Computes and saves bias for different monkeys.
- plot_bias_vs_score(): Plots monkey bias against performance scores.
- plot_bias_vs_score_diff(): Analyzes and visualizes score differences vs. bias.
- plot_bias_analysis(): Generates a comparative analysis of bias impact.
- plot_score_difference(): Compares risky and control condition scores.

Usage:
Import the script or run individual functions as needed.
"""

import pickle
from scipy import stats
from player import *
from prettytable import PrettyTable, SINGLE_BORDER
import json
import matplotlib.pyplot as plt

# Index associated to each model in the result.json file
idx_model = {'SG': 0, 'TK': 1, 'P1': 2, 'P2': 2, 'GE': 3, 'TK+': 1, 'P1+': 2, 'P2+': 2 , 'GE+': 3 }

# Set colors
dark2 = plt.cm.get_cmap('Dark2', 8)
tab20 = plt.cm.get_cmap('tab20', 20)
COLORS = np.concatenate([tab20(np.linspace(0, 1, 20)), dark2(np.linspace(0, 1, 8))], axis=0)


def compute_bias(save=False, lottery=0):
    """
    Computes the bias for each monkey in a given lottery or set of lotteries.

    Bias is calculated as the absolute difference between the proportion of responses
    classified as 0 and those classified as 1. The results are displayed in a sorted table,
    and optionally saved as a JSON file.

    Parameters:
    -----------
    save : bool, optional (default=False)
        If True, saves the computed biases to a JSON file ('monkey_biases.json').

    lottery : int or list, optional (default=0)
        - If an integer, computes bias for the specified lottery.
        - If a list, computes bias across multiple lotteries by merging responses.

    Returns:
    --------
    - Displays a sorted table of monkeys and their bias values.
    - Saves the computed biases to a JSON file if `save=True`.
    """
    from monkey import monkeys  # Import the monkey dataset

    # Initialize containers for results
    all_R = []  # List to store computed bias values
    all_monkeys = []  # List to store monkey names
    biases = {}  # Dictionary to store biases per monkey

    # Iterate through each monkey in the dataset
    for monkey in monkeys:
        if isinstance(lottery, list):
            # If multiple lotteries are provided, merge their responses
            trials, responses = [], []
            for lot in lottery:
                t, r = monkey.get_data(lottery=lot)  # Get data for each lottery
                trials.append(t)
                responses.append(r)
            responses = np.concatenate(responses)  # Merge all responses
        else:
            # Single lottery case
            trials, responses = monkey.get_data(lottery=lottery)

        # Compute the proportion of responses that are 0 and 1
        c0 = np.count_nonzero(responses == 0) / len(responses)
        c1 = np.count_nonzero(responses == 1) / len(responses)

        # Calculate bias as the absolute difference between c0 and c1
        bias = abs(c0 - c1)

        # Store results
        all_monkeys.append(monkey.shortname)
        all_R.append(bias)
        biases[monkey.shortname] = bias

    # Create a PrettyTable to display results in a structured format
    table = PrettyTable(border=True, align="l")
    table.set_style(SINGLE_BORDER)
    table.field_names = ["Players", "Bias"]

    # Sort the results by bias value in ascending order
    ordered_idx = np.argsort(all_R)
    for i in ordered_idx:
        table.add_row([all_monkeys[i], str(round(all_R[i], 3))])  # Add rows with rounded bias values

    # Print the table to the console
    print(table)

    # Save results to a JSON file if requested
    if save:
        with open('monkey_biases.json', 'w') as json_file:
            json.dump(biases, json_file, indent=4)  # Save with indentation for readability


def plot_bias_vs_score(plot_ratio=False, ax=None, result_file='', biases_file='./data/monkey_biases.json', model='TK'):
    """
    Plots monkey bias against score or score ratio.

    This function loads bias values and model scores from given files, then plots
    the relationship between bias and either absolute score or the score ratio.
    It also performs a linear regression and highlights regions where bias exceeds 0.4.

    Parameters:
    -----------
    plot_ratio : bool, optional (default=False)
        If True, plots the ratio SG/PT instead of the absolute SG score.

    ax : matplotlib.axes._subplots.AxesSubplot, optional (default=None)
        The Matplotlib axis to plot on. If None, the function assumes an existing axis.

    result_file : str, required
        Path to the file containing monkey scores (pickled dictionary).

    biases_file : str, optional (default='./data/monkey_biases.json')
        Path to the JSON file containing monkey biases.

    model : str, optional (default='TK')
        The model key to compare against in score ratios.

    Returns:
    --------
    ax : matplotlib.axes._subplots.AxesSubplot
        The axis containing the plot.

    all_monkeys : list
        List of monkey names included in the plot.

    Example Usage:
    --------------
    ```
    fig, ax = plt.subplots()
    plot_bias_vs_score(plot_ratio=True, ax=ax, result_file='scores.pkl')
    plt.show()
    ```
    """

    # Load bias data from JSON file
    with open(biases_file, 'r') as json_file:
        biases = json.load(json_file)

    # Load score data from pickled file
    with open(result_file, 'rb') as file:
        score = pickle.load(file)

    # Initialize lists to store relevant data
    all_score_SG = []  # Stores SG scores
    all_ratio_SG_PT = []  # Stores SG/PT score ratios
    all_bias = []  # Stores monkey biases
    all_monkeys = []  # Stores monkey names

    # Extract and filter monkey data
    for mk in score:
        if mk not in ['GAN', 'OLA']:  # Exclude specific monkeys from analysis
            all_monkeys.append(mk[:4])  # Store first 4 characters of monkey name
            all_score_SG.append(float(score[mk][idx_model['SG']]))  # SG model score
            all_ratio_SG_PT.append(
                float(score[mk][idx_model['SG']]) / float(score[mk][idx_model[model]])
            )  # Compute SG/PT ratio
            all_bias.append(biases[mk])  # Store monkey bias

    # Determine what to plot: absolute score or score ratio
    to_plot = all_ratio_SG_PT if plot_ratio else all_score_SG

    # Highlight region where bias > 0.4
    ax.fill_betweenx(
        [0.94, 0.82],
        0.4,  # Start of bias threshold
        max(all_bias) + 0.06,
        color='red',
        alpha=0.15,
        label='bias > 0.4'
    )

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Scatter plot of bias vs. score/score ratio
    for i in range(len(all_monkeys)):
        ax.scatter(all_bias[i], to_plot[i], label=None, color=COLORS[i])

    # Perform linear regression to analyze relationship
    slope, intercept, r_value, p_value, std_err = stats.linregress(all_bias, to_plot)
    regression_line = np.array(all_bias) * slope + intercept  # Compute regression line

    # Plot regression line
    ax.plot(all_bias, regression_line, color='red', label=f'r={r_value:.2f}, p={p_value:.3f}')
    ax.legend(loc='upper left')

    # Set plot labels and title
    short_title = 'SG/P1' if plot_ratio else 'SG score'
    ax.set_xlabel('Monkey Bias')
    ax.set_ylabel(short_title)
    ax.set_title(f'{short_title} vs Monkey Bias')

    return ax, all_monkeys


def plot_bias_vs_score_diff(ax=None, result_file='', biases_file='./data/monkey_biases.json', model='TK'):
    """
    Plots the absolute difference between SG and another model's score against monkey bias.

    This function loads bias values and model scores from given files, then plots
    the relationship between bias and the absolute difference in scores.
    It also highlights a region where the score difference is small (< 0.05)
    and performs a linear regression.

    Parameters:
    -----------
    ax : matplotlib.axes._subplots.AxesSubplot, optional (default=None)
        The Matplotlib axis to plot on. If None, the function assumes an existing axis.

    result_file : str, required
        Path to the file containing monkey scores (pickled dictionary).

    biases_file : str, optional (default='./data/monkey_biases.json')
        Path to the JSON file containing monkey biases.

    model : str, optional (default='TK')
        The model key to compare against SG scores.

    Returns:
    --------
    ax : matplotlib.axes._subplots.AxesSubplot
        The axis containing the plot.

    all_monkeys : list
        List of monkey names included in the plot.

    scatter_labels : list
        Labels corresponding to each scatter point.

    scatter_handles : list
        Handles for the scatter plot elements.

    Example Usage:
    --------------
    ```
    fig, ax = plt.subplots()
    plot_bias_vs_score_diff(ax=ax, result_file='scores.pkl')
    plt.show()
    ```
    """

    # Load bias data from JSON file
    with open(biases_file, 'r') as json_file:
        biases = json.load(json_file)

    # Load score data from pickled file
    with open(result_file, 'rb') as file:
        score = pickle.load(file)

    # Initialize lists to store relevant data
    all_score_SG = []  # SG model scores
    all_score_PT = []  # Scores from the compared model (e.g., TK)
    all_bias = []  # Monkey biases
    diff = []  # Absolute score differences
    all_monkeys = []  # Monkey names

    # Extract and filter monkey data
    for mk in score:
        if mk not in ['GAN', 'OLA']:  # Exclude specific monkeys from analysis
            all_monkeys.append(mk)
            all_score_SG.append(float(score[mk][idx_model['SG']]))  # SG model score
            all_score_PT.append(float(score[mk][idx_model[model]]))  # Compared model score
            diff.append(abs(float(score[mk][0]) - float(score[mk][idx_model[model]])))  # Compute score difference
            all_bias.append(biases[mk])  # Store monkey bias

    # Highlight region where bias > 0.4 and difference < 0.05
    fill_bet = ax.fill_betweenx(
        [0, 0.055],  # Y-axis range for shading
        0.4,  # Start of bias threshold
        max(all_bias) + 0.06,  # Extend slightly beyond max bias
        color='red',
        alpha=0.15,
        label='$\Delta(SG-TK) < 0.05$'
    )

    # Initialize lists to store scatter plot elements for legend
    scatter_handles = []
    scatter_labels = []

    # Plot data points
    for i in range(len(all_monkeys)):
        scatter = ax.scatter(all_bias[i], diff[i], color=COLORS[i], marker='o', label=all_monkeys[i])
        scatter_handles.append(scatter)
        scatter_labels.append(all_monkeys[i])

    # Perform linear regression to analyze relationship
    slope, intercept, r_value, p_value, std_err = stats.linregress(all_bias, diff)
    regression_line = np.array(all_bias) * slope + intercept  # Compute regression line

    # Plot regression line
    regression_line, = ax.plot(all_bias, regression_line, color='red', label=f'r={r_value:.2f}, p={p_value:.3f}')

    # Add legend for regression line and shaded area
    reg_legend = ax.legend(handles=[regression_line, fill_bet], loc='upper right')
    ax.add_artist(reg_legend)

    # Set plot limits, labels, and title
    ax.set_ylim(-0.05, 0.15)
    ax.set_xlabel('Monkey Bias')
    ax.set_ylabel('$\Delta(SG-TK)$')
    ax.set_title('$\Delta(SG-TK)$ vs Monkey Bias')

    # Remove unnecessary grid lines and spines for cleaner visualization
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Identify and print excluded monkeys based on criteria
    print('Excluded monkeys: ')
    for i in range(len(all_monkeys)):
        epsilon = all_score_PT[i] - all_score_SG[i]
        if epsilon < 0.055 and all_bias[i] > 0.4:
            print(all_monkeys[i])
    return ax, all_monkeys, scatter_labels, scatter_handles


def plot_bias_analysis(result_file, biases_file='./data/monkey_biases.json', model='TK'):
    """
    Generates a bias analysis plot with two subplots:
    1. Score vs. bias
    2. Score difference vs. bias

    This function creates a figure with two subplots to analyze the relationship
    between monkey biases and model scores.

    Parameters:
    -----------
    result_file : str
        Path to the file containing model scores (pickled dictionary).

    biases_file : str, optional (default='./data/monkey_biases.json')
        Path to the JSON file containing monkey biases.

    model : str, optional (default='TK')
        The model key to compare against SG scores.

    Returns:
    --------
    None (Displays the plot)
    """

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Score vs Bias
    ax1, _ = plot_bias_vs_score(plot_ratio=False, ax=ax1, result_file=result_file, biases_file=biases_file, model=model)

    # Plot Score Difference vs Bias
    ax2, _, combined_labels, combined_handles = plot_bias_vs_score_diff(ax=ax2, result_file=result_file, model=model)

    # Adjust layout and add a legend
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    fig.legend(
        handles=combined_handles,
        labels=combined_labels,
        bbox_to_anchor=(0.5, 0.14),
        loc='upper center',
        fontsize='large',
        frameon=False,
        ncol=11
    )

    # Display the plot
    plt.show()


def plot_score_difference(control_fit_file='results-fits/monkey-analysis-L0train-L67test-score.pkl',
                          risky_fit_file='results-fits/monkey-analysis-L0train-L12345test-score.pkl'):
    """
    Plots the score differences between SG and TK models for two conditions:
    1. Risky lotteries (types 6 and 7)
    2. Control lotteries (types 1 to 5)

    This function loads scores from two files and visualizes the comparison
    between SG and TK models across different monkeys.

    Parameters:
    -----------
    control_fit_file : str, optional (default='results-fits/monkey-analysis-L0train-L67test-score.pkl')
        Path to the file containing scores for the control condition (pickled dictionary).

    risky_fit_file : str, optional (default='results-fits/monkey-analysis-L0train-L12345test-score.pkl')
        Path to the file containing scores for the risky condition (pickled dictionary).

    Returns:
    --------
    None (Displays the plot)
    """

    # Load the first set of scores (risky lotteries)
    with open(control_fit_file, 'rb') as file:
        score_risk = pickle.load(file)

    all_score_SG_risk = []
    all_score_PT_risk = []
    all_monkeys_risk = []

    # Extract data for the risky condition
    for mk in score_risk:
        if mk not in ['GAN', 'OLA', 'YOH']:  # Exclude specific monkeys
            all_monkeys_risk.append(mk)
            all_score_SG_risk.append(float(score_risk[mk][0]))  # SG model (index 0)
            all_score_PT_risk.append(float(score_risk[mk][1]))  # TK model (index 1)

    # Load the second set of scores (control lotteries)
    with open(risky_fit_file, 'rb') as file:
        score_control = pickle.load(file)

    all_score_SG_control = []
    all_score_PT_control = []
    all_monkeys_control = []

    # Extract data for the control condition
    for mk in score_control:
        if mk not in ['GAN', 'OLA', 'YOH']:  # Exclude specific monkeys
            all_monkeys_control.append(mk)
            all_score_SG_control.append(float(score_control[mk][0]))  # SG model (index 0)
            all_score_PT_control.append(float(score_control[mk][1]))  # TK model (index 1)

    # Create a figure with two subplots
    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

    # First subplot (Risky lotteries)
    ax1.scatter(all_score_SG_risk, all_monkeys_risk, color='red', label='SG', marker='o', facecolor='none')
    ax1.scatter(all_score_PT_risk, all_monkeys_risk, color='blue', label='TK', marker='o')

    # Connect data points with dashed lines
    for i, monkey in enumerate(all_monkeys_risk):
        ax1.plot([all_score_SG_risk[i], all_score_PT_risk[i]], [monkey, monkey], 'k--')

    # Formatting the first subplot
    ax1.set_title('Risky lotteries (types 6 and 7)')
    ax1.set_xlabel('Score', fontsize=12)
    ax1.set_ylabel('Monkey', fontsize=12)
    ax1.set_xticks([0.75, 1])
    ax1.legend(loc='best')
    ax1.invert_yaxis()  # Invert the y-axis for consistency
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='both', which='major', labelsize=12)

    # Second subplot (Control lotteries)
    ax2.scatter(all_score_SG_control, all_monkeys_control, color='red', label='SG', marker='o', facecolor='none')
    ax2.scatter(all_score_PT_control, all_monkeys_control, color='blue', label='TK', marker='o')

    # Connect data points with dashed lines
    for i, monkey in enumerate(all_monkeys_control):
        ax2.plot([all_score_SG_control[i], all_score_PT_control[i]], [monkey, monkey], 'k--')

    # Formatting the second subplot
    ax2.set_title('Control lotteries (types 1, 2, 3, 4, 5)')
    ax2.set_xlabel('Score', fontsize=12)
    ax2.legend(loc='best')
    ax2.set_xticks([0.75, 1])
    ax2.invert_yaxis()  # Match order with the first subplot
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='both', which='major', labelsize=12)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_bias_analysis(result_file='results-fits/monkey-analysis-L0train-L67test-score.pkl', biases_file='./data/monkey_biases.json', model='TK')
    control_fit_file = 'results-fits/monkey-analysis-L0train-L67test-score.pkl'
    risky_fit_file = 'results-fits/monkey-analysis-L0train-L12345test-score.pkl'
    plot_score_difference(control_fit_file, risky_fit_file)




















