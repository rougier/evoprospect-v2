# Social hierarchy influences monkeys' risky decisions
# Copyright 2024 (c) Naomi Chaix-Echel & Nicolas P Rougier
# Released under a BSD 2-clauses license

"""
This script provides a collection of statistical functions for analyzing experimental data, 
performing hypothesis testing, and evaluating model performance. 

Functions:
----------
1. pairwise_t_test(df)
   - Performs pairwise independent t-tests between all groups in a given DataFrame.
   - Returns t-statistics and p-values for each pair.

2. compute_mean_parameter_values(excluded_monkeys, models)
   - Computes the mean parameter values for different models while excluding certain subjects.
   - Returns a DataFrame containing the averaged parameter values per model.

3. plot_score_selected_models(file_scores, file_params)
   - Loads model scores and parameters, formats them into a table, and highlights the best/worst performers.
   - Uses ANSI color codes for terminal output.

4. compute_ANOVA(data_dict)
   - Conducts a one-way ANOVA test to compare multiple groups.

5. compute_average_score(params, lottery)
   - Computes and displays the average model score for different experimental conditions.
   - Uses Bayesian Information Criterion (BIC) for model evaluation.

6. calculate_bic(y, y_pred, p)
   - Computes the Bayesian Information Criterion (BIC) given actual vs. predicted values.
   - Helps in comparing different models based on goodness-of-fit.

7. compute_mean_BIC(params, lottery)
   - Calculates and displays the mean BIC values across different models.
   - Performs ANOVA on BIC values to assess statistical significance.
"""

import pickle
from scipy import stats
from prettytable import SINGLE_BORDER
import pandas as pd
from player import *
from monkey import monkeys
from sklearn.metrics import mean_squared_error
from itertools import combinations


def pairwise_t_test(df):
    """
        Performs independent t-tests between all possible pairs of groups (columns) in a given DataFrame.

        The function assumes that each column in the DataFrame represents a different group, and it
        performs Welch’s t-test (which does not assume equal variances) for each pair of columns.

        Parameters:
        df (pd.DataFrame): A DataFrame where each column represents a different group.

        Returns:
        list of tuples: A list containing tuples of the form (group1, group2, t_statistic, p_value).
        """
    results = []
    groups = list(df.columns)
    for (group1, group2) in combinations(groups, 2):
        t_stat, p_val = stats.ttest_ind(df[group1].dropna(), df[group2].dropna(), equal_var=False)
        results.append((group1, group2, t_stat, p_val))
    return results


def compute_mean_parameter_values(excluded_monkeys=['ANU', 'NER', 'YIN', 'OLG', 'JEA', 'PAT', 'YOH', 'GAN'],
                                  models=['SG', 'TK', 'P1', 'P2', 'GE', 'TK+', 'P1+', 'P2+', 'GE+'] ):
    """
        Computes the mean parameter values for each model across monkeys, excluding specified monkeys.

        This function loads parameter data from a pickle file, processes it to extract parameter values for each model,
        and computes the mean parameter values, returning a formatted DataFrame.

        Parameters:
        excluded_monkeys (list, optional): A list of monkey identifiers to exclude from analysis.
                                            Defaults to ['ANU', 'NER', 'YIN', 'OLG', 'JEA', 'PAT', 'YOH', 'GAN'].
        models (list, optional): A list of model names to include in the analysis.
                                 Defaults to ['SG', 'TK', 'P1', 'P2', 'GE', 'TK+', 'P1+', 'P2+', 'GE+'].
        Returns:
        pd.DataFrame: A DataFrame where each row represents a model, and columns represent mean parameter values.
        """
    with open('results-fits/monkey-analysis-L0-params.pkl', 'rb') as file:
        params = pickle.load(file)
    means = {}  
    vals = {}
    for model in models:
        vals[model] = {}
    for monkey in params.keys():
        if monkey not in excluded_monkeys:
            for i in range(len(params[monkey])):
                for param in params[monkey][i].parameters.keys():
                    if param not in vals[params[monkey][i].shortname]:
                        vals[params[monkey][i].shortname][param] = [params[monkey][i].parameters[param]]
                    else:
                        vals[params[monkey][i].shortname][param].append(params[monkey][i].parameters[param])
    for model in vals:
        means[model] = {}
        for param in vals[model]:
            means[model][param] = np.mean(vals[model][param])
    df = pd.DataFrame(means).T
    df = df.reset_index().rename(columns={'index': 'Model_Type'})
    return df


def plot_score_selected_models(file_scores='results-fits/monkey-analysis-L0-score.pkl',
                               file_params='results-fits/monkey-analysis-L0-params.pkl'):
    """
        Generates a formatted table displaying scores of selected models for each monkey.

        This function loads score and parameter data from pickle files, extracts scores for specific models,
        and highlights the minimum and maximum scores.

        Parameters:
        file_scores (str, optional): Path to the pickle file containing score data.
                                     Defaults to 'results-fits/monkey-analysis-L0-score.pkl'.
        file_params (str, optional): Path to the pickle file containing parameter data.
                                     Defaults to 'results-fits/monkey-analysis-L0-params.pkl'.

        Returns:
        None: The function prints a formatted table with color-coded scores.
        """

    with open(file_scores, 'rb') as file:
        score = pickle.load(file)

    with open(file_params, 'rb') as file:
        params = pickle.load(file)

    # ANSI escape codes for colors
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

    x = PrettyTable(border=True, align="l")
    x.set_style(SINGLE_BORDER)

    selected_players = ['SG', 'P1', 'TK', 'P1+', 'TK+']

    # selected_players = ['SG', 'P1', 'TK','GE','P2', 'P1+', 'TK+','GE+','P2+']
    x.field_names = (["Players"] + selected_players)
    # Function to color values
    def color_value(value, is_min, is_max):
        if is_min:
            return RED + str(value) + RESET
        elif is_max:
            return BLUE + str(value) + RESET
        return str(value)

    # Process each monkey's scores
    for monkey in score.keys():
        R = []

        # Collect all scores for the current monkey
        all_scores = []
        for player_name in selected_players:
            for k in range(len(params[monkey])):
                if params[monkey][k].shortname == player_name:
                    all_scores.append(score[monkey][k])

        if all_scores:
            min_val = min(all_scores)
            max_val = max(all_scores)

            # Colorize scores
            for player_name in selected_players:
                player_scores = [score[monkey][k] for k in range(len(params[monkey])) if
                                 params[monkey][k].shortname == player_name]
                if player_scores:
                    colored_scores = [color_value(val, val == min_val, val == max_val) for val in player_scores]
                    R.extend(colored_scores)
                else:
                    R.extend([''] * len(selected_players))
        else:
            # If no scores for this monkey, add empty cells
            R.extend([''] * len(selected_players))

        # Add row to PrettyTable
        x.add_row([monkey] + R)

    # Print the table with colored output
    print(x)


def compute_ANOVA(data_dict):
    """
        Performs a one-way ANOVA (Analysis of Variance) on a given dataset.

        Parameters:
        data_dict (dict): A dictionary where keys are group labels, and values are lists or arrays of numerical data.

        Returns:
        None: The function prints the ANOVA results, including F-value, p-value, sum of squares,
              degrees of freedom, mean squares, and effect size.
        """
    # Convert the dictionary to a list of arrays
    groups = list(data_dict.values())

    # Perform one-way ANOVA
    f_value, p_value = stats.f_oneway(*groups)

    # Calculate additional statistics
    grand_mean = np.mean([np.mean(group) for group in groups])
    total_ss = sum(sum((value - grand_mean) ** 2 for value in group) for group in groups)
    between_ss = sum(len(group) * (np.mean(group) - grand_mean) ** 2 for group in groups)
    within_ss = total_ss - between_ss

    # Degrees of freedom
    df_between = len(groups) - 1
    df_within = sum(len(group) for group in groups) - len(groups)
    df_total = sum(len(group) for group in groups) - 1

    # Mean squares
    ms_between = between_ss / df_between
    ms_within = within_ss / df_within

    # Effect size (Eta-squared)
    eta_squared = between_ss / total_ss

    # Print results
    print(f"F-value: {f_value:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Degrees of freedom: Between groups = {df_between}, Within groups = {df_within}")
    print(f"Sum of squares: Between groups = {between_ss:.4f}, Within groups = {within_ss:.4f}")
    print(f"Mean squares: Between groups = {ms_between:.4f}, Within groups = {ms_within:.4f}")
    print(f"Eta-squared (effect size): {eta_squared:.4f}")


def compute_average_score(filename, outlier_monkey='YOH', impacted_by_outlier=['TK', 'P2']):
    """ Compute average score of each model and check if the results are statistically significant"""

    with open(filename, 'rb') as file:
        init_data = pickle.load(file)

    # Determine the number of indices (length of each list)
    num_indices = len(next(iter(init_data.values())))
    models = ['SG', 'TK', 'P1', 'P2', 'GE', 'TK+', 'P1+', 'P2+', 'GE+']

    # Initialize sums, counts, and lists for std calculation
    sums = [0.0] * num_indices
    counts = [0] * num_indices
    data = {}
    for model in models:
        data[model] = []

    # Calculate the sum, count, and collect all values for each index
    for values in init_data.values():
        for i, value in enumerate(values):
            float_value = float(value)
            sums[i] += float_value
            counts[i] += 1
            data[models[i]].append(float_value)

    # ANOVA
    compute_ANOVA(data)

    if impacted_by_outlier is not None:
        outlier_idx = list(init_data.keys()).index(outlier_monkey)
        for key in impacted_by_outlier:
            data[key].pop(outlier_idx)

    #post-hoc Tukey's HSD (Honestly Significant Difference)

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame({k: pd.Series(v) for k, v in data.items()})

    pairwise_results = pairwise_t_test(df)

    # Apply Bonferroni correction
    n_comparisons = len(pairwise_results)
    alpha = 0.05  # significance level
    bonferroni_threshold = alpha / n_comparisons

    # Sort results by p-value
    pairwise_results.sort(key=lambda x: x[3])

    print("\nTop 10 significant pairwise comparisons (Bonferroni-corrected):")
    for i, (group1, group2, t_stat, p_val) in enumerate(pairwise_results[:10], 1):
        significant = "Yes" if p_val < bonferroni_threshold else "No"
        print(
            f"{i}. {group1} vs {group2}: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}, Significant: {significant}")

    # Calculate means and standard deviations for each group
    means = {key: np.mean(values) for key, values in data.items()}
    stds = {key: np.std(values, ddof=1) for key, values in data.items()}

    # Create a summary dataframe
    summary_df = pd.DataFrame({
        'Mean': means,
        'Std Dev': stds
    })

    print("\n Mean score per model:")
    print(summary_df)


def calculate_bic(y, y_pred, p):
    """
        Computes the Bayesian Information Criterion (BIC) for a given model.

        The BIC is used to evaluate model fit while penalizing complexity.
        A lower BIC value indicates a better balance between fit and complexity.

        Parameters:
        y (array-like): True observed values.
        y_pred (array-like): Predicted values from the model.
        p (int): Number of parameters in the model.

        Returns:
        float: The BIC value."""
    n = len(y)
    mse = mean_squared_error(y, y_pred)
    bic_value = n * np.log(mse) + p * np.log(n)
    return bic_value


def compute_mean_BIC(params,lottery):
    # Initialize PrettyTable with single border style
    x = PrettyTable(border=True)
    x.set_style(SINGLE_BORDER)

    # Define player models and extract model shortnames
    player_models = [SigmoidPlayer, ProspectPlayerTK, ProspectPlayerP1, ProspectPlayerP2,
                     ProspectPlayerGE, DualProspectPlayerTK, DualProspectPlayerP1,
                     DualProspectPlayerP2, DualProspectPlayerGE]
    model_names = [p.shortname for p in player_models]

    # Set column headers for the table
    x.field_names = ["Players"] + model_names

    # Dictionary to store BIC values per model
    bic_values_per_model = {model: [] for model in model_names}

    for monkey in monkeys:
        if isinstance(lottery, list):
            trials, responses = [], []
            for lot in lottery:
                t, r = monkey.get_data(lottery=lot)
                trials.append(t)
                responses.append(r)
            trials = np.concatenate(trials)
            responses = np.concatenate(responses)
        else:
            trials, responses = monkey.get_data(lottery=lottery)

        for player in params[monkey.shortname]:
            responses_pred = player.play(trials)
            n_params = len(player.parameters.keys())
            BIC = calculate_bic(responses, responses_pred, n_params)
            rounded_bic = round(BIC, 0)
            bic_values_per_model[player.shortname].append(rounded_bic)

    compute_ANOVA(bic_values_per_model)
    # Compute the mean BIC for each model and add to the table
    mean_bics = []
    std_bics = []
    for model_name in model_names:
        model_bics = bic_values_per_model[model_name]
        mean_bic = round(np.nanmean(model_bics), 0) if model_bics else float('nan')
        std_bic = round(np.nanstd(model_bics), 0) if model_bics else float('nan')
        mean_bics.append(mean_bic)
        std_bics.append(std_bic)
    x.add_row(["Mean BIC"] + mean_bics)
    x.add_row(["Std BIC"] + std_bics)

    print(x)


if __name__ == "__main__":

    compute_mean_parameter_values()












