# Copyright 2024 (c) Naomi Chaix-Echel & Nicolas P Rougier
# Released under a BSD 2-clauses license
import pickle
from scipy import stats
from prettytable import SINGLE_BORDER
import pandas as pd
from player import *
from monkey import monkeys
from sklearn.metrics import mean_squared_error
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations


def pairwise_t_test(df):
        results = []
        groups = list(df.columns)
        for (group1, group2) in combinations(groups, 2):
            t_stat, p_val = stats.ttest_ind(df[group1].dropna(), df[group2].dropna(), equal_var=False)
            results.append((group1, group2, t_stat, p_val))
        return results


def compute_mean_parameter_values(excluded_monkeys=['ANU', 'NER', 'YIN', 'OLG', 'JEA', 'PAT', 'YOH', 'GAN'],
                                  models=['SG', 'TK', 'P1', 'P2', 'GE', 'TK+', 'P1+', 'P2+', 'GE+'] ):
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












