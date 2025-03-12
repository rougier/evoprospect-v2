# Copyright 2024 (c) Naomi Chaix-Echel & Nicolas P Rougier
# Released under a BSD 2-clauses license

from player import *
from scipy.signal import savgol_filter
import matplotlib.colors as cm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import math
import datetime
import warnings
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import linregress
warnings.filterwarnings('ignore')

colors = [cm.to_hex(plt.cm.tab20(i)) for i in range(24)]

#selected monkeys for the ELO analysis (biased monkeys rejected)
MONKEYS = ['abr', 'ala', 'alv', 'bar', 'ber', 'ces', 'dor', 'eri', 'fic', 'her', 'hor', 'las',
           'nem', 'oll', 'pac', 'yoh', 'gan', 'ola']

REJECTED_MONKEYS = ['anu', 'jea', 'ner', 'yin', 'olg', 'pat']

all_monkeys = np.concatenate((MONKEYS, REJECTED_MONKEYS))
monkey_colors = dict(zip(all_monkeys, colors))


def plot_all_elo_rates(smoothed=False):
    elo_rates = pd.read_excel('./data/elo_matrix_Tonk.xlsx')

    elo_rates.rename(columns={'oli': 'oll'}, inplace=True)
    cols = np.concatenate((['Date'], MONKEYS))
    elo_rates = elo_rates[cols]
    elo_rates['Date'] = pd.to_datetime(elo_rates['Date'])
    elo_rates.set_index('Date', inplace=True)


    # Filter data to include only dates from 2020
    elo_rates = elo_rates[
        (elo_rates.index >= '2020-01-01') &
        (elo_rates.index <= '2023-08-01')]

    dates = np.array(elo_rates.index.tolist())
    periods = pd.read_csv('./data/elo_periods.csv')

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.suptitle('Elo Ratings Over Time', fontsize=18, fontweight='bold')
    handles = []
    labels = []
    monkeys = [m for m in MONKEYS if m not in REJECTED_MONKEYS]
    # Loop over columns and plot each one
    """for i, monkey in enumerate(monkeys):
        print(monkey)
        start_1500 = periods[periods['mkname'] == monkey]['start']
        end_1500 = periods[periods['mkname'] == monkey]['end']

        if monkey == "her":
            y = np.array(elo_rates[elo_rates.index >= '2022-05-01'][monkey])
        elif monkey == 'hor':
            y = np.array(elo_rates[elo_rates.index >= '2021-09-01'][monkey])
        elif monkey in ['bar', 'abr', 'ala', 'ces']:
            y = np.array(elo_rates[elo_rates.index <= '2022-07-01'][monkey])
        else:
            y = np.array(elo_rates[monkey])

        # Apply the Savitzky-Golay filter if smoothed is True
        if smoothed:

            mask = np.isfinite(y)
            if monkey == 'her':
                x_dates = dates[dates >= pd.Timestamp('2022-05-01')][mask]
            elif monkey == 'hor':
                x_dates = dates[dates >= pd.Timestamp('2021-09-01')][mask]
            elif monkey in ['bar', 'abr', 'ala', 'ces']:
                x_dates = dates[dates <= pd.Timestamp('2022-07-01')][mask]
            else:
                x_dates = dates[mask]
            #elif monkey in ['pac', 'nem', 'ber', 'yoh', 'oll', 'fic', 'las', 'dor', 'gan']:
             #   x_dates = dates


            smoothed_data = savgol_filter(y[mask], window_length=31, polyorder=3)
            line, =  plt.plot(x_dates, smoothed_data, label=monkey, linestyle='solid', color=monkey_colors[monkey])
            # Append the line to handles and its label to labels
            handles.append(line)
            labels.append(monkey)

            for s, e in zip(start_1500, end_1500):
                # Find the indices of the x_dates that are between start_1500 and end_1500
                mask_thick = (x_dates >= pd.Timestamp(s)) & (x_dates <= pd.Timestamp(e))
                # Plot the thicker line only for the period within the start and end range
                plt.plot(x_dates[mask_thick], smoothed_data[mask_thick], linestyle='solid',
                         color=monkey_colors[monkey], linewidth=5)  # Thicker line
                # Add shaded vertical areas to highlight the periods
                #plt.axvspan(pd.Timestamp(s), pd.Timestamp(e), color='gray', alpha=0.3)  # Adjust color and transparency

                # Extract the corresponding y values in this range
                #y_in_period = smoothed_data[mask_thick]
                #print(y_in_period)
                # Calculate min and max of y within this period
                #if len(y_in_period) != 0 :
                #    y_min = np.min(y_in_period)
                #    y_max = np.max(y_in_period)
                    # Use fill_between to shade between the min and max y values
                 #   plt.fill_between(x_dates[mask_thick], y_min - 30, y_max+30, color='gray', alpha=0.3)
        else:
            plt.plot(elo_rates.index, elo_rates[monkey], label=monkey, linestyle='solid', color=monkey_colors[monkey])"""
    for i, monkey in enumerate(monkeys):
        start_1500 = periods[periods['mkname'] == monkey]['start']
        end_1500 = periods[periods['mkname'] == monkey]['end']

        if monkey == "her":
            y = np.array(elo_rates[elo_rates.index >= '2022-05-01'][monkey])
        elif monkey == 'hor':
            y = np.array(elo_rates[elo_rates.index >= '2021-09-01'][monkey])
        elif monkey in ['bar', 'abr', 'ala', 'ces']:
            y = np.array(elo_rates[elo_rates.index <= '2022-07-01'][monkey])
        else:
            y = np.array(elo_rates[monkey])

        # Apply the Savitzky-Golay filter if smoothed is True
        if smoothed:
            mask = np.isfinite(y)
            if monkey == 'her':
                x_dates = dates[dates >= pd.Timestamp('2022-05-01')][mask]
            elif monkey == 'hor':
                x_dates = dates[dates >= pd.Timestamp('2021-09-01')][mask]
            elif monkey in ['bar', 'abr', 'ala', 'ces']:
                x_dates = dates[dates <= pd.Timestamp('2022-07-01')][mask]
            else:
                x_dates = dates[mask]

            smoothed_data = savgol_filter(y[mask], window_length=31, polyorder=3)

            # Plot the regular curve in its original color
            line, = plt.plot(x_dates, smoothed_data, label=monkey, linestyle='solid', color=monkey_colors[monkey])
            handles.append(line)
            labels.append(monkey)

            # If the monkey is in the specified list, make the curve white between '2022-08-01' and '2023-01-01'
            if monkey in ['pac', 'nem', 'ber', 'yoh', 'oll', 'fic', 'las', 'dor', 'gan','her', 'hor', 'eri']:
                mask_white = (x_dates >= pd.Timestamp('2022-11-01')) & (x_dates <= pd.Timestamp('2023-01-20'))
                plt.plot(x_dates[mask_white], smoothed_data[mask_white], color='white', linewidth=2)

            for s, e in zip(start_1500, end_1500):
                mask_thick = (x_dates >= pd.Timestamp(s)) & (x_dates <= pd.Timestamp(e))
                plt.plot(x_dates[mask_thick], smoothed_data[mask_thick], linestyle='solid',
                         color=monkey_colors[monkey], linewidth=5)

        else:
            plt.plot(elo_rates.index, elo_rates[monkey], label=monkey, linestyle='solid', color=monkey_colors[monkey])

            # Add white curve between '2022-08-01' and '2023-01-01' for the specified monkeys
            if monkey in ['pac', 'nem', 'ber', 'yoh', 'oll', 'fic', 'las', 'dor', 'gan']:
                mask_white = (elo_rates.index >= pd.Timestamp('2022-10-01')) & (
                            elo_rates.index <= pd.Timestamp('2023-01-01'))
                plt.plot(elo_rates.index[mask_white], elo_rates[monkey][mask_white], color='white', linewidth=2)

    # Create custom legend for the "1500 best" thick line with grey area
    custom_line = Line2D([0], [0], color='black', linewidth=4)  # Thicker line in black for the legend
    custom_patch = Patch(facecolor='gray', alpha=0.3)  # Grey shaded area for the legend

    # Create a combined legend entry with a thick line and grey area
    combined_custom = [custom_patch, custom_line]
    # Add labels, title, and legend
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Elo-score', fontsize=14)
    # Add the custom legend items to the handles and labels
    #handles.append(custom_line)
    handles.append(custom_patch)
    labels.append("1500 best")  # Add custom label

    # Combine existing and custom legends
    plt.legend(handles, labels, bbox_to_anchor=(1.01, 1), ncol=1, fontsize=14)
    plt.xticks(fontsize=12)  # Increase fontsize for x-ticks
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_scatter_regression(df, x_col, y_col, ax):
    # Convert column to numeric
    df[x_col] = pd.to_numeric(df[x_col])
    df[y_col] = pd.to_numeric(df[y_col])

    # Calculate regression coefficients
    coeffs = np.polyfit(df[x_col], df[y_col], 1)
    f = np.poly1d(coeffs)

    # Calculate regression metrics
    mse = mean_squared_error(df[y_col], f(df[x_col]))
    r2 = r2_score(df[y_col], f(df[x_col]))
    res = linregress(df[x_col], df[y_col])
    p_value = res.pvalue

    # Plot scatter plot
    #ax.scatter(df[x_col], df[y_col])

    # Plot regression line
    x = np.linspace(min(df[x_col]), max(df[x_col]), 100)
    ax.plot(x, f(x), color='red')

    # Display regression metrics
    ax.text(0.05, 1.05, f'R-squared: {res.rvalue**2:.2f}\np-value: {res.pvalue:.2f}', transform=ax.transAxes,size=13)


def plot_linear_regression(df_PT):
    # Linear regression Elo vs. risk aversion parameter
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    fig.suptitle(f'Risk aversion vs Elo-score with model {df_PT["model"][0]}')
    # Plotting Rho gain
    axes[0].set_title('Gain domain', fontsize=14, fontweight='bold')
    for index, row in df_PT.iterrows():
        if row['monkey'] in MONKEYS:
            if row['model'] == 'TK' and row['monkey'] == 'yoh': #Exclude yoh when using TK (due to bad fit)
                pass
            else:
                axes[0].scatter(row['elo'], row['rho_g'], label=row['monkey'], color=colors[index],
                                edgecolor='grey', alpha=0.7, s=80)
                plot_scatter_regression(df_PT, 'elo', 'rho_g', axes[0])
    axes[0].set_xlabel('Elo Rating', fontsize=12)
    axes[0].set_ylabel('Rho Gain', fontsize=12)
    axes[0].grid(True)
    # Plotting Rho loss
    axes[1].set_title('Loss domain', fontsize=14, fontweight='bold')
    for index, row in df_PT.iterrows():
        if row['monkey'] in MONKEYS:
            if row['model'] == 'TK' and row['monkey'] == 'yoh': #Exclude yoh when using TK (due to bad fit)
                pass
            else:
                axes[1].scatter(row['elo'], row['rho_l'],  color=colors[index],
                                edgecolor='grey', alpha=0.7, s=80)
                plot_scatter_regression(df_PT, 'elo', 'rho_l', axes[1])
    axes[1].set_xlabel('Elo Rating', fontsize=13)
    axes[1].set_ylabel('Rho Loss', fontsize=13)
    axes[1].grid(True)
    # Create legend
    fig.legend(fontsize=12, ncol=8, bbox_to_anchor=(0.7, 0.2))
    # Adjust layout and display plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.3)  # Adjust the top margin for the title
    plt.show()


def build_table_static_all_dataset(lottery=0, save=False):
    """ Create a csv table with the mean elo score and the PT parameters per monkey,
        considering all the trials
    Args
        lottery: int 0 if all trials considered, 67 if only on risky trials
        """
    if lottery == 0:
         result_file = 'results-fits/monkey-analysis-L0'
    elif lottery == 67:
        result_file = 'results-fits/monkey-analysis-L67'

    # PT params
    with open(result_file + '-params.pkl', 'rb') as file:
        params = pickle.load(file)

    # Fit score
    with open(result_file + '-score.pkl', 'rb') as file:
        scores = pickle.load(file)

    date_limits = pd.read_csv('./data/dates_limits.csv')
    all_elo_file = './data/elo-scores.xlsx'
    all_elo = pd.read_excel(all_elo_file)
    all_elo.rename(columns={'oli': 'oll'}, inplace=True)

    all_ranks = all_elo.drop(columns=['Date'])
    all_ranks = all_ranks.rank(axis=1, method='min', ascending=False).astype('Int64')
    all_ranks['Date'] = all_elo['Date']

    player_indexes = {'TK': 1, 'P1': 2, 'P2': 3, 'P1+': 6, 'TK+': 5}
    mean_elo = {}
    for index, row in date_limits.iterrows():
        monkey = row['subject_id']
        mean_elo[monkey] = {}
        min_date, max_date = row['min'], row['max']
        mean_elo[monkey]['min_date'] = min_date
        mean_elo[monkey]['max_date'] = max_date
        mean_elo[monkey]['elo'] = np.nanmean(
            all_elo[(all_elo['Date'] >= min_date) & (all_elo['Date'] <= max_date)][monkey])
        # print(all_ranks[(all_ranks['Date'] >= min_date) & (all_ranks['Date'] <= max_date)][monkey])
        mean_elo[monkey]['rank'] = np.nanmean(
            all_ranks[(all_ranks['Date'] >= min_date) & (all_ranks['Date'] <= max_date)][monkey].astype(float))

    for shortname in player_indexes.keys():
        player = params['ALA'][player_indexes[shortname]]
        columns = np.concatenate((['monkey', 'elo', 'rank', 'date_start', 'date_end',
                                   'fit_score', 'model', 'lottery'], list(player.parameters.keys())))
        df_PT = pd.DataFrame(columns=columns)
        for monkey in mean_elo.keys():
            if (monkey == 'yoh' and shortname == 'TK') or (monkey == 'yoh' and shortname == 'P2'):
                pass
            else:
                score = scores[monkey.upper()][player_indexes[shortname]]
                if not math.isnan(mean_elo[monkey]['elo']):
                    new_row = {'monkey': monkey, 'elo': mean_elo[monkey]['elo'], 'rank': mean_elo[monkey]['rank'],
                               'date_start': mean_elo[monkey]['min_date'],
                               'date_end': mean_elo[monkey]['max_date'], 'fit_score': score, 'model': shortname,
                               'lottery': lottery}
                    player = params[monkey.upper()][player_indexes[shortname]]
                    for param_name in list(player.parameters.keys()):
                        new_row[param_name] = player.parameters[param_name]
                    df_PT = pd.concat([df_PT, pd.DataFrame([new_row])], ignore_index=True)

        #plot_linear_regression(df_PT)
        if save:
            df_PT.to_csv(f'results-fits/elo-score_vs_PT/elo_fit_L0_{player.shortname}.csv')


def built_table_static_1500_best(save=False):
    from monkey import Monkey
    elo = pd.read_csv('./data/elo_periods.csv')
    elo_1500_best = elo[['mkname', 'elo_best', 'start', 'end']]
    #for cls in [ProspectPlayerP1, ProspectPlayerTK, DualProspectPlayerP1, DualProspectPlayerTK]:
    for cls in [DualProspectPlayerTK]:
        columns = np.concatenate((['monkey', 'elo',  'date_start', 'date_end', 'fit_score', 'model', 'lottery'],
                                  list(cls.parameters.keys())))
        # Create empty dataframe that stores all PT parameters per monkey
        df_PT = pd.DataFrame(columns=columns)
        # Per monkey, compute PT parameters for each period and associate the elo
        for index, row in elo_1500_best.iterrows():
            monkey_name = row['mkname']
            monkey = Monkey(monkey_name, date_range=[row['start'], row['end']])
            trials, responses = monkey.get_data(lottery=0)
            player = cls.fit(trials, responses)
            score = evaluate_player_2(player, trials, responses, 1000)
            new_row = {'monkey': monkey_name, 'elo': row['elo_best'],
                       'date_start': row['start'], 'date_end': row['end'],
                       'fit_score': score, 'model': player.shortname, 'lottery': 0}
            for param_name in list(player.parameters.keys()):
                new_row[param_name] = player.parameters[param_name]
            df_PT = pd.concat([df_PT, pd.DataFrame([new_row])], ignore_index=True)

        plot_linear_regression(df_PT)
        if save:
            df_PT.to_csv(f'elo_1500best_L0_{player.shortname}.csv')


def built_table_dynamic_per_period(save=False):
    from monkey import Monkey
    elo = pd.read_csv('./data/elo_periods.csv')
    for cls in [ProspectPlayerP1, ProspectPlayerTK, DualProspectPlayerP1, DualProspectPlayerTK]:
        columns = np.concatenate((['monkey', 'elo',  'date_start', 'date_end', 'fit_score', 'model', 'lottery'],
                                  list(cls.parameters.keys())))
        # Create empty dataframe that stores all PT parameters per monkey
        df = pd.DataFrame(columns=columns)
        # Iterate through rows
        for index, row in elo.iterrows():
            monkey_name = row['mkname']
            data_list = []
            for i in range(1, 6):
                if not pd.isna(row[f'start_{str(i)}']) and not pd.isna(row[f'end_{str(i)}']):
                    monkey = Monkey(monkey_name, date_range=[row[f'start_{str(i)}'], row[f'end_{str(i)}']])
                    trials, responses = monkey.get_data(lottery=0)
                    player = cls.fit(trials, responses)
                    score = evaluate_player_2(player, trials, responses, 1000)
                    new_row = {'monkey': monkey_name, 'elo': row[f'eloscore{str(i)}'],
                               'date_start': row[f'start_{str(i)}'], 'date_end': row[f'end_{str(i)}'],
                               'fit_score': score, 'model': player.shortname, 'lottery': 0}
                    for param_name in list(player.parameters.keys()):
                        new_row[param_name] = player.parameters[param_name]
                    data_list.append(new_row)
            for row in data_list:
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df = df.dropna(subset=['elo'])
        print(df)
        #plot_linear_regression(df)
        if save:
            df.to_csv(f'elo_dynamic_L0_{player.shortname}.csv')


def RT_analysis_1500_best_static():
    from monkey import data
    data = data.dropna(subset=['response'])
    elo = pd.read_csv('./data/elo_periods.csv')
    elo_1500_best = elo[['mkname', 'elo_best', 'start', 'end']]
    df_RT = pd.DataFrame(columns=['monkey', 'elo', 'RT', 'date_start', 'date_end'])
    for index, row in elo_1500_best.iterrows():
        monkey = row['mkname']
        RT_mean = data[(data['subject_id'] == monkey) & (data['date'] >= row['start'])
                       & (data['date'] <= row['end'])]['RT'].mean()
        new_row = {'monkey': monkey, 'elo': row['elo_best'], 'RT': RT_mean, 'date_start': row['start'],
                   'date_end': row['end']}
        df_RT = df_RT.append(new_row, ignore_index=True)
    print(df_RT)
    df_RT.to_csv('./data/reaction_times/static_RT_elo.csv')
    fig, ax = plt.subplots(figsize=(9,7))
    ax.set_title('Mean Reaction Time vs Elo-score during the 1500 best trials', fontsize=14, fontweight='bold')
    for index, row in df_RT.iterrows():
        ax.scatter(row['elo'], row['RT'], label=row['monkey'], color=monkey_colors[row['monkey']],
                   edgecolor='grey', alpha=0.7, s=80)
    plot_scatter_regression(df_RT, 'elo', 'RT', ax)
    fig.legend(fontsize=12, ncol=8, bbox_to_anchor=(1, 0.2))
    plt.xlabel('Elo-score')
    plt.ylabel('Reaction time')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.show()


def RT_analysis_all_dataset():
    from monkey import data
    data = data.dropna(subset=['response'])
    date_limits = pd.read_csv('./data/dates_limits.csv')
    all_elo = pd.read_excel('./data/elo-scores.xlsx')
    all_elo.rename(columns={'oli': 'oll'}, inplace=True)

    df_RT = pd.DataFrame(columns=['monkey', 'elo', 'RT', 'date_start', 'date_end'])

    for index, row in date_limits.iterrows():
        monkey = row['subject_id']
        elo_mean = np.nanmean(all_elo[(all_elo['Date'] >= row['min']) & (all_elo['Date'] <= row['max'])][monkey])
        RT_mean = data[(data['subject_id'] == monkey) & (data['date'] >= row['min'])
                       & (data['date'] <= row['max'])]['RT'].mean()
        new_row = {'monkey': monkey, 'elo': elo_mean, 'RT': RT_mean, 'date_start': row['min'],
                   'date_end': row['max']}

        df_RT = df_RT.append(new_row, ignore_index=True)
    df_RT = df_RT.dropna(subset=['elo'])
    print(df_RT)
    df_RT.to_csv('./data/reaction_times/static_all_data_RT_elo.csv')
    fig, ax = plt.subplots(figsize=(9,7))
    ax.set_title('Mean Reaction Time vs Elo-score over all trials', fontsize=14, fontweight='bold')
    for index, row in df_RT.iterrows():
        ax.scatter(row['elo'], row['RT'], label=row['monkey'], color=monkey_colors[row['monkey']],
                   edgecolor='grey', alpha=0.7, s=80)
    plot_scatter_regression(df_RT, 'elo', 'RT', ax)
    fig.legend(fontsize=12, ncol=8, bbox_to_anchor=(1, 0.2))
    plt.xlabel('Elo-score')
    plt.ylabel('Reaction time')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.show()


def RT_analysis_dynamic():
    from monkey import data
    data = data.dropna(subset=['response'])
    elo = pd.read_csv('./data/elo_periods.csv')
    df_RT = pd.DataFrame(columns=['monkey', 'elo', 'RT', 'date_start','date_end' ])
    for index, row in elo.iterrows():
        monkey = row['mkname']
        for i in range(1, 6):
            if not pd.isna(row[f'start_{str(i)}']) and not pd.isna(row[f'end_{str(i)}']):
                RT_mean = data[(data['subject_id'] == monkey) & (data['date'] >= row[f'start_{str(i)}'])
                               & (data['date'] <= row[f'end_{str(i)}'])]['RT'].mean()
                new_row = {'monkey': monkey, 'elo': row[f'eloscore{str(i)}'], 'RT': RT_mean,
                           'date_start': row[f'start_{str(i)}'], 'date_end': row[f'end_{str(i)}']}
                df_RT = df_RT.append(new_row, ignore_index=True)
    print(df_RT)
    df_RT.to_csv('./data/reaction_times/dynamic_RT_elo.csv')
    handled_labels = set()
    fig, ax = plt.subplots(figsize=(9,7))
    ax.set_title('Mean Reaction Time vs Elo-score in selected periods', fontsize=14, fontweight='bold')
    for index, row in df_RT.iterrows():
        ax.scatter(row['elo'], row['RT'], color=monkey_colors[row['monkey']],
                   edgecolor='grey', alpha=0.7,
                   s=80)
        if row['monkey'] not in handled_labels:
            ax.scatter([], [], label=row['monkey'], color=monkey_colors[row['monkey']])  # Empty scatter just for legend
            handled_labels.add(row['monkey'])
    plot_scatter_regression(df_RT, 'elo', 'RT', ax)
    fig.legend(fontsize=12, ncol=6, bbox_to_anchor=(0.5, 0.01), loc='lower center')
    plt.xlabel('Elo-score')
    plt.ylabel('Reaction time')
    plt.ylim(600, 1600)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.show()


def plot_elo(elo_rates, monkey, range_dates, data, smoothed=False):
    """
    Plots Elo ratings over time with shaded regions for different date ranges and adds text to indicate
    the number of trials per period.

    Parameters:
        elo_rates (pd.DataFrame): DataFrame containing Elo ratings with a 'Date' column.
        monkey (str): Identifier for the monkey.
        range_dates (list of tuples): List of (start_date, end_date, color, label) tuples representing date ranges.
        data (pd.DataFrame): DataFrame containing trials with a 'subject_id' and 'date' columns.
        smoothed (bool): Whether to smooth the Elo ratings using Savitzky-Golay filter.
    """
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    # Filter data for the specified monkey
    data_monkey = data[data['subject_id'] == monkey]
    # Convert 'date' column to datetime
    data_monkey['date'] = pd.to_datetime(data_monkey['date'])
    # Group trials by date and count the number of trials per date
    trials_per_date = data_monkey.groupby(data_monkey['date'].dt.date).size()
    # Set the index of trials_per_date to a datetime index
    trials_per_date.index = pd.to_datetime(trials_per_date.index)
    # Determine the date range for Elo ratings
    dates = pd.to_datetime(elo_rates['Date'])
    mask = np.isfinite(elo_rates[monkey])

    # Plot Elo ratings over time
    fig, ax1 = plt.subplots(figsize=(12, 8))
    fig.suptitle(f'Elo Ratings over time for monkey {monkey}', fontsize=20,
                 fontweight='bold')

    if smoothed:
        ysmooth = savgol_filter(elo_rates[monkey][mask], 31, 3)
        ax1.plot(dates[mask], ysmooth, ls='-', lw=2, color='C0')
    else:
        ax1.plot(dates[mask], elo_rates[monkey][mask], ls='-', lw=2, color='C0')

    ax1.set_xlim(np.min(dates[mask]), np.max(dates[mask]))

    # Loop through each date range and plot shaded regions with text annotations for the number of trials
    for i, (start_date, end_date) in enumerate(range_dates):
        # Convert start_date and end_date to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Calculate mean Elo ratings for the current range
        elo_mean = elo_rates.loc[(elo_rates['Date'] > start_date) & (elo_rates['Date'] < end_date), monkey].mean()
        # Plot shaded region for the date range
        ax1.axvspan(start_date, end_date, alpha=0.3, color=colors[i], label=f'elo: {int(elo_mean)}')

        # Calculate the number of trials within the date range
        trials_in_range = trials_per_date.loc[
            (trials_per_date.index >= start_date) & (trials_per_date.index <= end_date)].sum()

        # Add text annotation for the number of trials in the center of the shaded region
        mid_date = start_date + (end_date - start_date) / 2
        ax1.text(mid_date, ax1.get_ylim()[0]+1.5 , f'{int(trials_in_range)} trials', ha='center', fontsize=12,
                 color=colors[i])

    # Set labels and legend for the Elo ratings plot
    ax1.legend(loc='upper left', fontsize=12)
    ax1.set_xlabel('Date', fontsize=16)
    ax1.set_ylabel('Elo Rating', fontsize=16)#, color='blue')
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)

    fig.tight_layout()
    plt.show()


def plot_elo_periods(monkey):
    from monkey import data
    all_elo = pd.read_excel('./data/elo-scores.xlsx')
    elo_periods = pd.read_csv('./data/elo_periods.csv')

    all_elo = all_elo[(all_elo["Date"] >= datetime.datetime(2020, 2, 22))]
    date_ranges = {}
    for index, row in elo_periods.iterrows():
        date_ranges[row['mkname']] = []
        for i in range(1, 6):
            if not pd.isna(row[f'start_{str(i)}']) and not pd.isna(row[f'end_{str(i)}']):
                date_ranges[row['mkname']].append((row[f'start_{str(i)}'], row[f'end_{str(i)}']))

    print(color.BOLD + 'Monkey ' + monkey.upper() + color.END)
    plot_elo(all_elo, monkey, date_ranges[monkey],  data, smoothed=True)


if __name__ == "__main__":
    ##### 1) Plot elo-score evolution of each monkey over time

    plot_all_elo_rates(smoothed=True)

    ##### 2) Build tables allowing to store PT parameters with the corresponding elo-score
    #built_table_static_1500_best()
    #build_table_static_all_dataset(lottery=0)
    #build_table_static_all_dataset(lottery=67)
    #built_table_dynamic_per_period()
    #plot_elo_periods(monkey='ala')

    #3) RT analysis
    #RT_analysis_1500_best_static()
    #RT_analysis_dynamic()
    #RT_analysis_all_dataset()






