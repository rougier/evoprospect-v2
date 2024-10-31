import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import stats
from player import *
from prettytable import PrettyTable, SINGLE_BORDER
import json
import matplotlib.pyplot as plt

# In the json result file, index associated to each model
idx_model = {'SG': 0, 'TK': 1, 'P1': 2, 'P2': 2, 'GE': 3, 'TK+': 1, 'P1+': 2, 'P2+': 2 , 'GE+': 3 }

# Set colors
dark2 = plt.cm.get_cmap('Dark2', 8)
tab20 = plt.cm.get_cmap('tab20', 20)
COLORS = np.concatenate([tab20(np.linspace(0, 1, 20)), dark2(np.linspace(0, 1, 8))], axis=0)


def compute_bias(save=False, lottery=0):
    from monkey import monkeys
    #Compute bias in all lotteries
    all_R = []
    all_monkeys = []
    all_ratio = []
    biases = {}
    for monkey in monkeys:
        if isinstance(lottery, list):
            trials, responses = [], []
            for lot in lottery:
                t, r = monkey.get_data(lottery=lot)
                trials.append(t)
                responses.append(r)
            responses = np.concatenate(responses)
        else:
            trials, responses = monkey.get_data(lottery=lottery)

        c0 = np.count_nonzero(responses == 0) / len(responses)
        c1 = np.count_nonzero(responses == 1) / len(responses)

        all_monkeys.append(monkey.shortname)
        all_R.append(abs(c0 - c1))
        biases[monkey.shortname] = abs(c0 - c1)

    x = PrettyTable(border=True, align="l")
    x.set_style(SINGLE_BORDER)
    x.field_names = (["Players"] + ['bias'])
    ordered_idx = np.argsort(all_R)
    for i in ordered_idx:
        x.add_row([all_monkeys[i]] + [str(round(all_R[i], 3))])
    print(x)
    if save:
        with open('monkey_biases.json', 'w') as json_file:
            json.dump(biases, json_file)


def plot_bias_vs_score(plot_ratio=False, ax=None, result_file='', biases_file='./data/monkey_biases.json', model='TK'):
    # Load biases
    with open(biases_file, 'r') as json_file:
        biases = json.load(json_file)

    # Load scores
    with open(result_file, 'rb') as file:
        score = pickle.load(file)

    all_score_SG = []
    all_ratio_SG_PT = []
    all_bias = []
    all_monkeys = []

    for mk in score:
        if mk not in ['GAN', 'OLA']:
            all_monkeys.append(mk[:4])
            all_score_SG.append(float(score[mk][idx_model['SG']]))
            all_ratio_SG_PT.append((float(score[mk][idx_model['SG']]) / float(score[mk][idx_model[model]])))
            all_bias.append(biases[mk])

    to_plot = all_ratio_SG_PT if plot_ratio else all_score_SG

    # Fill region where bias > 0.4
    ax.fill_betweenx(
        [0.94, 0.82],  # Arbitrary y limits to cover the plot area
        0.4,
        max(all_bias)+0.06,
        color='red',
        alpha=0.15,
        label='bias>0.4'
    )
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Plot data points
    for i in range(len(all_monkeys)):
        ax.scatter(all_bias[i], to_plot[i], label=None, color=COLORS[i])#, facecolor='none')

    # Performing linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(all_bias, to_plot)
    line = np.array(all_bias) * slope + intercept
    ax.plot(all_bias, line, color='red', label=f'r={r_value:.2f}, p_val={p_value:.3f}')
    ax.legend(loc='upper left')
    short_title = 'SG/P1' if plot_ratio else 'SG score'
    ax.set_xlabel('Monkey Bias')
    ax.set_ylabel(short_title)
    ax.set_title(short_title + ' score vs monkey bias')

    return ax, all_monkeys


def plot_bias_vs_score_diff(ax=None, result_file='', biases_file='./data/monkey_biases.json', model='TK'):
    # Load biases
    with open(biases_file, 'r') as json_file:
        biases = json.load(json_file)

    # Load scores
    with open(result_file, 'rb') as file:
        score = pickle.load(file)

    all_score_SG = []
    all_score_PT = []
    all_bias = []
    diff = []
    all_monkeys = []
    for mk in score:
        if mk not in ['GAN', 'OLA']:
            all_monkeys.append(mk)
            all_score_SG.append(float(score[mk][idx_model['SG']]))
            all_score_PT.append(float(score[mk][idx_model[model]]))
            diff.append(abs(float(score[mk][0])-float(score[mk][idx_model[model]])))
            all_bias.append(biases[mk])

    # Fill region where bias > 0.4
    fill_bet = ax.fill_betweenx(
        [0, 0.055],
        0.4,
        max(all_bias)+0.06,
        color='red',
        alpha=0.15,
        label='$\Delta(SG-TK) < 0.05$'
    )
    scatter_handles = []
    scatter_labels = []
    # Plot data points
    for i in range(len(all_monkeys)):
        scatter = ax.scatter(all_bias[i], diff[i], color=COLORS[i], marker='o', label=all_monkeys[i]) #edgecolor='black'
        scatter_handles.append(scatter)
        scatter_labels.append(all_monkeys[i])

    slope, intercept, r_value, p_value, std_err = stats.linregress(all_bias, diff)
    line = np.array(all_bias) * slope + intercept
    regression_line, = ax.plot(all_bias, line, color='red', label=f'r={r_value:.2f}, p_val={p_value:.3f}')

    reg_legend = ax.legend(handles=[regression_line, fill_bet], loc='upper right')
    ax.add_artist(reg_legend)
    ax.set_ylim(-0.05,0.15)
    ax.set_xlabel('Monkey Bias')
    ax.set_ylabel('$\Delta(SG-TK)$')
    ax.set_title('$\Delta(SG-TK)$ vs Monkey Bias')

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    print('Excluded monkeys: ')

    for i in range(len(all_monkeys)):
        epsilon = (all_score_PT[i] - all_score_SG[i])
        if epsilon <0.055 and all_bias[i] > 0.4:
            print(all_monkeys[i])

    return ax, all_monkeys, scatter_labels, scatter_handles


def plot_bias_analysis(result_file, biases_file='./data/monkey_biases.json', model='TK'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the first graph
    ax1, _ = plot_bias_vs_score(plot_ratio=False, ax=ax1, result_file=result_file,biases_file=biases_file, model= model)

    # Plot the second graph
    ax2, _, combined_labels, combined_handles = plot_bias_vs_score_diff(ax=ax2,result_file=result_file, model=model)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    fig.legend(handles=combined_handles, labels=combined_labels, bbox_to_anchor=(0.5, 0.14), loc='upper center',
               fontsize='large', frameon=False, ncol=11)
    plt.show()


def plot_score_difference(controle_fit_file = 'results-fits/monkey-analysis-L0train-L67test-score.pkl',
                    risky_fit_file = 'results-fits/monkey-analysis-L0train-L12345test-score.pkl'):
    # Load the first set of scores (risk)
    with open(controle_fit_file ,'rb') as file:
        score_risk = pickle.load(file)
    # score risk order: SG - TK - P1 - P2 - GE - TK+ - P1+ - P2+ - GE+
    all_score_SG_risk = []
    all_score_PT_risk = []
    all_monkeys_risk = []
    for mk in score_risk:
        if mk not in ['GAN', 'OLA', 'YOH']:
            all_monkeys_risk.append(mk)
            all_score_SG_risk.append(float(score_risk[mk][0])) # index 0: SG
            all_score_PT_risk.append(float(score_risk[mk][1])) # index 1: TK

    # Load the second set of scores (control)
    with open(risky_fit_file, 'rb') as file:
        score_control = pickle.load(file)

    all_score_SG_control = []
    all_score_PT_control = []
    all_monkeys_control = []
    for mk in score_control:
        if mk not in ['GAN', 'OLA', 'YOH']:
            all_monkeys_control.append(mk)
            all_score_SG_control.append(float(score_control[mk][0])) # index 0: SG
            all_score_PT_control.append(float(score_control[mk][1]))  # index 1: TK

    # Plotting
    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

    # First subplot (Risk)
    ax1.scatter(all_score_SG_risk, all_monkeys_risk, color='red', label='SG', marker='o', facecolor='none')
    ax1.scatter(all_score_PT_risk, all_monkeys_risk, color='blue', label='TK', marker='o')
    for i, monkey in enumerate(all_monkeys_risk):
        ax1.plot([all_score_SG_risk[i], all_score_PT_risk[i]], [monkey, monkey], 'k--')
    ax1.set_title('Risky lotteries (types 6 and 7)')
    ax1.set_xlabel('Score', fontsize=12)
    ax1.set_ylabel('Monkey', fontsize=12)
    ax1.set_xticks([0.75, 1])
    ax1.legend(loc='best')
    ax1.invert_yaxis()  # Invert the y-axis to have the same order in both plots
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='both', which='major', labelsize=12)

    # Second subplot (Control)
    ax2.scatter(all_score_SG_control, all_monkeys_control, color='red', label='SG', marker='o', facecolor='none')
    ax2.scatter(all_score_PT_control, all_monkeys_control, color='blue', label='TK', marker='o')
    for i, monkey in enumerate(all_monkeys_control):
        ax2.plot([all_score_SG_control[i], all_score_PT_control[i]], [monkey, monkey], 'k--')
    ax2.set_title('Control lotteries (types 1, 2, 3, 4, 5)')
    ax2.set_xlabel('Score', fontsize=12)
    ax2.legend(loc='best')
    ax2.set_xticks([0.75,1])
    ax2.invert_yaxis()  # Invert the y-axis to match the first subplot
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax2.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_bias_analysis(result_file='results-fits/monkey-analysis-L0train-L67test-score.pkl', biases_file='./data/monkey_biases.json', model='TK')
    pass




















