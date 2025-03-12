# Social hierarchy influences monkeys' risky decisions
# Copyright 2024 (c) Naomi Chaix-Echel & Nicolas P Rougier
# Released under a BSD 2-clauses license
"""
This script allows to visualize all the decision-making models
 based on subjective utility and probability weighting functions.

Key components:
- **Subjectivre utility Functions**: Implements a subjective utility function that transforms objective values
                        into perceived values using parameters for gain, loss, and risk aversion.
- **Probability Weighting Functions**: Defines several probability transformation models,
                                    including the Prelec (P1, P2), Tversky-Kahneman (TK), and generalized (GE) functions,
                                     to model how probabilities are perceived differently from their objective values.
"""

from player import *
import matplotlib.pyplot as plt
import pickle

COLORS = {'gain': 'C0', 'loss': 'C1'}

def sigmoid(X, x0=0.0, mu=1.0):
    return 1 / (1 + np.exp(-mu * (X - x0)))

def subjective_utility(V, rho_g, rho_l,lambda_):
    return np.where(V > 0,
                    np.power(np.abs(V), rho_g),
                    -lambda_ * np.power(np.abs(V), rho_l))

def P1(P, alpha):
    return np.exp(- np.power((-np.log(P)), alpha))

def DualP1(P, alpha_g=None, alpha_l=None):
    if alpha_g is not None:
        return np.exp(- np.power((-np.log(P)), alpha_g))
    else:
         return np.exp(- np.power((-np.log(P)), alpha_l))

def P2(P, alpha, delta):
    return np.exp(-delta*np.power((-np.log(P)), alpha))

def DualP2(P, alpha_g=None, delta_g = None, alpha_l=None, delta_l = None):
    if alpha_g is not None and delta_g is not None:
        return np.exp(-delta_g*np.power((-np.log(P)), alpha_g))
    elif alpha_l is not None and delta_l is not None:
        return np.exp(-delta_l*np.power((-np.log(P)), alpha_l))

def DualTK(P, alpha_g=None, alpha_l=None):
    if alpha_g is not None:
        return np.power(P, alpha_g) /np.power((np.power(P, alpha_g)
                               + np.power(1 - P, alpha_g)), 1 / alpha_g)
    else:
        return np.power(P, alpha_l) / np.power((np.power(P, alpha_l)
                               + np.power(1 - P, alpha_l)), 1 /alpha_l)

def TK(P, alpha):
    return (np.power(P, alpha) /
            np.power((np.power(P, alpha)
                      + np.power(1-P, alpha)), 1/alpha))

def GE(P, delta, gamma):
    return (delta*np.power(P,gamma) /
       (delta *np.power(P,gamma) + np.power(1-P, gamma)))

def DualGE(P, delta_g=None, gamma_g=None, delta_l=None, gamma_l=None):
    if gamma_g is not None and delta_g is not None:
        return (delta_g*np.power(P,gamma_g) /
           (delta_g *np.power(P,gamma_g) + np.power(1-P, gamma_g)))
    elif gamma_l is not None and delta_l is not None:
        return (delta_l * np.power(P, gamma_l) /
                (delta_l * np.power(P, gamma_l) + np.power(1 - P, gamma_l)))

def range_subjective_utility():
    lambda_x = (0.2, 10)
    rho_g_x = (0.1, 2)
    rho_l_x = (0.1, 2)
    # Define the range for V
    V = np.linspace(-1, 1, 100)

    # Initialize lists to store the extreme values
    max_values = np.full_like(V, -np.inf)
    min_values = np.full_like(V, np.inf)

    # Plot the extreme functions and fill in between
    for lambda_ in lambda_x:
        for rho_g in rho_g_x:
            for rho_l in rho_g_x:
                # Calculate the subjective utility for the given parameters
                utility = subjective_utility(V, rho_g, rho_l, lambda_)

                # Update the max and min values
                max_values = np.maximum(max_values, utility)
                min_values = np.minimum(min_values, utility)

                # Plot the current curve with a label
                label = f"λ={lambda_}, $ρ_g={rho_g}$, $ρ_l={rho_l}$"
                plt.plot(V, utility, alpha=0.5, label=label)

    # Fill the area between the max and min values
    plt.fill_between(V, min_values, max_values, color='gray', alpha=0.3)
    plt.plot(V, V, color='black', linestyle='dotted', label='V=V')

    # Plot configuration
    plt.xlabel('V')
    plt.ylabel('Subjective Utility')
    plt.ylim(-2, 2)
    plt.title('Range of Subjective Utility Functions')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small') # Place legend outside the plot
    plt.tight_layout()
    plt.show()

def range_P1():
    alpha_x = (0.1, 2)
    # Define the range for V
    P = np.linspace(0, 1, 100)

    # Initialize lists to store the extreme values
    max_values = np.full_like(P, -np.inf)
    min_values = np.full_like(P, np.inf)

    # Plot the extreme functions and fill in between
    for alpha in alpha_x:
        # Calculate the subjective utility for the given parameters
        prob = P1(P, alpha)
        # Update the max and min values
        max_values = np.maximum(max_values, prob)
        min_values = np.minimum(min_values, prob)

        # Plot the current curve with a label
        label = f"$alpha$={alpha}"
        plt.plot(P, prob, alpha=0.5, label=label)

    # Fill the area between the max and min values
    plt.fill_between(P, min_values, max_values, color='gray', alpha=0.3)
    plt.plot(P, P, color='black', linestyle='dotted', label='V=V')

    # Plot configuration
    plt.xlabel('P')
    plt.ylabel('P1')
    plt.ylim(-0.1, 1.2)
    plt.title('Range of P1')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')  # Place legend outside the plot
    plt.tight_layout()
    plt.show()

def range_P2():
    alpha_x = (0.1, 2)
    delta_x = (0.1, 2)
    # Define the range for V
    P = np.linspace(0, 1, 100)

    # Initialize lists to store the extreme values
    max_values = np.full_like(P, -np.inf)
    min_values = np.full_like(P, np.inf)

    # Plot the extreme functions and fill in between
    for alpha in alpha_x:
        for delta in delta_x:
            # Calculate the subjective utility for the given parameters
            prob = P2(P, alpha, delta)
            # Update the max and min values
            max_values = np.maximum(max_values, prob)
            min_values = np.minimum(min_values, prob)

            # Plot the current curve with a label
            label = f"$alpha$={alpha}, delta={delta}"
            plt.plot(P, prob, alpha=0.5, label=label)

    # Fill the area between the max and min values
    plt.fill_between(P, min_values, max_values, color='gray', alpha=0.3)
    plt.plot(P, P, color='black', linestyle='dotted', label='V=V')

    # Plot configuration
    plt.xlabel('P')
    plt.ylabel('P2')
    plt.ylim(-0.1, 1.2)
    plt.title('Range of P2')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')  # Place legend outside the plot
    plt.tight_layout()
    plt.show()

def range_TK():
    alpha_x = (0.2, 0.5, 1, 2)
    # Define the range for V
    P = np.linspace(0, 1, 100)
    # Initialize lists to store the extreme values
    max_values = np.full_like(P, -np.inf)
    min_values = np.full_like(P, np.inf)

    # Plot the extreme functions and fill in between
    for alpha in alpha_x:
        # Calculate the subjective utility for the given parameters
        prob = TK(P, alpha)
        # Update the max and min values
        max_values = np.maximum(max_values, prob)
        min_values = np.minimum(min_values, prob)

        # Plot the current curve with a label
        label = f"$alpha$={alpha}"
        plt.plot(P, prob, alpha=0.5, label=label)

    # Fill the area between the max and min values
    plt.fill_between(P, min_values, max_values, color='gray', alpha=0.3)
    plt.plot(P, P, color='black', linestyle='dotted', label='V=V')

    # Plot configuration
    plt.xlabel('P')
    plt.ylabel('TK')
    plt.ylim(-0.1, 1.2)
    plt.title('Range of TK')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')  # Place legend outside the plot
    plt.tight_layout()
    plt.show()

def range_GE():
    gamma_x = (0.1, 2)
    delta_x = (0.1, 2)

    # Define the range for V
    P = np.linspace(0, 1, 100)

    # Initialize lists to store the extreme values
    max_values = np.full_like(P, -np.inf)
    min_values = np.full_like(P, np.inf)

    # Plot the extreme functions and fill in between
    for gamma in gamma_x:
        for delta in delta_x:
            # Calculate the subjective utility for the given parameters
            prob = P2(P, delta, gamma)
            # Update the max and min values
            max_values = np.maximum(max_values, prob)
            min_values = np.minimum(min_values, prob)

            # Plot the current curve with a label
            label = f"gamma={gamma}, delta={delta}"
            plt.plot(P, prob, alpha=0.5, label=label)

    # Fill the area between the max and min values
    plt.fill_between(P, min_values, max_values, color='gray', alpha=0.3)
    plt.plot(P, P, color='black', linestyle='dotted', label='V=V')

    # Plot configuration
    plt.xlabel('P')
    plt.ylabel('GE')
    plt.ylim(-0.1, 1.2)
    plt.title('Range of GE')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')  # Place legend outside the plot
    plt.tight_layout()
    plt.show()

def plot(ax, f, fits, mean_fit, xlimits, expected_params, color_gain='C0', color_loss='C1'):
    xmin, xmax = xlimits
    X = np.linspace(xmin, xmax, 500)

    # Plot each fit
    for p in fits.values():
        # Filter the parameters to pass only the expected ones
        filtered_params = {k: p[k] for k in expected_params if k in p}
        #ax.plot(X, f(X, **filtered_params), color=color, lw=0.5, alpha=0.25)
        ax.plot(X[X < 0], f(X[X < 0], **filtered_params), color=color_loss, lw=0.5, alpha=0.25)
        ax.plot(X[X >= 0], f(X[X >= 0], **filtered_params), color=color_gain, lw=0.5, alpha=0.25)

    filtered_mean_fit = {k: p[k] for k in expected_params if k in mean_fit}
    # Plot mean fit
    #ax.plot(X, f(X, **filtered_mean_fit), color="C0", lw=1.25)
    # Adjust plot aesthetics
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return X

def compute_mean_fit(fits_player):
    # Mean parameters
    mean_fit = {}
    params = list(fits_player.values())[0]
    for pname in params.keys():
        mean_fit[pname] = np.mean([fits_player[sid][pname] for sid in fits_player.keys()])
    return mean_fit

def plot_sigmoid(ax, fits, mean_fit, color='black'):
    thickness = 1.8
    X = plot(ax, sigmoid, fits, mean_fit,  xlimits=(-2, 2), expected_params=['x0', 'mu'], color_loss=color,
             color_gain=color)
    ax.plot(X, sigmoid(X, x0=mean_fit['x0'], mu=mean_fit['mu']), color=color, lw=thickness)
    ax.axvline(0.0, color="black", ls="--", lw=.75)
    ax.axhline(0.5, color="black", ls="--", lw=.75)
    ax.set_title("sigmoid(Δx)")
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([0, 0.5, 1])
    ax.text(1, 0.025, r"$\mu = %.2f, x_0=%.2f$" % (mean_fit["mu"], mean_fit["x0"]),
            ha="right", alpha=.5, transform=ax.transAxes)
    plt.tight_layout()

def plot_utility(ax, fits, mean_fit):
    thickness = 1.8
    X = plot(ax, subjective_utility, fits, mean_fit, xlimits=(-1, 1), expected_params=['rho_g', 'rho_l', 'lambda_'],
             color_gain= COLORS['gain'], color_loss=COLORS['loss'])
    ax.plot(X, X, color="black", lw=0.5, ls="--")
    ax.plot(X[X < 0], subjective_utility(X[X < 0], rho_g=mean_fit['rho_g'], rho_l=mean_fit['rho_l'], lambda_=mean_fit['lambda_']),
            color=COLORS['loss'], lw=thickness)

    ax.plot(X[X >=0], subjective_utility(X[X >= 0], rho_g=mean_fit['rho_g'], rho_l=mean_fit['rho_l'], lambda_=mean_fit['lambda_']),
            color=COLORS['gain'], lw=thickness)

    ax.axvline(0, color="black", ls="--", lw=.75)
    ax.axhline(0, color="black", ls="--", lw=.75)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-2, -1, 0, 1])
    ax.set_ylim(-2, 1.2)
    ax.set_title("u(x)")
    ax.text(1, 0.025, r"$\lambda = %.2f, \rho_g=%.2f, \rho_l=%.2f$" % (mean_fit["lambda_"], mean_fit["rho_g"],
                                                                       mean_fit["rho_l"]),
            ha="right", alpha=.5, transform=ax.transAxes)

def plot_probability(ax, fits, mean_fit, type='TK', domain='gain'):
    thickness = 1.8
    if type == 'TK':
        X = plot(ax, TK, fits, mean_fit, xlimits=(0.01, 1), expected_params=['alpha'])
        ax.text(1, 0.025, r"$\gamma = %.2f$" % mean_fit["alpha"],
                ha="right", alpha=.5, transform=ax.transAxes)
        ax.plot(X, TK(X, alpha=mean_fit['alpha']),
                color="C0", lw=1.25)
    elif type == 'TK+':
        if domain == 'gain':
            X = plot(ax, DualTK, fits, mean_fit, xlimits=(0.01, 1), expected_params=['alpha_g'],
                     color_gain= COLORS['gain'], color_loss=COLORS['loss'])
            ax.plot(X, DualTK(X, alpha_g=mean_fit['alpha_g']), color=COLORS['gain'], lw=thickness)
            ax.text(1, 0.025, r"$\alpha_g = %.2f$" % mean_fit['alpha_g'],
                    ha="right", alpha=.5, transform=ax.transAxes)
        else:
            X = plot(ax, DualTK, fits, mean_fit, xlimits=(0.01, 1), expected_params=['alpha_l']
                     ,color_loss= COLORS['loss'],color_gain= COLORS['loss'])
            ax.plot(X, DualTK(X, alpha_l=mean_fit['alpha_l']), color=COLORS['loss'], lw=thickness)
            ax.text(1, 0.025, r"$\alpha_l = %.2f$" % mean_fit['alpha_l'],
                    ha="right", alpha=.5, transform=ax.transAxes)

    elif type == 'P1':
        X = plot(ax, P1, fits, mean_fit, xlimits=(0.01, 1), expected_params=['alpha'])
        ax.plot(X, P1(X, alpha=mean_fit['alpha']),
                color="C0", lw=1.25)
        ax.text(1, 0.025, r"$\alpha = %.2f$" % mean_fit["alpha"],
                ha="right", alpha=.5, transform=ax.transAxes)
        x0 = np.exp(-1)
        ax.axvline(x0, color="black", ls="--", lw=.75)
        ax.axhline(x0, color="black", ls="--", lw=.75)
        ax.set_xticks([0, x0, 1])
        ax.set_xticklabels(["0", "1/e", 1])
        ax.set_yticklabels(["0", "1/e", 1])
    elif type == 'P1+':
        if domain == 'gain':
            param = 'alpha_g'
            X = plot(ax, DualP1, fits, mean_fit, xlimits=(0.01, 1), expected_params=[param])
            ax.plot(X, DualP1(X, alpha_g=mean_fit[param]), color=COLORS['gain'], lw=1.25)
            ax.text(1, 0.025, r"$\alpha_g = %.2f$" % mean_fit[param], ha="right", alpha=.5, transform=ax.transAxes)
        else:
            param = 'alpha_l'
            X = plot(ax, DualP1, fits, mean_fit, xlimits=(0.01, 1), expected_params=[param],
                     color_loss= COLORS['loss'],color_gain= COLORS['loss'])
            ax.plot(X, DualP1(X, alpha_l=mean_fit[param]), color=COLORS['loss'], lw=1.25)
            ax.text(1, 0.025, r"$\alpha_l = %.2f$" % mean_fit[param], ha="right", alpha=.5, transform=ax.transAxes)

        x0 = np.exp(-1)
        ax.axvline(x0, color="black", ls="--", lw=.75)
        ax.axhline(x0, color="black", ls="--", lw=.75)
        ax.set_xticks([0, x0, 1])
        ax.set_xticklabels(["0", "1/e", 1])
        ax.set_yticklabels(["0", "1/e", 1])
    elif type == 'P2':
        X = plot(ax, P2, fits, mean_fit, xlimits=(0.01, 1), expected_params=['alpha', 'delta'])
        ax.plot(X, P2(X, alpha=mean_fit['alpha'], delta=mean_fit['delta']),
                color="C0", lw=1.25)
        ax.text(1, 0.025, r"$\alpha = %.2f, \delta= %.2f$" % (mean_fit["alpha"], mean_fit["delta"]),
                ha="right", alpha=.5, transform=ax.transAxes)
    elif type == 'P2+':
        if domain == 'gain':
            X = plot(ax, DualP2, fits, mean_fit, xlimits=(0.01, 1), expected_params=['alpha_g', 'delta_g'])
            ax.plot(X, DualP2(X, alpha_g=mean_fit['alpha_g'],delta_g=mean_fit['delta_g']), color=COLORS['gain'], lw=1.25)
            #ax.text(1, 0.025, r"$\alpha_g = %.2f$" % mean_fit['alpha_g'],
             #       ha="right", alpha=.5, transform=ax.transAxes)
        else:
            X = plot(ax, DualP2, fits, mean_fit, xlimits=(0.01, 1), expected_params=['alpha_l', 'delta_l'],
                     color_loss= COLORS['loss'],color_gain= COLORS['loss'])
            ax.plot(X, DualP2(X, alpha_l=mean_fit['alpha_l'], delta_l=mean_fit['delta_l']), color=COLORS['loss'], lw=1.25)
            #ax.text(1, 0.025, r"$\alpha_l = %.2f$" % mean_fit['alpha_l'],
             #       ha="right", alpha=.5, transform=ax.transAxes)
    elif type == 'GE':
        X = plot(ax, GE, fits, mean_fit, xlimits=(0.01, 1), expected_params=['delta', 'gamma'])
        ax.plot(X, GE(X, gamma=mean_fit['gamma'], delta=mean_fit['delta']),
                color="C0", lw=1.25)
        ax.text(1, 0.025, r"$\alpha = %.2f, \gamma= %.2f$" % (mean_fit["delta"], mean_fit["gamma"]),
                ha="right", alpha=.5, transform=ax.transAxes)
    elif type == 'GE+':
        if domain == 'gain':
            X = plot(ax, DualGE, fits, mean_fit, xlimits=(0.01, 1), expected_params=['gamma_g', 'delta_g'])
            ax.plot(X, DualGE(X, gamma_g=mean_fit['gamma_g'], delta_g=mean_fit['delta_g']), color=COLORS['gain'], lw=1.25)
            #ax.text(1, 0.025, r"$\alpha_g = %.2f$" % mean_fit['alpha_g'],
             #       ha="right", alpha=.5, transform=ax.transAxes)
        else:
            X = plot(ax, DualGE, fits, mean_fit, xlimits=(0.01, 1), expected_params=['gamma_l', 'delta_l'],
                     color_loss= COLORS['loss'],color_gain= COLORS['loss'])
            ax.plot(X, DualGE(X, gamma_l=mean_fit['gamma_l'], delta_l=mean_fit['delta_l']), color=COLORS['loss'], lw=1.25)
            #ax.text(1, 0.025, r"$\alpha_l = %.2f$" % mean_fit['alpha_l'],
             #       ha="right", alpha=.5, transform=ax.transAxes)
    else:
        print('Function of probability not recognized')

    ax.plot(X, X, color="black", lw=0.5, ls="--")
    if domain == 'gain':
        ax.set_title("w(p) - GAIN")
    elif domain == 'loss':
        ax.set_title("w(p) - LOSS")


if __name__ == "__main__":
    with open('results-fits/monkey-analysis-L0-params.pkl', 'rb') as file:
    #with open('results-fits/distort03-3-monkey-analysis-L0-params.pkl', 'rb') as file:
        params = pickle.load(file)
    fits = {}
    players = list(params.values())[0] # Name of the players
    for player in players:
        fits[player.shortname] = {}
    for monkey in params.keys():
        # rejected monkeys
        if monkey not in ['ANU', 'NER', 'YIN', 'OLG', 'JEA', 'PAT', 'YOH']:
            for player_fit in params[monkey]:
                #if player_fit.shortname in ['P1', 'TK',  'P2',  'GE','P1+','TK+','P2+', 'GE+']:
                fits[player_fit.shortname][monkey] = player_fit.parameters
        else:
            print(monkey)
    for type in ['P1', 'TK',  'P2',  'GE']:
        if type in fits:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('{} fit'.format(type))
            plot_sigmoid(ax1, fits[type], compute_mean_fit(fits[type]))
            plot_utility(ax2, fits[type], compute_mean_fit(fits[type]))
            plot_probability(ax3, fits[type], compute_mean_fit(fits[type]), type=type, domain=None)
            plt.show()









