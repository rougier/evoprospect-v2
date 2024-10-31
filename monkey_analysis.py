# Copyright 2024 (c) aomi Chaix-Echel & Nicolas P Rougier
# Released under a BSD 2-clauses license
import matplotlib.pyplot as plt

from player import *
from monkey import monkeys
from prettytable import PrettyTable, SINGLE_BORDER
import pickle
np.random.seed(123)

def fit_PT_players(lottery=0, players=[SigmoidPlayer, ProspectPlayerTK, ProspectPlayerP1, ProspectPlayerP2,
                                       ProspectPlayerGE, DualProspectPlayerTK, DualProspectPlayerP1,
                                       DualProspectPlayerP2, DualProspectPlayerGE], save=False, filename_to_save="results-fits/monkey-analysis"):
    """
    This function trains the PT players on the monkeys' data given the lottery type.
    Args:
        lottery: int or list of int,  lottery type containing the trials. If lottery=0, the models are fitted with all trials.
        player: list of players to  be fitted
        save: bool,  True if needed to be saved in pickle format (.pkl)
    """
    fits = {}
    results = {}
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

        # Fit players
        players_fit = []
        for cls in players:
            player = cls.fit(trials, responses)
            players_fit.append(player)
        R = [evaluate_player_2(p, trials, responses) for p in players_fit] #test on same trials
        results[monkey.shortname] = R
        fits[monkey.shortname] = players_fit
    if save:
        filename_to_save += '-L'
        if isinstance(lottery, list):
            for lot in lottery:
                filename_to_save += str(lot)
        else:
            filename_to_save += str(lottery)

        with open(filename_to_save + "-params.pkl", 'wb') as fp:
            pickle.dump(fits, fp)

        with open(filename_to_save + "-score.pkl", 'wb') as fp:
            pickle.dump(results, fp)

def test_PT_fitted_PT_players(result_file='results-fits/monkey-analysis-L0-params.pkl', testing_lottery=7,
                              save=False):
    """ This function test the fitted PT players on the given testing lotteries
        and outputs their scores in a table
    Args:
        result_file: str , path towards the pickle file containing the pickle results.
        testing_lottery: int or list of int,  lottery type containing the trials to be tested.
                        If lottery=0, the models are tested with all trials.
        save: bool,  True if needed to be saved in pickle format (.pkl)
        """
    with open(result_file, 'rb') as file:
        params = pickle.load(file)
    results = {}
    x = PrettyTable(border=True, align="l")
    x.set_style(SINGLE_BORDER)
    x.field_names = (["Players"] +
                     [p.shortname for p in params['ALA']])
    for monkey in monkeys:
        # Select the right monkey trials given the lottery
        if isinstance(testing_lottery, list):
            trials, responses = [], []
            for lot in testing_lottery:
                t, r = monkey.get_data(lottery=lot)
                trials.append(t)
                responses.append(r)
            trials = np.concatenate(trials)
            responses = np.concatenate(responses)
        else:
            trials, responses = monkey.get_data(lottery=testing_lottery)

        # Evaluate the score of each model
        R = [evaluate_player_2(p, trials, responses) for p in params[monkey.shortname]]

        results[monkey.shortname] = R
        # fits[monkey.shortname] = player
        # R.append(R[1]/R[2])
        Rmin, Rmax = np.min(R), np.max(R)
        for i in range(len(R)):
            if abs(R[i] - Rmax) < 1e-5:
                R[i] = "%.3f" % R[i]
            elif abs(R[i] - Rmin) < 1e-3:
                R[i] = "%.3f" % R[i]
            else:
                R[i] = "%.3f" % R[i]
        x.add_row([monkey.shortname] + R)
        print(x)
    print(results)
    if save:
        filename = f"monkey-analysis-L{result_file[-12]}train-L"
        if isinstance(testing_lottery, list):
            for lot in testing_lottery:
                filename += str(lot)
        else:
            filename += str(testing_lottery)

        with open(filename + "test-score.pkl", 'wb') as fp:
            pickle.dump(results, fp)


if __name__ == "__main__":
    # 1) Fit the players with the
    # lottery = 0    #it can be a list : [6,7]
    # players = [SigmoidPlayer, ProspectPlayerTK, ProspectPlayerP1, ProspectPlayerP2, ProspectPlayerGE,
    #           DualProspectPlayerTK, DualProspectPlayerP1, DualProspectPlayerP2, DualProspectPlayerGE]
    # fit_PT_players(lottery=lottery,players=players, save=False)

    # 2) Test the fitted players
    #result_file = 'results_pkl/monkey-analysis-L0-params.pkl'
    #testing_lottery = 0   #it can be a list : [6,7]
    #test_PT_fitted_PT_players(result_file=result_file, testing_lottery=testing_lottery,
    #                          save=False)

    pass








