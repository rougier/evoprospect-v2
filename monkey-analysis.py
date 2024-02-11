from player import *
from prettytable import PrettyTable, SINGLE_BORDER

import pandas as pd
data = pd.read_csv("./data/data-processed.csv")
subject_ids = data['subject_id'].unique()
task_ids = np.sort(data['task_id'].unique())

def get_trials(data, subject_id=None, task_id=None):    
    if isinstance(subject_id, str):
        subject_id = [subject_id]        
    if isinstance(task_id, int):
        task_id = [task_id]        
    if subject_id is not None and task_id is not None:
        df = data.loc[(data['task_id'].isin(task_id)) &
                      (data['subject_id'].isin(subject_id))]
    elif subject_id is not None:
        df = data.loc[(data['subject_id'].isin(subject_id))]
    elif task_id is not None:
        df = data.loc[(data['task_id'].isin(task_id))]
    else:
        df = data
    
    trials = np.zeros((len(df),4))
    trials[:,0] = df["V_left"]
    trials[:,1] = df["P_left"]
    trials[:,2] = df["V_right"]
    trials[:,3] = df["P_right"]
    responses = np.zeros(len(df))
    responses[...] = df["response"]
    return trials, responses


def evaluate(subject_id, players):

    x = PrettyTable(border=True, align="l")
    x.set_style(SINGLE_BORDER)
    x.field_names = ([bold("Lottery")] +
                     [bold(p.shortname) for p in players])
    for task_id in range(8):
        trials, R0 = get_trials(data, subject_id, task_id)
        name = bold("L%d" % task_id)
        R = [p.play(trials) for p in players]
        R = [1 - ((abs(R0 - R).sum())/len(R0)) for R in R]
        Rmin, Rmax = np.min(R), np.max(R)
        for i in range(len(R)):
            if abs(R[i] - Rmax) < 1e-5:
                R[i] = bold("%.3f" % R[i])
            elif abs(R[i] - Rmin) < 1e-3:
                R[i] = red("%.3f" % R[i])
            else:
                R[i] = "%.3f" % R[i]
        x.add_row([name] + R)
    print(x)

    

if __name__ == "__main__":

    for subject_id in subject_ids:
        print()

        #subject_id = subject_ids[1]
        trials, responses = get_trials(data, subject_id)

        n = len(trials)
        bias = 0.5 - responses.sum() / len(trials)

        print(color.BOLD + color.RED +
              "Fitting %s (n=%d, bias=%.3f): " % (subject_id.upper(), n, bias),
              end="", flush=True)

        players = []
        for cls in [RandomPlayer, SigmoidPlayer, ProspectPlayerP1,
                    ProspectPlayerP2, ProspectPlayerGE, ProspectPlayerTK]:
            print(".", end="", flush=True)
            players.append(cls.fit(trials, responses))

        print(" done!" + color.END)

        # We compare players parameters
        #print(color.BOLD + "Comparison of fitted players" + color.END)
        show(players)

        # We evaluate players, including the original one
        # print(color.BOLD + "Evaluation (per trials)" + color.END)
        evaluate(subject_id, players)

