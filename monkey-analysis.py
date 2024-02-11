from player import *

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


def evaluate(subject_id, players, per_trial=True):
    """
    Evaluate players on all lotteries and reports accuracy per
    trial (default) or per_lottery
    """    

    s = "        "
    for pname in players.keys():
        s = s + "%-5s  " % pname
    print(s)
    print("----" + "-------"*len(players))
    
#    for name, L in lotteries.items():
    for task_id in range(8):
        trials, R0 = get_trials(data, subject_id, task_id)
        #trials = generate_trials(10_000, L)
        #R0 = player.play(trials)
        
        s = "L%d: " % task_id
        for p in players.values():
            R1 = p.play(trials)
            if per_trial:
                d = 1 - ((abs(R0-R1).sum())/len(R0))
            else:
                d = 1 - (abs(R0.sum()-R1.sum())/len(R0))            
            s = s + "   %.2f" % d
        print(s)
    print()


for subject_id in subject_ids:
    print()
    
    #subject_id = subject_ids[1]
    trials, responses = get_trials(data, subject_id)

    n = len(trials)
    bias = 0.5 - responses.sum() / len(trials)
    
    print(color.BOLD + color.RED +
          "Fitting %s (n=%d, bias=%.3f): " % (subject_id, n, bias),
          end="", flush=True)

    print(".", end="", flush=True)
    rd_fit = RandomPlayer.fit(trials, responses)
    print(".", end="", flush=True)
    sg_fit = SigmoidPlayer.fit(trials, responses)
    print(".", end="", flush=True)
    p1_fit = ProspectPlayerP1.fit(trials, responses)
    print(".", end="", flush=True)
    p2_fit = ProspectPlayerP2.fit(trials, responses)
    print(".", end="", flush=True)
    ge_fit = ProspectPlayerGE.fit(trials, responses)
    print(".", end="", flush=True)
    tk_fit = ProspectPlayerTK.fit(trials, responses)
        
    players = { " RD " : rd_fit,
                " SG " : sg_fit,
                " P1 " : p1_fit,
                " P2 " : p2_fit,
                " GE " : ge_fit,
                " TK " : tk_fit }

    print(" done!" + color.END)
    
    # We compare players parameters
    print(color.BOLD + "Comparison of fitted players" + color.END)
    show(players)
    print()

    # We evaluate players, including the original one
    print(color.BOLD + "Evaluation (per trials)" + color.END)
    evaluate(subject_id, players)

