# Copyright 2024 (c) aomi Chaix-Echel & Nicolas P Rougier
# Released under a BSD 2-clauses license
import os.path

def analyze(data, monkey, player, every=2500, n=5000):

    df = data.loc[data["subject_id"] == monkey].copy()
    df.sort_values(by="date", inplace=True)

    T = np.array(df[["V_left","P_left","V_right","P_right"]])
    R = np.array(df["response"])
    D = np.array(df["date"], dtype=np.datetime64)
    RT = np.array(df["RT"])

    fits, RTs, X = [], [], []
    for i in range(0, len(T), every):
        trials   = T[i:i+n]
        responses = R[i:i+n]
        fits.append(player.fit(trials, responses))
        X.append(D[min(i+n, len(D)-1)])
        RTs.append(RT[i:i+n].mean())
        print(fits[-1])

    return fits, RTs, X

if __name__ == "__main__":
    import pickle
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from player import *

    player = DualProspectPlayerP1
    every = 1000
    n = 5000
    filename = "monkey-sliding-analysis-%d-%d.pkl" % (every, n)

    # Don't recompute if 0
    if not os.path.exists(filename):
        print(f"'{filename}' not found, computing results (slow).")
        results = {}
        data = pd.read_csv("./data/data-processed.csv")
        monkeys = data["subject_id"].unique()
        for monkey in monkeys:
            print(monkey)
            results[monkey] = analyze(data, monkey, player, every, n)
        with open(filename, 'wb') as fp:
            pickle.dump(results, fp)
    else:
        print(f"Loading previous results ({filename}).")
        with open(filename, 'rb') as fp:
            results = pickle.load(fp)

    keys = list(player.parameters.keys())
    monkeys = list(results.keys())

    for monkey in monkeys:
        fig, ax = plt.subplots(nrows = 1+len(keys), sharex=True, figsize=(14,10))
        for m in monkeys:
            fits, RTs, X = results[m]
            for i,k in enumerate(keys):
                Y = [fit.parameters[k] for fit in fits]
                ax[i].plot(X, Y, color="black", linewidth=0.5 ,alpha=0.25)
            ax[-1].plot(X, RTs, color="C1", linewidth=0.5, alpha=0.25)

        fits, RTs, X = results[monkey]
        for i,k in enumerate(keys):
            Y = [fit.parameters[k] for fit in fits]
            ax[i].plot(X, Y, color="C0", linewidth=1.5)
        ax[-1].plot(X, RTs, color="C1", linewidth=1.5)

        fig.suptitle(monkey)
        for i,k in enumerate(keys):
            ax[i].set_ylabel(k)
        ax[-1].set_ylabel("RT")
        plt.tight_layout()

        filename = f"results/{monkey}-sliding-analysis-{every}-{n}.pdf"
        print(f"Saving {filename}")
        fig.savefig(filename)
        plt.close()
