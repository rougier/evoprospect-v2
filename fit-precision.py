# To measure the precision of the fit procedure, we use a random
# player (target) of the specified type and make it play a lottery
# with a various number of trials (from 10 to 2000). We then try to
# fit p (=25) players against the trials and responses and measure how
# well the p players predict the responses of the target. We do the
# same for the target since the evaluation procedure is stochastic. We
# finally plot the difference betweeen the grountruth (target) and the
# mean of the p players. From the results, 1500 trials seems
# (visually) to be enough to have a very good precision (less than
# 0.5% error).

if __name__ == "__main__":
    import warnings
    import os.path
    import matplotlib.pyplot as plt

    from player import *
    from lottery import *

    # np.random.seed(1)
    warnings.filterwarnings('ignore')
    filename = "./fit-precision-2000-100.npy"

    N = (list(range(   1,      10,   1)) +
         list(range(  10,    100,   10)) +
         list(range( 100,  1_000,  100)) +
         list(range(1000,10_001,  1_000)))
    N = np.arange(10,2_001,10)

    if not os.path.exists(filename):
        results = np.zeros((len(N), 25), dtype=[("fit", float),
                                                ("target", float)])
        for i in range(len(N)):
            for j in range(results.shape[1]):
                target = DualProspectPlayerGE.random()

                # Generate trials/responses for fit
                trials = generate_trials(N[i], L0)
                responses = target.play(trials)
                player = DualProspectPlayerGE.fit(trials, responses)

                # Test fitted player on a new set of trials/responses
                trials = generate_trials(1_000, L0)
                responses = target.play(trials)

                results["target"][i,j] = evaluate_player_2(target, trials, responses, 1_000)
                results["fit"][i,j] = evaluate_player_2(player, trials, responses, 1_000)


            print("FIT (n=%d): %.3f ±%.3f" %(N[i], np.mean(results["fit"][i]),
                                                   np.std(results["fit"][i])))
            print("TGT (n=%d): %.3f ±%.3f" %(N[i], np.mean(results["target"][i]),
                                                   np.std(results["target"][i])))
        np.save(filename, results)
    else:
        results = np.load(filename)

    X = N[10:]
    results = results[10:]

    Y = abs(results["fit"] - results["target"]).mean(axis=-1)
    E = abs(results["fit"] - results["target"]).std(axis=-1)


    plt.figure(figsize=(6,3))
    plt.plot(X, Y, color="C1")
    plt.fill_between(X, Y-E, Y+E, facecolor="C1", edgecolor="None", alpha=0.25)

    plt.axvline(1500, color="black", linestyle="--", linewidth=1.0)

    plt.xlabel("Number of trials used for fitting")
    plt.ylabel("Fitting error")
    plt.xlim(N[10],2000)

    plt.tight_layout()
    plt.savefig("fit-precision.pdf")
    plt.show()
