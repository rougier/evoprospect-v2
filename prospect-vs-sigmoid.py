# Copyright 2024 (c) Naomi Chaix-Echel & Nicolas P Rougier
# Released under a BSD 2-clauses license




if __name__ == "__main__":
    import numpy as np
    import warnings
    from player import *
    from lottery import *

    warnings.filterwarnings('ignore')
    np.random.seed(123)

    L = L0 #np.concatenate([L6,L7], axis=0)

    for x0 in np.linspace(-5, 5, 21):
        print("x0 = %.2f" % x0)

        sigmoid = []
        prospect = []

        for i in range(25):
            player = ProspectPlayerP2.random(x0=x0)
            trials = generate_trials(5_000, L)
            responses = player.play(trials)

            p1 = SigmoidPlayer.fit(trials, responses)
            score = evaluate_player_2(p1, trials, responses, 100)
            sigmoid.append(score)

            p2 = ProspectPlayerP2.fit(trials, responses)
            score = evaluate_player_2(p2, trials, responses, 100)
            prospect.append(score)

            #show(player, [p1,p2])
            #evaluate(player, [p1,p2], 100, evaluate_player_2)



        print("Sigmoid:  ", np.mean(sigmoid))
        print("Prospect: ", np.mean(prospect))
        print()
