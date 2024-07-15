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

    for d in np.arange(0, 6, 0.5):
        print("delta = %.2f" % d)

        sigmoid = []
        prospect = []
        bias = []
        for i in range(25):
            player = LazyLeftPlayer.random(delta=d)
            trials = generate_trials(5_000, L)
            responses = player.play(trials)

#            print("Bias: %.3f / %.3f" % (
#                responses.sum()/len(responses),
#                1-responses.sum()/len(responses)))
            bias.append(responses.sum()/len(responses))


            p1 = SigmoidPlayer.fit(trials, responses)
            score = evaluate_player_2(p1, trials, responses, 100)
            sigmoid.append(score)

            p2 = ProspectPlayerGE.fit(trials, responses)
            score = evaluate_player_2(p2, trials, responses, 100)
            prospect.append(score)

            #show(player, [p1,p2])
            #evaluate(player, [p1,p2], 100, evaluate_player_2)

        print("Mean bias:  ", np.mean(bias))
        print("Sigmoid:  ", np.mean(sigmoid))
        print("Prospect: ", np.mean(prospect))
        print()
