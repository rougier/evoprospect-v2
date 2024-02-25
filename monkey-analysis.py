# Copyright 2024 (c) aomi Chaix-Echel & Nicolas P Rougier
# Released under a BSD 2-clauses license

if __name__ == "__main__":

    from player import *
    from monkey import monkeys

    lottery = 0
    for monkey in monkeys:
        trials, responses = monkey.get_data(lottery=lottery)
        n = len(trials)
        print(color.BOLD + color.RED + repr(monkey) + color.END)
        print(f"Fitting using lottery {lottery} (n={n:,})",  end="", flush=True)
        
        players = []        
        for cls in [RandomPlayer, SigmoidPlayer,
 #                   ProspectPlayerXX,
                    DualProspectPlayerP1, DualProspectPlayerP2,
                    DualProspectPlayerGE, DualProspectPlayerTK,
#                    ProspectPlayerP1, ProspectPlayerP2,
#                    ProspectPlayerGE, ProspectPlayerTK
                    ]:
            player = cls.fit(trials, responses)
            players.append(player)
            if player.valid:
                print(".", end="", flush=True)
            else:
                print("x", end="", flush=True)
        print(" done!")
        
        # We compare players parameters
        show(players=players)

        # We evaluate players, including the original one
        evaluate(monkey, players, n=1000, evaluate_method=evaluate_player_2)


                
