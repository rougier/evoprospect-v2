
if __name__ == "__main__":

    from player import *
    from monkey import monkeys

    for monkey in monkeys:
        print(color.BOLD + color.RED + repr(monkey) + color.END)
        print("Fitting ",  end="", flush=True)
        
        trials, responses = monkey.get_data(lottery=0)
        players = []        
        for cls in [RandomPlayer, SigmoidPlayer,
                    ProspectPlayerXX,
                    ProspectPlayerP1, ProspectPlayerP2,
                    ProspectPlayerGE, ProspectPlayerTK]:
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


                
