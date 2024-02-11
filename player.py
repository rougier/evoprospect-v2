import numpy as np
from collections import namedtuple
from prettytable import PrettyTable
from scipy.optimize import minimize
from lottery import lotteries, generate_trials


Parameter = namedtuple('Parameter', ['default', 'bounds'])

class color:

    # Color
    BLACK = '\033[90m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    GRAY = '\033[97m'

    # Style
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    # BackgroundColor
    BgBLACK = '\033[40m'
    BgRED = '\033[41m'
    BgGREEN = '\033[42m'
    BgORANGE = '\033[43m'
    BgBLUE = '\033[44m'
    BgPURPLE = '\033[45m'
    BgCYAN = '\033[46m'
    BgGRAY = '\033[47m'

    # End
    END = '\033[0m'

def bold(text): return color.BOLD + color.BLACK + text + color.END
def red(text): return color.RED + text + color.END

	
class Player:
    """
    Generic player
    """
    shortname = "GP"
    parameters = { }
    
    def __init__(self, *args, **kwargs):
        keys    = list(type(self).parameters.keys())
        values  = list(type(self).parameters.values())
        self.parameters = { }
        for key, value in zip(keys,values):
            self.parameters[key] = value.default
        for key, value in zip(keys[:len(args)], args):
            self.parameters[key] = value
        self.parameters.update(kwargs)

    def __getattr__(self, name):
        keys = type(self).parameters.keys()
        if name in keys:
            return self.parameters[name]
        raise AttributeError

    def __setattr__(self, name, value):
        keys = type(self).parameters.keys()
        if name in keys:
            self.parameters[name] = value
        object.__setattr__(self, name, value)

    def __repr__(self):
        return "{cls}({attrs})".format(
            cls = self.__class__.__name__,
            attrs=", ".join("{}={:.3f}".format(k, v)
                           for k, v in self.parameters.items()))
    
    def play(self, trials):
        
        # Compute proability of accepting first option
        P = self.accept(trials)

        # Actual play
        return (np.random.uniform(0, 1, len(P)) < P).astype(int)

    @classmethod
    def random(cls, **kwargs):        
        parameters = {}
        for key in cls.parameters.keys():
            vmin,vmax = cls.parameters[key].bounds
            parameters[key] = np.random.uniform(vmin, vmax)
        for key in kwargs:
            parameters[key] = kwargs[key]
        return cls(**parameters)

        
    @classmethod
    def log_likelihood(cls, params, trials, responses, kwargs):
        """
        Compute the log likelihood of a player (cls/params)
        playing trials.
        """

        player = cls(*params, **kwargs)
        P = player.accept(trials)
        I = np.where(np.logical_and(P > 0, P < 1))
        P, R = P[I], responses[I]
        log_likelihood = R*np.log(P) + (1-R)*(np.log(1-P))
        return -log_likelihood.sum()

    @classmethod
    def fit(cls, trials, responses, **kwargs):
        """
        Create a player of the given class that best fit the given responses.
        Fitting is done through maximum likelihood.

        During fit, some parameters can be fixed if their value is
        provided (kwargs).
        """

        default = [p.default for p in cls.parameters.values()]
        bounds = [p.bounds for p in cls.parameters.values()]

        # print("Fitting %s... " % cls.__name__, end="")
        res = minimize(cls.log_likelihood,
                       x0=default,
                       bounds=bounds,
                       method="L-BFGS-B",
                       tol=1e-10,
                       options = {"maxiter": 1000,
                                 "disp" : False },
                       args = (trials, responses, kwargs))

        if res.success:
            return cls(*res.x)
        else:
            return None
            


class RandomPlayer(Player):
    """
    Random player that choose first or second option randomly
    (modulo bias)
    """
    shortname = "RD"
    parameters = Player.parameters | {
        "bias": Parameter(0.0, (-0.5, +0.5))
    }
                   
    def accept(self, trials):
         P = (0.5 + self.bias) * np.ones(len(trials))
         return P # Player.accept(self, P)
    
class SigmoidPlayer(Player):
    """
    Player that plays according to a sigmoid applied to the
    different in expected value between first and second option.
    """
    shortname = "SG"
    parameters = Player.parameters | {
        "x0": Parameter(0.0, (-3.0, +3.0)),
        "mu": Parameter(5.0, ( 0.1, 10.0))
    }

    def sigmoid(self, X, x0=0.0, mu=1.0):
        return 1 / (1 + np.exp(-mu*(X - x0)))

    def subjective_utility(self, V):
        return V

    def subjective_probability(self, P):
        return P
    
    def accept(self, trials):
        V1, P1 = trials[:,0], trials[:,1]
        V2, P2 = trials[:,2], trials[:,3]
        V1 = self.subjective_utility(V1)
        P1 = self.subjective_probability(P1)
        V2 = self.subjective_utility(V2)
        P2 = self.subjective_probability(P2)
        P = self.sigmoid(V2*P2 - V1*P1, self.x0, self.mu)
        return P # Player.accept(self, P)

    
class ProspectPlayer(SigmoidPlayer):
    """
    Generic prospect player with unspecified subjective utility
    and probability.
    """
    shortname = "PT"
    parameters = SigmoidPlayer.parameters | {
        "lambda_": Parameter(1.0, (0.1, 3.0)),
        "rho":     Parameter(1.0, (0.1, 3.0))
    }

    def subjective_utility(self, V):
        return np.where(V > 0,
                        np.power(np.abs(V), self.rho),
                        -self.lambda_ * np.power(np.abs(V), self.rho))

    def subjective_probability(self, P):
        raise NotImplementedError

    
class ProspectPlayerP1(ProspectPlayer):
    """
    """
    shortname = "P1"
    parameters = ProspectPlayer.parameters | {
        "alpha": Parameter(1.0, (0.1, 3.0)),
    }

    def subjective_probability(self, P):
        return np.exp(- np.power((-np.log(P)), self.alpha))


class ProspectPlayerP2(ProspectPlayer):
    """
    """
    shortname = "P2"
    parameters = ProspectPlayer.parameters | {
        "alpha": Parameter(1.0, (0.1, 3.0)),
        "delta": Parameter(1.0, (0.1, 3.0)),
    }

    def subjective_probability(self, P):
        return np.exp(-self.delta*np.power((-np.log(P)), self.alpha))

    
class ProspectPlayerGE(ProspectPlayer):
    """
    """
    shortname = "GE"
    parameters = ProspectPlayer.parameters | {
        "alpha": Parameter(1.0, (0.1, 3.0)),
        "delta": Parameter(1.0, (0.1, 3.0)),
        "gamma": Parameter(1.0, (0.1, 3.0)),
    }

    def subjective_probability(self, P):
        return (self.delta*np.power(P,self.alpha) /
           (self.delta *np.power(P,self.gamma) + np.power(1-P, self.alpha)))


class ProspectPlayerTK(ProspectPlayer):
    """
    """
    shortname = "TK"
    parameters = ProspectPlayer.parameters | {
        "alpha": Parameter(1.0, (0.1, 3.0)),
    }

    def subjective_probability(self, P):
        return (np.power(P, self.alpha) /
                np.power((np.power(P, self.alpha)
                          + np.power(1-P, self.alpha)), 1/self.alpha))

    
def show(players):
    "Display players parameters side by side"

    x = PrettyTable(border=False, align="l")
    x.field_names = ([bold("Parameter")] +
                     [bold("(%s)" % players[0].shortname)] + 
                     [bold(p.shortname) for p in players[1:]])
    names = []
    for player in players:
        for key in player.parameters.keys():
            if key not in names:
                names += [key]
    for name in names:
        row = [bold(name)]
        for player in players:
            if name in player.parameters.keys():
                row += ["%+.3f" % getattr(player, name)]
            else:
                row += [" -----"]
        x.add_row(row)
    print(x)


def evaluate(players, lotteries):

    x = PrettyTable(border=False, align="l")
    x.field_names = ([bold("Lottery")] +
                     [bold("(%s)" % players[0].shortname)] + 
                     [bold(p.shortname) for p in players[1:]])
    for i in range(len(lotteries)):
        name = bold("L%d" % i)
        lottery = lotteries[i]
        trials = generate_trials(10_000, lottery)
        R0 = players[0].play(trials)
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

# def evaluate(players, trials, responses):
#     """
#     Evaluate players on all lotteries and reports accuracy per
#     trial (default) or per_lottery
#     """

#     x = PrettyTable(border=False, align="l")
#     x.field_names = ["Lottery"] + list(players.keys())
#     for name, lottery in lotteries.items():
#         trials = generate_trials(10_000, lottery)
#         R0 = player.play(trials)
#         row = [1 - ((abs(R0-R).sum())/len(R0))
#                for R in [p.play(trials) for p in players.values()]]
# rowmax = np.max(row)
#         x.add_row([name] + ["%.3f" % r for r in row])

#     print(x)
        

        
if __name__ == "__main__":
    import warnings 
    from lottery import *

    warnings.filterwarnings('ignore')
    # np.random.seed(123)
    
    # We create a player and make it play lottery 0
    # player = RandomPlayer.random(bias=0)
    # player = SigmoidPlayer.random()
    player = ProspectPlayerP1.random()
    trials = generate_trials(10_000, L0)
    responses = player.play(trials)

    players = [ RandomPlayer.fit(trials, responses),
                SigmoidPlayer.fit(trials, responses),
                ProspectPlayerP1.fit(trials, responses),
                ProspectPlayerP2.fit(trials, responses),
                ProspectPlayerGE.fit(trials, responses),
                ProspectPlayerTK.fit(trials, responses) ]

    show([player] + players)
    print()
    evaluate([player] + players, lotteries)

        
    # Bias estimated from responses (should correspond to player bias)
    # print(player)
    # print("Estimated bias: %.3f" % (responses.sum()/len(responses)))
    # print()
    
    # We try to fit all kind of players to the actual player
    # based on its responses

    
    # rd_fit = RandomPlayer.fit(trials, responses)
    # sg_fit = SigmoidPlayer.fit(trials, responses)
    # p1_fit = ProspectPlayerP1.fit(trials, responses)
    # p2_fit = ProspectPlayerP2.fit(trials, responses)
    # ge_fit = ProspectPlayerGE.fit(trials, responses)
    # tk_fit = ProspectPlayerTK.fit(trials, responses)
    # print()

    # players = { "(%s)" % player.shortname : player,
    #             "RD" : rd_fit,
    #             "SG" : sg_fit,
    #             "P1" : p1_fit,
    #             "P2" : p2_fit,
    #             "GE" : ge_fit,
    #             "TK" : tk_fit }

    # # We compare players parameters
    # print(color.BOLD + "Comparison of fitted players" + color.END)
    # show(players)
    # print()

    # We evaluate players, including the original one
    # print(color.BOLD + "Evaluation (per trials)" + color.END)
    # evaluate([player] + players, lotteries)


    # print()
    # print(color.BOLD + "Evaluation per lottery type" + color.END)
    # # We evaluate players, including the original one
    # evaluate(players, trials, responses, True)


