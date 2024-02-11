import numpy as np
from collections import namedtuple
from scipy.optimize import minimize, curve_fit
from lottery import lotteries, generate_trials

Parameter = namedtuple('Parameter', ['default', 'bounds'])

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

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

#    def accept(self, P):
#        # Add bias in favor of first or second option
#        vmin = max(self.bias,     0)
#        vmax = min(1 + self.bias, 1)
#        return vmin + (vmax - vmin)*P
    
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

        print("Fitting %s... " % cls.__name__, end="")
        res = minimize(cls.log_likelihood,
                       x0=default,
                       bounds=bounds,
                       method="L-BFGS-B",
                       tol=1e-10,
                       options = {"maxiter": 1000,
                                 "disp" : False },
                       args = (trials, responses, kwargs))

        if res.success:
            print("success")
            return cls(*res.x)
        else:
            print("failed")
            return cls()
            


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
        "x0": Parameter(0.0, (-2.0, +2.0)),
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
    
    pnames = []
    for player in players.values():
        for key in player.parameters.keys():
            if key not in pnames:
                pnames += [key]
    s = "             "
    for pname in players.keys():
        s = s + "%-6s  " % pname
    print(s)
    print("--------" + "--------"*len(players))    
    for name in pnames:
        s = "%8s" % name
        for player in players.values():
            if name in player.parameters.keys():
                v = "%.3f" % getattr(player, name)
            else:
                v = "*****"
            s += "%8s" % v
        print(s)


def evaluate(players, trials, responses, per_trial=True):
    """
    Evaluate players on all lotteries and reports accuracy per
    trial (default) or per_lottery
    """
    
    s = "        "
    for pname in players.keys():
        s = s + "%-5s  " % pname
    print(s)
    print("----" + "-------"*len(players))
    
    for name, L in lotteries.items():
        trials = generate_trials(10_000, L)
        R0 = player.play(trials)
        s = "%s: " % name
        for p in players.values():
            R1 = p.play(trials)
            if per_trial:
                d = 1 - ((abs(R0-R1).sum())/len(R0))
            else:
                d = 1 - (abs(R0.sum()-R1.sum())/len(R0))            
            s = s + "   %.2f" % d
        print(s)
    print()

        
if __name__ == "__main__":
    import warnings 
    from lottery import L0, generate_trials

    warnings.filterwarnings('ignore')
    np.random.seed(123)
    
    # We create a random player and make it play lottery 0
    # player = RandomPlayer.random(bias=0)
    # player = SigmoidPlayer.random()
    player = ProspectPlayerP1.random()
    trials = generate_trials(10_000, L0)
    responses = player.play(trials)

    # Bias estimated from responses (should correspond to player bias)
    # print(player)
    # print("Estimated bias: %.3f" % (responses.sum()/len(responses)))
    # print()
    
    # We try to fit all kind of players to the actual player
    # based on its responses
    rd_fit = RandomPlayer.fit(trials, responses)
    sg_fit = SigmoidPlayer.fit(trials, responses)
    p1_fit = ProspectPlayerP1.fit(trials, responses)
    p2_fit = ProspectPlayerP2.fit(trials, responses)
    ge_fit = ProspectPlayerGE.fit(trials, responses)
    tk_fit = ProspectPlayerTK.fit(trials, responses)
    print()


    players = { "(%s)" % player.shortname : player,
                " RD " : rd_fit,
                " SG " : sg_fit,
                " P1 " : p1_fit,
                " P2 " : p2_fit,
                " GE " : ge_fit,
                " TK " : tk_fit }

    # We compare players parameters
    print(color.BOLD + "Comparison of fitted players" + color.END)
    show(players)
    print()

    # We evaluate players, including the original one
    print(color.BOLD + "Evaluation per trial" + color.END)
    evaluate(players, trials, responses)

    print()
    print(color.BOLD + "Evaluation per lottery type" + color.END)
    # We evaluate players, including the original one
    evaluate(players, trials, responses, True)


