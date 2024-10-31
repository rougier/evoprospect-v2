# Copyright 2024 (c) Naomi Chaix-Echel & Nicolas P Rougier
# Released under a BSD 2-clauses license

from collections import namedtuple
from prettytable import PrettyTable, SINGLE_BORDER
from scipy.optimize import minimize
from lottery import *

Parameter = namedtuple('Parameter', ['default', 'bounds'])
ALPHA_MAX = 2
ALPHA_MIN = 0.01
DELTA_MAX = 2
DELTA_MIN = 0.01


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
        "Make layer play given trials and return responses"
        # Compute probability of accepting first option
        P = self.accept(trials)
        # Actual play
        return (np.random.uniform(0, 1, len(P)) < P).astype(int)

    def get_data(self, lottery, n=10_000):
        """
        Generate n trials for the given lottery and make player
        play them. Return trials and responses.
        """

        trials = generate_trials(n, lottery)
        responses = self.play(trials)
        return trials, responses

    @classmethod
    def random(cls, **kwargs):
        parameters = {}
        for key in cls.parameters.keys():
            vmin, vmax = cls.parameters[key].bounds
            vmid = cls.parameters[key].default
            if np.random.uniform(0, 1) < (vmax - vmid) / (vmax - vmin):
                parameters[key] = np.random.uniform(vmin, vmid)
            else:
                parameters[key] = np.random.uniform(vmid, vmax)
        for key in kwargs:
            parameters[key] = kwargs[key]
        return cls(**parameters)

    @classmethod
    def min(cls, **kwargs):
        parameters = {}
        for key in cls.parameters.keys():
            vmin, vmax = cls.parameters[key].bounds
            parameters[key] = vmin
        for key in kwargs:
            parameters[key] = kwargs[key]
        return cls(**parameters)

    @classmethod
    def max(cls, **kwargs):
        parameters = {}
        for key in cls.parameters.keys():
            vmin, vmax = cls.parameters[key].bounds
            parameters[key] = vmax
        for key in kwargs:
            parameters[key] = kwargs[key]
        return cls(**parameters)

    @classmethod
    def get_bound(cls, key):
        for i,p in enumerate(cls.parameters.keys()):
            if p == key:
                idx = i
        bounds = [p.bounds for p in cls.parameters.values()]
        return bounds[idx]



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
                       options = {"maxiter": 10000,
                                 "disp" : False },
                       args = (trials, responses, kwargs))
        valid = res.success
        player = cls(*res.x)
        # See https://stackoverflow.com/questions/78545168/
        # -> finding-unused-variables-after-minimizing


        player.valid = valid
        if not valid:
            player.shortname = "!" + player.shortname

        return player

class RandomPlayer(Player):
    """
    Random player that choose first or second option randomly but
    with bias x0. Bias bounds are -3/+3 to be similar to sigmoid
    player.
    """
    shortname = "RND"
    parameters = {**Player.parameters, **{
        "x0": Parameter(0.0, (-3.0, +3.0))
    }}

    def accept(self, trials):
         P = (0.5 - self.x0/3) * np.ones(len(trials))
         return P # Player.accept(self, P)

class LeftPlayer(Player):
    """
    Left player only plays left option
    """
    shortname = "LPY"

    def accept(self, trials):
         return np.zeros(len(trials))


class RightPlayer(Player):
    """
    Right player only plays right option
    """
    shortname = "RPY"

    def accept(self, trials):
         return np.ones(len(trials))


class LazyLeftPlayer(Player):
    """
    Lazy left player plays left option all the time unless there's
    a big reward on the right
    """
    shortname = "LazyL"
    parameters = {**Player.parameters, **{
        "delta": Parameter(0.0, (0.0, +6.0))
    }}

    def accept(self, trials):
        V1, P1 = trials[:, 0], trials[:, 1]
        V2, P2 = trials[:, 2], trials[:, 3]
        return (V2*P2 - V1*P1) > self.delta


class LazyRightPlayer(Player):
    """
    Lazy right player plays right option all the time unless there's
    a big reward on the left.
    """
    shortname = "LazyR"
    parameters = {**Player.parameters, **{
        "delta": Parameter(0.0, (0.0, +6.0))
    }}

    def accept(self, trials):
        V1, P1 = trials[:, 0], trials[:, 1]
        V2, P2 = trials[:, 2], trials[:, 3]
        # return np.where((V1-V2) > self.delta, 1.0, 0.0)
        return (V1*P1 - V2*P2) > self.delta


class SigmoidPlayer(Player):
    """
    Player that plays according to a sigmoid applied to the
    different in expected value between first and second option.
    """
    shortname = "SG"
    parameters = {**Player.parameters, **{
        "x0": Parameter(0.0, (-5.0, +5.0)),
        "mu": Parameter(3.0, (0.1, 20.0))
    }}

    def sigmoid(self, X, x0=0.0, mu=1.0):
        return 1 / (1 + np.exp(-mu*(X - x0)))

    def subjective_utility(self, V):
        return V

    def subjective_probability(self, P, V):
        return P

    def accept(self, trials):
        V1, P1 = trials[:, 0], trials[:, 1]
        V2, P2 = trials[:, 2], trials[:, 3]
        V1 = self.subjective_utility(V1)
        P1 = self.subjective_probability(P1,V1)
        V2 = self.subjective_utility(V2)
        P2 = self.subjective_probability(P2,V2)
        P = self.sigmoid(V2*P2 - V1*P1, self.x0, self.mu)
        return P # Player.accept(self, P)


class ProspectPlayer(SigmoidPlayer):
    """
    Generic prospect player with unspecified subjective probability.
    """
    shortname = "PT"
    parameters = {**SigmoidPlayer.parameters, **{
        "lambda_": Parameter(1.0, (0.2, 10)),
        "rho_g":     Parameter(1.0, (0.1, 2)),
        "rho_l":     Parameter(1.0, (0.1, 2))
    }}

    def subjective_utility(self, V):
        return np.where(V > 0,
                        np.power(np.abs(V), self.rho_g),
                        -self.lambda_ * np.power(np.abs(V), self.rho_l))


class ProspectPlayerP1(ProspectPlayer):
    """
    """
    shortname = "P1"
    parameters = {**ProspectPlayer.parameters, **{
        "alpha": Parameter(1.0, (ALPHA_MIN, ALPHA_MAX)),
    }}

    def subjective_probability(self, P, V):
        return np.exp(- np.power((-np.log(P)), self.alpha))


class ProspectPlayerP2(ProspectPlayer):
    """
    """
    shortname = "P2"
    parameters = {**ProspectPlayer.parameters, **{
        "alpha": Parameter(1.0, (ALPHA_MIN, ALPHA_MAX)),
        "delta": Parameter(1.0, (DELTA_MIN, DELTA_MAX)),
    }}

    def subjective_probability(self, P, V):
        return np.exp(-self.delta*np.power((-np.log(P)), self.alpha))


class ProspectPlayerGE(ProspectPlayer):
    """
    """
    shortname = "GE"
    parameters = {**ProspectPlayer.parameters, **{
        "delta": Parameter(1.0, (DELTA_MIN, DELTA_MAX)),
        "gamma": Parameter(1.0, (ALPHA_MIN, ALPHA_MAX)),
    }}

    def subjective_probability(self, P, V):
        return (self.delta*np.power(P,self.gamma) /
           (self.delta *np.power(P,self.gamma) + np.power(1-P, self.gamma)))


class ProspectPlayerTK(ProspectPlayer):
    """
    """
    shortname = "TK"
    parameters = {**ProspectPlayer.parameters, **{
        "alpha": Parameter(1.0, (ALPHA_MIN, ALPHA_MAX)),
    }}

    def subjective_probability(self, P, V):
        return (np.power(P, self.alpha) /
                np.power((np.power(P, self.alpha)
                          + np.power(1-P, self.alpha)), 1/self.alpha))


class ProspectPlayerXX(SigmoidPlayer):
    """
    Experimental prospect player.
    """
    shortname = "XX"
    parameters = {**SigmoidPlayer.parameters, **{
        "gain": Parameter(1.0, (0.1, 2.0)),
        "alpha_g": Parameter(1.0, (0.1, 2.0)),
        "alpha_l": Parameter(1.0, (0.1, 2.0)),
    }}

    def subjective_utility(self, V):
        return np.where(V > 0, V, self.gain*V)

    def subjective_probability(self, P, V):
        return np.where(V > 0,
                        P**self.alpha_g,
                        P**self.alpha_l)


class DualProspectPlayerGE(ProspectPlayer):
    """
    """
    shortname = "GE+"
    parameters = {**ProspectPlayer.parameters, **{
        "alpha_g": Parameter(1.0, (ALPHA_MIN, ALPHA_MAX)),
        "alpha_l": Parameter(1.0, (ALPHA_MIN, ALPHA_MAX)),
        "delta_g": Parameter(1.0, (DELTA_MIN, DELTA_MAX)),
        "delta_l": Parameter(1.0, (DELTA_MIN, DELTA_MAX)),
        "gamma_g": Parameter(1.0, (0.1, 2)),
        "gamma_l": Parameter(1.0, (0.1, 2)),
    }}

    def subjective_probability(self, P, V):
        return np.where(V > 0,
                        (self.delta_g * np.power(P, self.alpha_g) /
                         (self.delta_g * np.power(P, self.gamma_g) + np.power(1-P, self.alpha_g))),
                        (self.delta_l * np.power(P, self.alpha_l) /
                         (self.delta_l * np.power(P, self.gamma_l) + np.power(1-P, self.alpha_l))))


class DualProspectPlayerTK(ProspectPlayer):
    """
    """
    shortname = "TK+"
    parameters = {**ProspectPlayer.parameters, **{
        "alpha_g": Parameter(1.0, (ALPHA_MIN, ALPHA_MAX)),
        "alpha_l": Parameter(1.0, (ALPHA_MIN, ALPHA_MAX))
    }}

    def subjective_probability(self, P, V):
        return np.where(V > 0,
                        (np.power(P, self.alpha_g) /
                         np.power((np.power(P, self.alpha_g)
                                   + np.power(1-P, self.alpha_g)), 1/self.alpha_g)),
                        (np.power(P, self.alpha_l) /
                         np.power((np.power(P, self.alpha_l)
                                   + np.power(1-P, self.alpha_l)), 1/self.alpha_l)))


class DualProspectPlayerP2(ProspectPlayer):
    """
    """
    shortname = "P2+"
    parameters = {**ProspectPlayer.parameters, **{
        "alpha_g": Parameter(1.0, (ALPHA_MIN, ALPHA_MAX)),
        "alpha_l": Parameter(1.0, (ALPHA_MIN, ALPHA_MAX)),
        "delta_g": Parameter(1.0, (DELTA_MIN, DELTA_MAX)),
        "delta_l": Parameter(1.0, (DELTA_MIN, DELTA_MAX)),
    }}

    def subjective_probability(self, P, V):
        return np.where(V > 0,
                        np.exp(-self.delta_g*np.power((-np.log(P)), self.alpha_g)),
                        np.exp(-self.delta_l*np.power((-np.log(P)), self.alpha_l)))


class DualProspectPlayerP1(ProspectPlayer):
    """
    """
    shortname = "P1+"
    parameters = {**ProspectPlayer.parameters, **{
        "alpha_g": Parameter(1.0, (ALPHA_MIN, ALPHA_MAX)),
        "alpha_l": Parameter(1.0, (ALPHA_MIN, ALPHA_MAX)),
    }}

    def subjective_probability(self, P, V):
        return np.where(V > 0,
                        np.exp(- np.power((-np.log(P)), self.alpha_g)),
                        np.exp(- np.power((-np.log(P)), self.alpha_l)))


def show(reference=None, players=[]):
    "Display reference and players parameters side by side"

    x = PrettyTable(border=True, align="l")
    x.set_style(SINGLE_BORDER)
    if reference is not None:
        x.field_names = ([bold("Parameter")] +
                         [bold("(%s)" % reference.shortname)] +
                         [bold("%s" % player.shortname) for player in players])
        players = [reference] + players
    else:
        x.field_names = ([bold("Parameter")] +
                         [bold("%s" % player.shortname) for player in players])
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


def evaluate_player_1(player, trials, responses, n=100):
    """
    Make the player play n times the trials and average the
    difference with reponses.
    """

    R = [player.play(trials) for _ in range(n)]
    R = [1 - (abs(responses-r).sum() / len(trials)) for r in R]
    return np.mean(R)

def evaluate_player_2(player, trials, responses, n=1000):
    """
    Separate trials in unique trials and evaluate each type of trials
    """
    # Get unique trials (and counts)
    T = np.unique(trials, axis=0)

    # Get mean response over each type of trial
    R0 = [np.mean(responses[np.argwhere((trials == trial).all(axis=1))]) for trial in T]

    # Get mean response from player for each type of trial played n times
    R = np.mean([player.play(T) for _ in range(n)], axis=0)

    #print(R0)
    #print(R)

    return 1 - np.mean(abs(R0 - R))

def calculate_diff(R0, R_player):
    return np.mean(np.abs(R0 - R_player))

def evaluate(reference, players, n=1_000, evaluate_method=evaluate_player_2):

    x = PrettyTable(border=True, align="l")
    x.set_style(SINGLE_BORDER)
    if isinstance(reference, (Player)):
        x.field_names = ([bold("Lottery")] +
                         [bold("(%s)" % reference.shortname)] +
                         [bold(p.shortname) for p in players])
        players = [reference] + players
    else:
        x.field_names = ([bold("Lottery")] +
                         [bold(p.shortname) for p in players])

    for i in range(8):
        name = bold("L%d" % i)
        trials, responses = reference.get_data(i)
        R = [evaluate_method(p, trials, responses) for p in players]
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

    # We create a player and make it play lottery 0
    # player = RandomPlayer.random(x0=0)
    # player = SigmoidPlayer.random()
    # player = RightPlayer.random()
    player = ProspectPlayerP2.random(x0=3)
    trials = generate_trials(5_000, L0)
    responses = player.play(trials)

    # We try to fit each player against player
    players = [ RandomPlayer.fit(trials, responses),
                SigmoidPlayer.fit(trials, responses),
                ProspectPlayerP1.fit(trials, responses),
                ProspectPlayerP2.fit(trials, responses),
                ProspectPlayerGE.fit(trials, responses),
                ProspectPlayerTK.fit(trials, responses) ]
    show(player, players)
    evaluate(player, players, 100, evaluate_player_2)







