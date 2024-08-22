import tqdm
from player import *
from lottery import *

def reward(player, trials):
    choices = player.play(trials)
    V = np.where(choices==0, trials[:,0], trials[:,2])
    P = np.where(choices==0, trials[:,1], trials[:,3])
    return (V*(np.random.uniform(0,1,len(trials)) < P)).sum()


def mix(player1, player2):
    from scipy import interpolate, optimize
    Player = player1.__class__
    parameters = []

    # Interpolation between p1 and p2
    d = np.random.uniform(0, 1, len(Player.parameters))

    # Mix between p1 and p2
    # d = np.random.choice([0,1], len(p1))

    # We mix players in the uniform x domain to avoid biases
    for i, (name,parameter) in enumerate(Player.parameters.items()):
        vmin, vmax = parameter.bounds
        vmid = parameter.default
        f = interpolate.interp1d([0.0, 0.5, 1.0], [vmin, vmid, vmax])
        x1 = optimize.newton(lambda x: f(x) - player1[name], 0.5)
        x2 = optimize.newton(lambda x: f(x) - player2[name], 0.5)
        x = d[i]*x1 + (1-d[i])*x2
        parameters.append(f(x))
    return Player(*parameters)


def crossover(players, rewards):

    # n = len(players)
    # selection = 0.25, 1.00
    # I = np.argsort(-rewards)
    # Imin = int(selection[0]*len(I))
    # Imax = int(selection[1]*len(I))
    # I = I[Imin:Imax]
    # parents = np.random.choice(I, size=(n,2))
    # children = [mix(players[p1],players[p2]) for p1,p2 in parents]
    # return children

    n = len(players)
    P = rewards - np.min(rewards)
    parents = np.random.choice(np.arange(n), size=(n,2), p=P/P.sum())
    children = [mix(players[p1],players[p2]) for p1,p2 in parents]
    return children

def mutate(players, rate):
    n = len(players)
    Player = players[0].__class__
    mutated = []
    mutations = np.random.uniform(0, 1, n) < rate
    for i in mutations.nonzero()[0]:
        players[i] = Player.random()
    return players




seed = 1
n_trials = 100
n_players = 250
n_generation = 100
mutation_rate = 0.05
np.random.seed(seed)

Player = ProspectPlayerP1
player_min = Player.min()
player_max = Player.max()
players = [Player.random() for i in range(n_players)]

# Lotteries with same expected value
lottery = np.random.uniform(0.1, .9, (100,4))
lottery[:,0] *= 4
lottery[:,2] *= 4
lottery[:,3] = (lottery[:,0]*lottery[:,1])/lottery[:,2]
# lottery = L6
# lottery = [(v1,p1,v2,p2) for v1,p1,v2,p2 in L6 if v1*p1 == v2*p2]

p = np.mean([list(player.parameters.values()) for player in players], axis=0)
player_mean = Player(*p)
print("Mean player: ", player_mean)
print()

for i in tqdm.trange(n_generation):
    trials  = generate_trials(n_trials, lottery)
    rewards = np.asarray([reward(player, trials) for player in players])
    players = crossover(players, rewards)
    players = mutate(players, mutation_rate)

p = np.mean([list(player.parameters.values()) for player in players], axis=0)
player_mean = Player(*p)

# Display
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(1,3,1)
X = np.linspace(0, 1, 256)
for player in players:
    ax.plot(X, player.subjective_utility(X), color="C0", alpha=0.25, linewidth=0.5)
ax.plot(X, X, color="black", linestyle="--", linewidth=1)
ax.plot(X, player_min.subjective_utility(X), color="black", linestyle="--", linewidth=1)
ax.plot(X, player_max.subjective_utility(X), color="black", linestyle="--", linewidth=1)
ax.plot(X, player_mean.subjective_utility(X), color="black", linestyle="-", linewidth=1.5)
ax.set_title("Subjective utility")

ax = fig.add_subplot(1,3,2)
X = np.linspace(0,1,256)
for player in players:
    ax.plot(X, player.subjective_probability(X, 1), color="C1", alpha=0.25, linewidth=0.5)
ax.plot(X, X, color="black", linestyle="--", linewidth=1)
ax.plot(X, player_min.subjective_probability(X, 1), color="black", linestyle="--", linewidth=1)
ax.plot(X, player_max.subjective_probability(X, 1), color="black", linestyle="--", linewidth=1)
ax.plot(X, player_mean.subjective_probability(X, 1), color="black", linestyle="-", linewidth=1.5)
ax.set_title("Subjective probability")


ax = fig.add_subplot(1,3,3)
X = np.linspace(-6,6,256)
for player in players:
    ax.plot(X, player.sigmoid(X), color="C2", alpha=0.25, linewidth=0.5)
ax.plot(X, player_mean.sigmoid(X), color="black", linestyle="-", linewidth=1.5)
ax.set_title("Softmax function")

plt.tight_layout()
plt.savefig("prospect-evolution.pdf")
plt.show()
