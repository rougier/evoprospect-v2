import tqdm
from player import *
from lottery import *

def reward(player, trials):
    choices = player.play(trials)
    V = np.where(choices==0, trials[:,0], trials[:,2])
    P = np.where(choices==0, trials[:,1], trials[:,3])
    return (V*(np.random.uniform(0,1,len(trials)) < P)).sum()

def mix(player1, player2):
    Player = player1.__class__
    p1 = np.asarray(list(player1.parameters.values()))
    p2 = np.asarray(list(player2.parameters.values()))
    d = np.random.uniform(0, 1, len(p1))
    p = d*p1 + (1-d)*p2
    return Player(*p)

def crossover(players, rewards):
    n = len(players)
    parents = np.random.choice(np.arange(n), size=(n,2), p=rewards/rewards.sum())
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


n_trials = 100
n_players = 1000
n_generation = 1000
mutation_rate = 0.01
lottery = L6
Player = ProspectPlayerP1
player_min = Player.min()
player_max = Player.max()
# players = [ProspectPlayerP1() for i in range(n_players)]
players = [Player.random() for i in range(n_players)]

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
