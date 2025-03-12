# Social hierarchy influences monkeys' risky decisions
# Copyright 2024 (c) Naomi Chaix-Echel & Nicolas P Rougier
# Released under a BSD 2-clauses license
import itertools
import numpy as np

# This generates all the lotteries (L1 to L7), L0 being all of them
# L0: no condition
# L1: p1 = p2, v1 > 0, v2 < 0
# L2: p1 = p2, v1 > v2 > 0
# L3: p1 = p2, v1 < v2 < 0
# L4: p1 > p2, v1 = v2 > 0
# L5: p1 < p2, v1 = v2 < 0
# L6: p1 < p2, v1 > v2 > 0
# L7: p1 < p2, x1 < v2 < 0

# Values of stimuli
V = [-3, -2, -1, 0, +1, +2, +3]

# Probabilities of stimuli
P = [0.25, 0.50, 0.75, 1.00]

# All the combinations VxP
VP = list(itertools.product(V, P))

# All the combinations VxP x VxP
L0 = list(itertools.product(VP, VP))

# All the lotteries we're interested in
L1 = [(v1, p1, v2, p2) for (v1, p1), (v2 ,p2) in L0
      if (v1 > 0) and (v2 < 0) and (p1 == p2)]

L2 = [(v1, p1, v2, p2) for (v1, p1), (v2, p2) in L0
      if (v1 > v2) and (v2 > 0) and (p1 == p2)]

L3 = [(v1, p1, v2, p2) for (v1, p1),(v2, p2) in L0
      if (v1 < v2) and (v2 < 0) and (p1 == p2)]

L4 = [(v1, p1, v2, p2) for (v1, p1), (v2, p2) in L0
      if (v1 == v2) and (v2 > 0) and (p1> p2)]

L5 = [(v1, p1, v2, p2) for (v1, p1), (v2, p2) in L0
      if (v1 == v2) and (v2 < 0) and (p1 < p2)]

L6 = [(v1, p1, v2, p2) for (v1, p1), (v2, p2) in L0
      if (v1 > v2) and (v2 > 0) and (p1 < p2)]

L7 = [(v1, p1, v2, p2) for (v1, p1), (v2, p2) in L0
      if (v1 < v2) and (v2 < 0) and (p1 < p2)]

L0 = np.concatenate([L1, L2, L3, L4, L5, L6, L7], axis=0)
lotteries = [L0, L1, L2, L3, L4, L5, L6, L7]



def generate_trials(n, lottery):

    if isinstance(lottery, (int)):
        lottery = lotteries[lottery]
    
    lottery = np.asarray(lottery).reshape(-1,4)
    L = lottery[np.random.randint(0, len(lottery), n)]
    swap = np.random.randint(0, 2, n)
    for index in range(n):
        v1, p1, v2, p2 = L[index]
        if swap[index]:
            L[index] = v2, p2, v1, p1
    return L



