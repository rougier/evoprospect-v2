
import numpy as np
import pandas as pd
data = pd.read_csv("./data/data-processed.csv")
subject_ids = data['subject_id'].unique()
task_ids = np.sort(data['task_id'].unique())

def get_trials(subject_id=None, task_id=None):
    if isinstance(subject_id, str):
        subject_id = [subject_id]        
    if isinstance(task_id, int):
        task_id = [task_id]        
    if subject_id is not None and task_id is not None:
        df = data.loc[(data['task_id'].isin(task_id)) &
                      (data['subject_id'].isin(subject_id))]
    elif subject_id is not None:
        df = data.loc[(data['subject_id'].isin(subject_id))]
    elif task_id is not None:
        df = data.loc[(data['task_id'].isin(task_id))]
    else:
        df = data

    trials = df[["V_left","P_left","V_right","P_right"]]
    responses = df["response"]
    return np.asarray(trials), np.asarray(responses)

class Monkey:
    """Monday data abstraction """
    
    def __init__(self, name):
        self.shortname = name.upper()
        self.data = []
        for index in range(1,8):
            trials, responses = get_trials(name, index)
            self.data.append((trials, responses))

        # To mimic lotteries,  data[0] is the concatenation
        # of all other data even though there exist data for index 0
        #all = (np.concatenate([d[0] for d in self.data], axis=0),
        #       np.concatenate([d[1] for d in self.data], axis=0))
        all = get_trials(name)        
        self.count = len(all[0])
        self.bias = 0.5 - all[1].sum() / len(all[0])
        self.data = [all] + self.data

    def __repr__(self):
        return f"Monkey({self.shortname}, n={self.count:,}, bias={self.bias:.3f})"

    def get_data(self, lottery, n=0):
        trials, responses = self.data[lottery]
        if n < 1:
            return trials, responses
        I = np.random.randint(0, len(trials), n)
        return trials[I], responses[I]
        

monkeys = []
print("Get monkey data", end="", flush=True)
for name in subject_ids:
    print(".", end="", flush=True)
    monkey = Monkey(name)
    monkeys.append(monkey)
print(" done!")

# for monkey in monkeys: print(monkey)
# print(monkeys[0].get_data(0, n=10))
