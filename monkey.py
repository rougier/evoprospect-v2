# Copyright 2024 (c) Naomi Chaix-Echel & Nicolas P Rougier
# Released under a BSD 2-clauses license

import numpy as np
import pandas as pd
from prettytable import PrettyTable, SINGLE_BORDER

data = pd.read_csv("./data/data-processed.csv", sep=",", parse_dates=['date'])
data.sort_values(by="date", inplace=True)
subject_ids = data['subject_id'].unique()
task_ids = np.sort(data['task_id'].unique())
subject_ids = data['subject_id'].unique()
task_ids = np.sort(data['task_id'].unique())

#data = data.dropna(subset=['response'])


def get_trials(subject_id=None, task_id=None, date_range=None, return_rewards=False):
    if isinstance(subject_id, str):
        subject_id = [subject_id]
    if isinstance(task_id, int):
        task_id = [task_id]
    if subject_id is not None and task_id is not None and date_range is not None:
        df = data.loc[(data['task_id'].isin(task_id)) &
                      (data['subject_id'].isin(subject_id)) &
                      (data['date'] <= date_range[1]) &
                      (data['date'] >= date_range[0])]
    elif subject_id is not None and task_id is not None:
        df = data.loc[(data['task_id'].isin(task_id)) & (data['subject_id'].isin(subject_id))]
    elif subject_id is not None and date_range is not None:
        df = data.loc[(data['subject_id'].isin(subject_id)) &
                      (data['date'] <= date_range[1]) &
                      (data['date'] >= date_range[0])]
    elif subject_id is not None:
        df = data.loc[(data['subject_id'].isin(subject_id))]
    elif task_id is not None:
        df = data.loc[(data['task_id'].isin(task_id))]
    else:
        df = data

    trials = df[["V1_left", "P1_left", "V1_right", "P1_right"]]
    responses = df["response"]
    if return_rewards:
        rewards = df['reward']
        return np.asarray(trials), np.asarray(responses), np.asarray(rewards)
    else:
        return np.asarray(trials), np.asarray(responses)


def print_n_trials_per_monkey():
    x = PrettyTable(border=True, align="l")
    x.set_style(SINGLE_BORDER)
    x.field_names = (["Lottery"] + ['L{}'.format(str(lottery)) for lottery in range(1, 8)])
    for monkey in monkeys:
        all_n = []
        for lottery in range(1, 8):
            n = len(data[(data['subject_id'] == monkey.shortname.lower()) & (data['task_id'] == lottery)])
            all_n.append(n)
        x.add_row([monkey.shortname] + all_n)

    print(x)


class Monkey:
    """Monday data abstraction """
    
    def __init__(self, name, date_range=None, return_rewards=False):
        self.shortname = name.upper()
        self.data = []
        for index in range(1, 8):
            # tuple = (trials, response, rewards)
            tuples = get_trials(name, index, date_range=date_range, return_rewards=return_rewards)
            self.data.append(tuples)
        all = get_trials(name, date_range=date_range, return_rewards=return_rewards)
        self.count = len(all[0])
        self.bias = 0.5 - all[1].sum() / len(all[0])
        self.data = [all] + self.data

    def __repr__(self):
        return f"Monkey({self.shortname}, n={self.count:,}, bias={self.bias:.3f})"

    def get_data(self, lottery, n=0, n_last=None, return_rewards=False):
        if return_rewards:
            data = self.data[lottery]
        else:
            data = self.data[lottery][:2]  # only get trials and responses
        if n_last is not None:
            data = [d[-n_last:] for d in data]

        if n < 1:
            return data
        I = np.random.randint(0, len(data[0]), n)
        return [d[I] for d in data]


monkeys = []
print("Get monkey data", end="", flush=True)
for name in subject_ids:
    print(".", end="", flush=True)
    monkey = Monkey(name)
    monkeys.append(monkey)
print(" done!")



if __name__ == "__main__":
    print_n_trials_per_monkey()





