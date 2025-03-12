## Content 
We developed a Python framework to study risk-taking behavior in a group of monkeys during gambling tasks. This framework enables fitting various Prospect Theory (PT) models, based on Kahneman and Tversky (1979), to a dataset of monkey behavioral responses. It allows for comparisons between observed behavior and rational decision-making models. Additionally, we analyze the social hierarchy—determined by each monkey’s Elo score—to explore potential correlations and interactions between decision-making behavior and social ranking.


## Dependencies

- numpy
- matplotlib
- pandas
- scipy
- pickle
- prettytable
- pickle
- itertools
- collections


## Steps to run the analysis


- Define time periods for each monkey: Use the function .. to create distinct time periods per monkey, ensuring that each period contains at least 1,500 trials and compute their associated mean elo-score (scoial hierachy ranking) per period.

- Run `built_table_dynamic_per_period()` in `hierarchy_vs_PT.py`. This function will:
1)  Retrieve different Elo-rating periods and the processed dataset containing all monkey experiments.
2) Fit the dataset for each period across different Prospect Theory players.
3) Generate a comprehensive table where each row corresponds to a monkey, a specific period, and the fitted parameters per player.

- Verify Prospect Theory Predictions: Use the generated table to evaluate whether the monkeys' behavior aligns with Prospect Theory.

- Analyze Social Hierarchy & Prospect Theory Parameters in R: Investigate potential relationships between social hierarchy and Prospect Theory parameters using R for statistical analysis.


## Jupyter notebooks

The notebooks primarily focus on cleaning and analyzing the dataset that compiles all monkey data from the economic risky decision-making task.

- `00-common`: gather common functions used in other notenook. 
`01-preprocessing`: read and process the original dataset to ensure that tasks are named properly. The original dataset is untouched and the processed dataset is saved using an alternative filename.
- `02-dataset-analysis`:  allows to have a first overview of the cleaned datset
- `03-task-analysis`:  allows to have a first overview of the cleaned datset
- `04-decision-fit`: fit subjects response to measure their willingness to choose one option over the other (sigmoid fit).
- `05-bias-analysis`: plot figures related to the bias analysis. 
- `06-prospect-fit`: fit the models and save in json files. 


## Python files 

The python files are used to proceed to the different fits of the monkey data on the different decision-making model under uncertainty (Prospect Theory, or rational sigmoid models). 


`monkey.py`: This script processes experimental data on monkeys performing an economic decision-making task. It creates the class Monkey, loads trial data from a CSV file, filters and extracts relevant trials based on specified conditions (e.g., subject, task, and date), and structures the data for further analysis.

`lottery.py`: This script generates different sets of lotteries (L1 to L7) based on specific probability (p) and value (v) conditions. 
L0 represents the complete set of all lotteries. The script also provides a function to generate random trials from a selected lottery.

`player.py`: This script defines all the player classes that model decision-making behavior under uncertainty (Prospect Theory and Rationnal behavior). 

`monkey_analysis.py`: This script fits and evaluates the different player classes (Prospect Theory models) on monkeys' economic decision-making data. The models are trained on different lottery conditions and tested for their predictive accuracy.

`bias_analysis.py`: This script contains functions for analyzing monkey bias in decision-making tasks. It includes functions to compute bias, visualize bias vs. performance scores, compare score differences, and generate related plots.

`fit-precision.py`: This script contain function for creating figures related to the fit-precision (explanation of the need of 1500 trials minimum per fit).


`hierarchy_vs_PT.py`: This script processes and analyzes Elo scores and reaction times (RT) for monkeys in different experimental settings. It includes functions to compute and visualize the relationship between Elo scores and RTs across various periods
(static, dynamic, and best trials).







