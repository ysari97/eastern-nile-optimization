# Importing external packages

import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib as mpl
import math
import seaborn as sns
import pandas as pd
import numpy as np
import statistics
from collections import defaultdict
from tqdm import tqdm
from pandas.plotting import parallel_coordinates
from matplotlib.lines import Line2D
from ema_workbench.analysis import feature_scoring
from ema_workbench.analysis import dimensional_stacking
import importlib
import warnings

warnings.filterwarnings(action="once")

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Get the grandparent directory using relative path notation
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Add the grandparent directory to the sys path
sys.path.append(parent_dir)

from model.model_nile import ModelNile
# from experimentation.data_generation import generate_input_data
import plotter

import matplotlib
import matplotlib.font_manager as fm

# sns.set(font_scale=1.2)

fm.fontManager.addfont("Minion Pro Regular.ttf")
matplotlib.rc("font", family="Minion Pro")

from matplotlib import rcParams

rcParams["font.family"] = "Minion Pro"
rcParams["font.sans-serif"] = "Minion Pro"
rcParams["font.size"] = 12

experiments = pd.read_csv("../outputs/experiments_exploration.csv").drop(
    columns="Unnamed: 0"
)
outcomes = pd.read_csv("../outputs/outcomes_exploration.csv").drop(columns="Unnamed: 0")


outcomes["policy"] = experiments["policy"]
outcomes["scenario"] = experiments["scenario"]

directions = ["min", "max", "min", "max"]
best_performer_df = pd.DataFrame(range(5000), columns=["Scenario"])
best_performer_df["mean_factor"] = experiments["mean_factor"][:5000]
best_performer_df["stdev_factor"] = experiments["stdev_factor"][:5000]

performers = defaultdict(list)
improvements = defaultdict(list)
for scen in tqdm(range(5000)):
    scen_df = outcomes.loc[outcomes.scenario == scen].copy()
    for i, obj in enumerate(outcomes.iloc[:, :-2].columns):
        best = scen_df.sort_values(
            by=[obj], ascending=(directions[i] == "min")
        ).iloc[0][obj]
        average = np.mean(scen_df[obj])
        policy = list(scen_df.loc[scen_df[obj] == best, "policy"])[0]
        improvement = abs(best - average)
        performers[obj].append(policy)
        improvements[obj].append(improvement)

for obj in outcomes.iloc[:, :-2].columns:
    best_performer_df[obj] = performers[obj]
    best_performer_df[f"{obj}_improvement"] = improvements[obj]

policy_colors = {
    "Best Egypt Irrigation Deficit": "#fdaa09",
    "Best Egypt Irrigation": "#fdaa09",
    "Best Egypt Minimum HAD Level": "#0195fb",
    "Best Ethiopia Hydroenergy": "#41ab5d",
    "Best Ethiopia Hydropower": "#41ab5d",
    "Compromise: Percentile Threshold": "#6C0C86",
    "Compromise: Absolute Threshold": "orchid",
    "All Solutions": "#bdbdbd"
}
fancy_objective_names = {
    "egypt_def": "Egypt Irrigation Deficit",
    "min_HAD": "Egypt Minimum HAD Level",
    "sudan_def": "Sudan Irrigation Deficit",
    "ethiopia_hydro": "Ethiopia Hydroenergy"
}
letters=['a', 'b', 'c', 'd']

fig, ax = plt.subplots(nrows=2, ncols=2,figsize=(16, 9))

for i, obj in enumerate(outcomes.iloc[:, :-2].columns):
    print("------------")
    print(obj)
    print("------------\n")

    legend_par = False
    if i ==3: legend_par = True

    my_data = best_performer_df[best_performer_df[f"{obj}_improvement"] > 0]

    category_order = my_data[obj].value_counts().index[::-1]
    
    sns.kdeplot(
        data=my_data,
        x="mean_factor",
        multiple="fill",
        hue=obj,
        palette=policy_colors,
        hue_order=category_order,
        ax=ax[i//2][i%2],
        legend=False
    )


    ax[i//2][i%2].yaxis.label.set_size(20)
    ax[i//2][i%2].xaxis.label.set_size(20)
    ax[i//2][i%2].set_xlabel("Mean streamflow multiplier", fontsize=14)
    ax[i//2][i%2].set_ylabel("Density", fontsize=14)
    ax[i//2][i%2].set_title(f"{letters[i]}) {fancy_objective_names[obj]}", fontsize=18)

handles = [
    mpl.patches.Patch(facecolor=color, label=name) for name, color in policy_colors.items() if name not in ["All Solutions", "Best Egypt Irrigation", "Best Ethiopia Hydropower"]
]

fig.legend(handles=handles, loc = "lower center", ncol=5, mode= "expand")

        
fig.tight_layout(pad=2.8)
plt.savefig(
        "stacked_density_against_mean_streamflow.pdf",
        bbox_inches="tight",
    )

# plt.show()