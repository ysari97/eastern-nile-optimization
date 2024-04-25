import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Change the font type of matplotlib figures to make it match with the report
import matplotlib
import matplotlib.font_manager as fm

# sns.set(font_scale=1.2)

fm.fontManager.addfont("Minion Pro Regular.ttf")
matplotlib.rc("font", family="Minion Pro")
from matplotlib import rcParams

rcParams["font.family"] = "Minion Pro"
rcParams["font.sans-serif"] = "Minion Pro"
rcParams["font.size"] = 12
from tqdm import tqdm

from model.model_nile import ModelNile
import pickle

policies = pd.read_csv("../output_analysis/merged_dvs.csv")
objectives = pd.read_csv("../output_analysis/merged_objectives.csv")

solution_indices = [973, 93, 1065, 688, 514]
solution_names = [
    "Best Egypt Irrigation Deficit",
    "Best Egypt Minimum HAD Level",
    "Best Ethiopia Hydroenergy",
    "Compromise: Percentile Threshold",
    "Compromise: Absolute Threshold",
]

solutions = [policies.loc[i] for i in solution_indices]
running_models = dict()
for i, sol_name in tqdm(enumerate(solution_names)):
    new_model = ModelNile()
    full_return = new_model.evaluate(solutions[i])
    running_models[sol_name] = full_return

with open("data.pickle", "wb") as f:
    pickle.dump(running_models, f)

# create 3x1 subplots
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(24, 24), constrained_layout=True)
fig.suptitle(
    "a) GERD Level                                                                                        b) HAD Level",
    fontsize=36,
)

# clear subplots
for ax in axs:
    ax.remove()

# add subfigure per subplot
gridspec = axs[0].get_subplotspec().get_gridspec()
subfigs = [fig.add_subfigure(gs) for gs in gridspec]

policies = [
    "Best Egypt Irrigation Deficit",
    "Compromise: Percentile Threshold",
    "Best Ethiopia Hydroenergy",
]

policy_colors = {
    "Best Egypt Irrigation": "#fdaa09",
    "Best Egypt Irrigation Deficit": "#fdaa09",
    "Best Egypt Minimum HAD Level": "#0195fb",
    "Best Ethiopia Hydropower": "#41ab5d",
    "Compromise: Percentile Threshold": "#6C0C86",
    "Compromise: Absolute Threshold": "orchid",
    "All Solutions": "#bdbdbd",
    "Best Ethiopia Hydroenergy": "#41ab5d",
}

dam_plot_parameters = [
    ("GERD", 590, 640),
    ("HAD", 147, 185),
]

for row, subfig in enumerate(subfigs):
    subfig.suptitle(f"{policies[row]} Policy", fontsize=32)

    # create 1x2 subplots per subfig
    axs = subfig.subplots(nrows=1, ncols=2)

    for column, ax in enumerate(axs):
        dam, mini, maxi = dam_plot_parameters[column]
        dam_color = policy_colors[policies[row]]

        for key, value in running_models[policies[row]][0].items():
            # print(key)
            if key == 0:
                ax.plot(
                    value[0].object_by_name(dam).level_vector,
                    color=dam_color,
                    label=f"{dam} Level",
                )
            else:
                ax.plot(value[0].object_by_name(dam).level_vector, color=dam_color)

        ax.set_xlabel("Time", fontsize=22)
        ax.set_ylabel("Level (masl)", fontsize=22)
        for i, label, color in [(mini, "Minimum Operating Level", "silver")]:
            ax.hlines(
                y=i,
                linewidth=3,
                xmin=0,
                xmax=240,
                label=label,
                color=color,
                linestyle=":",
            )

        ax.set_xticks(np.arange(0, 20 * 12 + 1, 4 * 12))
        ax.set_xticklabels(
            [f"Jan-{2023+i*4}" for i in range(int(20 / 4) + 1)],
            fontsize=14,
        )
        ax.tick_params(labelsize=18)

        ax.legend(fontsize=18)
        # ax.set_title(titles[row][column], fontsize= 20, fontweight='bold')

plt.savefig("overall_dams.pdf", bbox_inches="tight")
# plt.show()
