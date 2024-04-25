import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Change the font type of matplotlib figures to make it match with the report
import matplotlib
import matplotlib.font_manager as fm
import seaborn as sns

sns.set(font_scale=1.2)

fm.fontManager.addfont("Minion Pro Regular.ttf")
matplotlib.rc("font", family="Minion Pro")
from matplotlib import rcParams

rcParams["font.family"] = "Minion Pro"
rcParams["font.sans-serif"] = "Minion Pro"
rcParams["font.size"] = 12
from tqdm import tqdm

from model.model_nile import ModelNile

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


empty_list = []
policy_list = []

for policy in [
    "Best Egypt Irrigation Deficit",
    "Compromise: Percentile Threshold",
    "Best Ethiopia Hydroenergy",
]:
    for replication in range(30):
        releases = (
            running_models[policy][0][replication][0]
            .object_by_name("GERD")
            .release_vector
        )
        for i in range(12):
            empty_list.append(
                np.mean([release for j, release in enumerate(releases) if j % 12 == i])
            )
            policy_list.append(policy)

for replication in range(30):
    releases = (
        running_models[policy][0][replication][0].object_by_name("BlueNile").streamflow
    )
    for i in range(12):
        empty_list.append(
            np.mean([release for j, release in enumerate(releases) if j % 12 == i])
        )
        policy_list.append("No GERD")


my_dict = {"GERD Release": empty_list, "Policy": policy_list}
df = pd.DataFrame.from_dict(my_dict)
month_mod = [(i % 12) + 1 for i in df.index]
df["Month"] = month_mod

from matplotlib.patches import PathPatch


def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # setting new width of box
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


policy_colors = {
    "Best Egypt Irrigation Deficit": "#fdaa09",
    "Best Egypt Minimum HAD Level": "#0195fb",
    "Best Ethiopia Hydroenergy": "#41ab5d",
    "Compromise: Percentile Threshold": "#6C0C86",
    "Compromise: Absolute Threshold": "orchid",
    "All Solutions": "#bdbdbd",
    "No GERD": "#bdbdbd",
}
fig, ax = plt.subplots(figsize=(18, 12))
sns.boxplot(
    data=df,
    x="Month",
    y="GERD Release",
    hue="Policy",
    orient="v",
    whis=1.5,
    fliersize=2,
    palette=policy_colors,
)
adjust_box_widths(fig, 0.8)
month_list = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
ax.set_xticklabels(month_list)
ax.set_ylabel("Release (m3/s)", fontsize=18)
ax.set_xlabel("Month", fontsize=18)
ax.vlines(
    x=[i + 0.5 for i in range(-1, 12)],
    ymin=0,
    ymax=6000,
    linestyles="dotted",
    colors="black",
)

ax.legend(fontsize=18)
plt.savefig("GERD_release_boxplots.pdf", bbox_inches="tight")
# plt.show()
