import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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


policy_colors = {
    "Best Egypt Irrigation Deficit": "#fdaa09",
    "Best Egypt Minimum HAD Level": "#0195fb",
    "Best Ethiopia Hydroenergy": "#41ab5d",
    "Compromise: Percentile Threshold": "#6C0C86",
    "Compromise: Absolute Threshold": "orchid",
    "All Solutions": "#bdbdbd",
}
norm_df = objectives.copy()
norm_df.columns = [
    "Egypt Irrigation Deficit",
    "Egypt Minimum HAD Level",
    "Sudan Irrigation Deficit",
    "Ethiopia Hydroenergy",
]

norm_df["Name"] = "All Solutions"
for i, solution_index in enumerate(solution_indices):
    norm_df.loc[solution_index, "Name"] = solution_names[i]
    # norm_df = norm_df.append(norm_df.loc[solution_index, :].copy())

g = sns.pairplot(
    norm_df,
    corner=True,
    kind="reg",
    hue="Name",
    diag_kind="kde",
    plot_kws={"line_kws": {"color": "salmon"}, "scatter_kws": {"s": 100}},
    palette=policy_colors,
)
handles = g._legend_data.values()
labels = g._legend_data.keys()
g._legend.remove()

order = [0, 4, 1, 5, 3, 2]
g.fig.legend(
    handles=[list(handles)[i] for i in order],
    labels=[list(labels)[i] for i in order],
    bbox_to_anchor=(0.75, 0.90),
)
#     for i in range(len(g.axes)-1):
#         g.axes[i,0].invert_yaxis()
#         g.axes[i,i].invert_xaxis()
g.axes[-1, 0].invert_yaxis()
g.axes[1, 0].invert_yaxis()
g.axes[-1, -1].invert_xaxis()
g.axes[-1, 1].invert_xaxis()
plt.savefig("pairplot.pdf", bbox_inches="tight")
# plt.show()