import matplotlib.pyplot as plt
import pandas as pd
import plotter
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

plotter.parallel_plots_many_policies(
    objectives,
    solution_indices=solution_indices,
    solution_names=solution_names,
)

plt.savefig("parallel_plot.pdf", bbox_inches="tight")
# plt.show()