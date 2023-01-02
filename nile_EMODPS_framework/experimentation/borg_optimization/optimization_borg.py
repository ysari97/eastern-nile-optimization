# Script for baseline optimization

import numpy as np
import os
import pandas as pd
import sys
import random

from datetime import datetime

from borg import *

module_path = os.path.abspath(os.path.join("../../model"))
if module_path not in sys.path:
    sys.path.append(module_path)
from model_nile import ModelNile
from model_wrapper import nile_wrapper

# set max time in hours
maxtime = 16

random_seed = 10

Configuration.seed(random_seed)

# print("BEFORE STARTMPI", flush=True)
# need to start up MPI first
Configuration.startMPI()

# print("AFTER STARTMPI", flush=True)

nile_model = ModelNile()
total_parameter_count = nile_model.overarching_policy.get_total_parameter_count()
release_parameter_count = nile_model.overarching_policy.functions["release"].get_free_parameter_number()
n_inputs_release = nile_model.overarching_policy.functions["release"].n_inputs
n_outputs_release = nile_model.overarching_policy.functions["release"].n_outputs
RBF_count = nile_model.overarching_policy.functions["release"].RBF_count
p_per_RBF = 2 * n_inputs_release + n_outputs_release

# Since we first introduce the release policy to the model, first parameters
# belong to the RBFs. Only then, the parameters of hedging functions.
# Let's first put the levers for the release policy.

lever_list = list()
for i in range(release_parameter_count):
    modulus = (i - n_outputs_release) % p_per_RBF
    if (
        (i >= n_outputs_release)
        and (modulus < (p_per_RBF - n_outputs_release))
        and (modulus % 2 == 0)
    ):  # centers:
        lever_list.append([-1, 1])
    else:  # linear parameters for each release, radii and weights of RBFs:
        lever_list.append([0, 1])

for j in range(release_parameter_count, total_parameter_count):
    lever_list.append([0, 1])

# create an instance of Borg with the Nile problem
borg = Borg(total_parameter_count, 4, 0, nile_wrapper, directions=[0,1,0,1])

borg.setBounds(*lever_list)
borg.setEpsilons(0.1, 0.1, 0.1, 0.1)

# perform the optimization
nfes = 250000
result = borg.solveMPI(maxTime=maxtime, maxEvaluations=nfes, runtime=f"runtime/runtime_seed_{random_seed}.txt".encode('utf-8'))

# shut down MPI
Configuration.stopMPI()

# only the master node returns a result
# print the objectives to output
solution_list = []
objectives_list = []
# print(type(result), flush=True)
if result:
    for solution in result:
        solution_list.append(solution.getVariables())
        objectives_list.append(solution.getObjectives())
        # print(solution.getObjectives(), flush=True)

# print(objectives_list, flush=True)
if len(solution_list) > 0:
    d_vars = pd.DataFrame(solution_list, columns=[f"v{i}" for i in range(total_parameter_count)])
    objs = pd.DataFrame(objectives_list, columns=["Egypt_irr_def", "HAD_min_level", "Sudan_irr_def", "Ethiopia_hydroenergy"])

    output_directory = "../../outputs/"

    d_vars.to_csv(f"{output_directory}baseline_opt_dvs.csv")
    objs.to_csv(f"{output_directory}baseline_opt_objs.csv")
