# Script for baseline optimization


import numpy as np
import os
import pandas as pd
import sys
import random

from datetime import datetime


from ema_workbench import RealParameter, ScalarOutcome, Constant, Model
from ema_workbench import MultiprocessingEvaluator, SequentialEvaluator, ema_logging, IpyparallelEvaluator
from ema_workbench.em_framework.optimization import (
    EpsilonProgress,
    ArchiveLogger,
)

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)
from model.model_nile import ModelNile

optimization_seed = 10

if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    output_directory = "../outputs/"
    nile_model = ModelNile()

    em_model = Model("NileProblem", function=nile_model)

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
            lever_list.append(RealParameter(f"v{i}", -1, 1))
        else:  # linear parameters for each release, radii and weights of RBFs:
            lever_list.append(RealParameter(f"v{i}", 0, 1))

    for j in range(release_parameter_count, total_parameter_count):
        lever_list.append(RealParameter(f"v{j}", 0, 1))

    em_model.levers = lever_list

    # specify outcomes
    em_model.outcomes = [
        ScalarOutcome("egypt_def", ScalarOutcome.MINIMIZE),
        ScalarOutcome("min_HAD", ScalarOutcome.MAXIMIZE),
        ScalarOutcome("sudan_def", ScalarOutcome.MINIMIZE),
        ScalarOutcome("ethiopia_hydro", ScalarOutcome.MAXIMIZE),
    ]

    convergence_metrics = [
        EpsilonProgress(),
        ArchiveLogger(
            f"{output_directory}archive_logs",
            [lever.name for lever in em_model.levers],
            [outcome.name for outcome in em_model.outcomes],
        ),
    ]

    nfe = 300000
    epsilon_list = [1e-1, 1e-1, 1e-1, 1e-1]

    random.seed(optimization_seed)
    before = datetime.now()

    with IpyparallelEvaluator(em_model) as evaluator:
        results, convergence = evaluator.optimize(
            nfe=nfe,
            searchover="levers",
            epsilons=epsilon_list,
            convergence_freq=500,
            convergence=convergence_metrics,
        )

    after = datetime.now()

    with open(f"{output_directory}time_counter.txt", "w") as f:
        f.write(
            f"It took {after-before} time to do {nfe} NFEs with epsilons: {epsilon_list}"
        )

    results.to_csv(f"{output_directory}baseline_results.csv")
    convergence.to_csv(f"{output_directory}baseline_convergence.csv")
